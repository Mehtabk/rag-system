"""Routes: /health, /query. Ingestion is triggered out-of-band (data path),
not via the request path, but a thin /ingest hook is provided for convenience.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.schemas import QueryRequest, QueryResponse, SourceOut
from rag.api.security import require_admin
from rag.pipelines.index_pipeline import run_index
from rag.pipelines.query_pipeline import QueryPipeline

router = APIRouter()

# Build the query pipeline once and reuse it across requests.
_pipeline: QueryPipeline | None = None


def get_pipeline() -> QueryPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
    return _pipeline


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    ans = get_pipeline().answer(req.question)
    return QueryResponse(
        answer=ans.text,
        grounding_score=ans.grounding_score,
        blocked=ans.blocked,
        reason=ans.reason,
        sources=[
            SourceOut(
                source=s.chunk.source,
                score=round(s.score, 4),
                snippet=s.chunk.text[:200],
            )
            for s in ans.sources
        ],
    )


@router.post("/ingest", dependencies=[Depends(require_admin)])
def ingest() -> dict:
    """Privileged: rebuild the index. Requires X-Admin-Token (see settings).

    Ingestion is a data-path operation; exposing it on the request path is a
    convenience, gated behind admin auth so it can't be triggered anonymously.
    """
    n = run_index()
    # Rebuild the pipeline so it picks up the fresh index.
    global _pipeline
    _pipeline = None
    return {"indexed_chunks": n}
