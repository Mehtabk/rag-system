"""End-to-end: index the sample corpus, then query through the full pipeline
and the FastAPI app.
"""

from fastapi.testclient import TestClient

from rag.api.app import app
from rag.config import load_config
from rag.pipelines.index_pipeline import run_index
from rag.pipelines.query_pipeline import QueryPipeline


def test_index_then_query_pipeline():
    cfg = load_config()
    n = run_index(cfg)
    assert n > 0
    ans = QueryPipeline(cfg).answer("What is hybrid retrieval?")
    assert ans.sources, "expected at least one retrieved source"
    # The default HashingEmbedder is a deterministic placeholder, not a semantic
    # model, so exact rank order isn't guaranteed — assert the correct source is
    # retrieved (present in the top results), which is what recall measures.
    retrieved_sources = {s.chunk.source for s in ans.sources}
    assert "hybrid_retrieval.md" in retrieved_sources
    assert ans.blocked is False


def test_api_query_endpoint():
    run_index()
    client = TestClient(app)
    r = client.post("/query", json={"question": "What is hybrid retrieval?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "sources" in body
    assert any(s["source"] == "hybrid_retrieval.md" for s in body["sources"])


def test_api_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
