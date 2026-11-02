"""Reranking stage. Default is an identity passthrough; real systems plug in a
cross-encoder (e.g. Cohere Rerank, bge-reranker) that re-scores each
(query, chunk) pair far more accurately than first-stage retrieval.
"""

from __future__ import annotations

from typing import Protocol

from rag.types import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


class IdentityReranker:
    """Keeps retrieval order; just truncates to top_n. Placeholder."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return chunks[:top_n]


def get_reranker(cfg) -> Reranker:
    provider = cfg.retrieval.rerank_provider
    if provider == "identity":
        return IdentityReranker()
    # if provider == "cohere":
    #     from rag.retrieval.providers.cohere_reranker import CohereReranker
    #     return CohereReranker(...)
    raise ValueError(f"Unknown rerank provider: {provider!r}")
