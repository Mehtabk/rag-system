"""Stage-attributable RAG metrics.

Keeping retrieval metrics (recall@k, MRR) separate from answer metrics
(groundedness, must-include) lets you attribute a regression to the right
stage — the core principle behind a usable eval suite.
"""

from __future__ import annotations

from rag.types import ScoredChunk


def retrieval_recall_at_k(retrieved: list[ScoredChunk], expected_sources: list[str]) -> float:
    """Fraction of expected sources that appear anywhere in the retrieved set."""
    if not expected_sources:
        return 1.0
    got = {s.chunk.source for s in retrieved}
    hits = sum(1 for src in expected_sources if src in got)
    return hits / len(expected_sources)


def reciprocal_rank(retrieved: list[ScoredChunk], expected_sources: list[str]) -> float:
    """1/rank of the first correct source (MRR component)."""
    for i, s in enumerate(retrieved, start=1):
        if s.chunk.source in expected_sources:
            return 1.0 / i
    return 0.0


def must_include_score(answer: str, must_include: list[str]) -> float:
    """Fraction of required terms present in the answer (answer-quality proxy)."""
    if not must_include:
        return 1.0
    low = answer.lower()
    hits = sum(1 for term in must_include if term.lower() in low)
    return hits / len(must_include)
