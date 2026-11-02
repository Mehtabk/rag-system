"""Hybrid retrieval: fuse dense (vector) and sparse (keyword/BM25-ish) scores.

Hybrid search is now the production baseline, not an optimization — pure vector
search misses exact-term matches, pure keyword misses paraphrases. We combine
both with a configurable alpha.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag.embeddings import Embedder
from rag.types import ScoredChunk
from rag.vectorstore import VectorStore

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, alpha: float = 0.5):
        self.store = store
        self.embedder = embedder
        self.alpha = alpha  # weight on the dense (vector) component

    def _keyword_scores(self, query: str, pool) -> dict[str, float]:
        """Lightweight TF-IDF-ish keyword scoring over the candidate pool."""
        q_terms = set(_tokenize(query))
        n = len(pool) or 1
        df = Counter()
        for sc in pool:
            for term in set(_tokenize(sc.chunk.text)):
                df[term] += 1
        scores: dict[str, float] = {}
        for sc in pool:
            tf = Counter(_tokenize(sc.chunk.text))
            s = 0.0
            for term in q_terms:
                if term in tf:
                    idf = math.log((n + 1) / (df[term] + 0.5))
                    s += tf[term] * idf
            scores[sc.chunk.chunk_id] = s
        return scores

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        # Dense: pull a candidate pool wider than top_k so keyword can re-weight.
        q_emb = self.embedder.embed([query])[0]
        pool = self.store.vector_search(q_emb, top_k=max(top_k * 3, top_k))
        if not pool:
            return []

        dense = _minmax({sc.chunk.chunk_id: sc.score for sc in pool})
        sparse = _minmax(self._keyword_scores(query, pool))

        fused: list[ScoredChunk] = []
        for sc in pool:
            cid = sc.chunk.chunk_id
            score = self.alpha * dense.get(cid, 0.0) + (1 - self.alpha) * sparse.get(cid, 0.0)
            fused.append(ScoredChunk(chunk=sc.chunk, score=score))
        fused.sort(key=lambda s: s.score, reverse=True)
        return fused[:top_k]
