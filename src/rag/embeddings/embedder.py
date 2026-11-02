"""Provider-agnostic embedder interface + a dependency-free default.

The default `HashingEmbedder` produces deterministic vectors with no network
calls or API keys, so the skeleton runs out of the box. Swap to a real model
(OpenAI, bge, e5) by adding a provider and pointing config at it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    """Any embedder must turn a list of texts into a list of vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedding. No deps, no keys.

    Not semantically strong — it exists so the pipeline is runnable and
    testable end-to-end. Replace with a real embedding model in production.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


def get_embedder(cfg) -> Embedder:
    """Factory: select an embedder implementation from config."""
    provider = cfg.embeddings.provider
    if provider == "hashing":
        return HashingEmbedder(dim=cfg.embeddings.dim)
    if provider == "local":
        from rag.embeddings.providers.local_st import LocalSentenceTransformerEmbedder

        return LocalSentenceTransformerEmbedder(model=cfg.embeddings.model, dim=cfg.embeddings.dim)
    # if provider == "openai":
    #     from rag.embeddings.providers.openai_embedder import OpenAIEmbedder
    #     return OpenAIEmbedder(model=cfg.embeddings.model, dim=cfg.embeddings.dim)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")
