"""Vector store interface + a dependency-free in-memory implementation.

The in-memory store persists to a JSON file under data/ so the index survives
between `make ingest` and `make serve`. Swap to Pinecone / pgvector / Qdrant
by implementing this same interface under providers/.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Protocol

from rag.types import Chunk, ScoredChunk

logger = logging.getLogger("rag.vectorstore")


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def vector_search(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Brute-force cosine search over an in-memory list, JSON-persisted."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupted/tampered index must not crash startup. Start empty
            # and log; a reindex rebuilds it cleanly.
            logger.warning("Could not read index at %s (%s); starting empty.", self.path, exc)
            return
        if not isinstance(data, list):
            logger.warning("Index at %s is malformed (expected a list); starting empty.", self.path)
            return
        chunks: list[Chunk] = []
        for d in data:
            if not isinstance(d, dict) or "chunk_id" not in d or "text" not in d:
                continue  # skip malformed records rather than trusting them
            chunks.append(
                Chunk(
                    chunk_id=str(d["chunk_id"]),
                    text=str(d["text"]),
                    source=str(d.get("source", "")),
                    doc_id=str(d.get("doc_id", "")),
                    metadata=d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {},
                    embedding=d.get("embedding"),
                )
            )
        self._chunks = chunks

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "doc_id": c.doc_id,
                "metadata": c.metadata,
                "embedding": c.embedding,
            }
            for c in self._chunks
        ]
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def upsert(self, chunks: list[Chunk]) -> None:
        by_id = {c.chunk_id: c for c in self._chunks}
        for c in chunks:
            by_id[c.chunk_id] = c
        self._chunks = list(by_id.values())
        self._persist()

    def vector_search(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=c, score=_cosine(query_embedding, c.embedding or []))
            for c in self._chunks
            if c.embedding
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


def get_vectorstore(cfg) -> VectorStore:
    """Factory: select a vector store implementation from config."""
    provider = cfg.vectorstore.provider
    if provider == "memory":
        path = Path(cfg._repo_root) / cfg.paths.data_dir / "index" / "memory_store.json"
        return InMemoryVectorStore(path=path)
    # if provider == "pinecone":
    #     from rag.vectorstore.providers.pinecone_store import PineconeStore
    #     return PineconeStore(...)
    raise ValueError(f"Unknown vectorstore provider: {provider!r}")
