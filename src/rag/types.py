"""Shared data types that flow between stages.

Keeping these in one place lets every stage depend on stable contracts rather
than on each other's internals — the key to swappable, independently testable
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A raw source document, before chunking."""

    doc_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval-sized piece of a document, with provenance metadata."""

    chunk_id: str
    text: str
    source: str
    doc_id: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class ScoredChunk:
    """A chunk returned by retrieval/reranking, with a relevance score."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """The final generated answer plus the chunks it was grounded on."""

    text: str
    sources: list[ScoredChunk]
    grounding_score: float = 0.0
    blocked: bool = False
    reason: str | None = None
