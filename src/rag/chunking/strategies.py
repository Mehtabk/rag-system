"""Chunking strategies. Default: recursive character splitter with overlap.

~80% of RAG quality problems trace back to this layer, so it is deliberately
isolated and swappable (recursive | semantic | layout-based).
"""

from __future__ import annotations

from rag.types import Chunk, Document


class RecursiveChunker:
    """Split on paragraph boundaries, packing up to chunk_size with overlap."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, doc: Document) -> list[Chunk]:
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        buf = ""
        idx = 0

        def flush(text: str):
            nonlocal idx
            text = text.strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{idx}",
                    text=text,
                    source=doc.source,
                    doc_id=doc.doc_id,
                    metadata=dict(doc.metadata),
                )
            )
            idx += 1

        for para in paragraphs:
            if len(buf) + len(para) + 2 <= self.chunk_size:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                flush(buf)
                # carry overlap from the tail of the previous buffer
                tail = buf[-self.chunk_overlap :] if self.chunk_overlap else ""
                buf = f"{tail}\n\n{para}" if tail else para
        flush(buf)
        return chunks
