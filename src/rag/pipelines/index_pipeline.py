"""DATA PATH: ingest -> chunk -> embed -> write.

Run offline/asynchronously, decoupled from serving. `python -m
rag.pipelines.index_pipeline` indexes everything under the configured data dir
(falls back to the bundled sample corpus if none is present).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.chunking.strategies import RecursiveChunker
from rag.config import load_config
from rag.embeddings import get_embedder
from rag.ingestion.loaders import DirectoryLoader
from rag.types import Chunk
from rag.vectorstore import get_vectorstore

logger = logging.getLogger("rag.index")


def run_index(cfg=None) -> int:
    cfg = cfg or load_config()
    data_dir = Path(cfg._repo_root) / cfg.paths.data_dir / "raw"
    sample_dir = Path(cfg._repo_root) / "data" / "sample"

    source_dir = data_dir if data_dir.exists() else sample_dir
    logger.info("Loading documents from %s", source_dir)

    docs = DirectoryLoader(source_dir).load()
    if not docs:
        logger.warning("No documents found in %s", source_dir)
        return 0

    chunker = RecursiveChunker(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.split(doc))

    embedder = get_embedder(cfg)
    vectors = embedder.embed([c.text for c in chunks])
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    store = get_vectorstore(cfg)
    store.upsert(chunks)
    logger.info("Indexed %d chunks from %d documents", len(chunks), len(docs))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    n = run_index()
    print(f"Indexed {n} chunks.")
