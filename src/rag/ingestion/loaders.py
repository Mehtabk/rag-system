"""Source loaders. The default loads .txt/.md files from a directory.

Real systems add PDF (layout-aware), HTML, CSV, and connector loaders
(Confluence, S3, Google Drive) behind this same simple contract:
`load() -> list[Document]`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from rag.types import Document

logger = logging.getLogger("rag.ingestion")


class DirectoryLoader:
    """Load plain-text and markdown files from a directory tree."""

    SUFFIXES: ClassVar[set[str]] = {".txt", ".md"}
    # Skip pathologically large files to bound memory use during ingestion.
    MAX_FILE_BYTES: ClassVar[int] = 10 * 1024 * 1024  # 10 MB

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def load(self) -> list[Document]:
        docs: list[Document] = []
        if not self.root.exists():
            return docs
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in self.SUFFIXES or not path.is_file():
                continue
            resolved = path.resolve()
            # Defense-in-depth: never follow a symlink that escapes the root.
            if not str(resolved).startswith(str(self.root)):
                logger.warning("Skipping %s: resolves outside ingestion root", path)
                continue
            if resolved.stat().st_size > self.MAX_FILE_BYTES:
                logger.warning("Skipping %s: exceeds %d bytes", path, self.MAX_FILE_BYTES)
                continue
            text = resolved.read_text(encoding="utf-8", errors="ignore")
            rel = resolved.relative_to(self.root).as_posix()
            docs.append(
                Document(
                    doc_id=rel,
                    text=text,
                    source=rel,
                    metadata={"path": str(resolved)},
                )
            )
        return docs
