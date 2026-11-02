"""Minimal span timer. Replace with OpenTelemetry spans in production so each
stage (retrieve, rerank, generate) is independently measurable.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("rag.trace")


@contextmanager
def span(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("span %s took %.1f ms", name, elapsed_ms)
