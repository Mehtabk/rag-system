"""Loads config/config.yaml into a typed, attribute-accessible object.

A single load point means tuning parameters (chunk size, top_k, providers)
live in YAML, never scattered through code.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import yaml

# Repo root = three parents up from this file: src/rag/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _to_namespace(obj):
    """Recursively convert dicts to attribute-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SimpleNamespace:
    """Load and cache the YAML config as a namespace (cfg.retrieval.top_k, ...)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ns = _to_namespace(raw)
    ns._raw = raw  # keep the dict around for anything that wants it
    ns._repo_root = REPO_ROOT
    return ns
