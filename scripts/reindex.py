"""Convenience: wipe the local index and rebuild from the configured data dir."""

from pathlib import Path

from rag.config import load_config
from rag.pipelines.index_pipeline import run_index

if __name__ == "__main__":
    cfg = load_config()
    index_file = Path(cfg._repo_root) / cfg.paths.data_dir / "index" / "memory_store.json"
    if index_file.exists():
        index_file.unlink()
        print(f"Removed {index_file}")
    n = run_index(cfg)
    print(f"Reindexed {n} chunks.")
