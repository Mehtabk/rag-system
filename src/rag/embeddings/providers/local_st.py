"""Local sentence-transformers embedder — real semantic embeddings, no API keys.

Runs fully offline after the model is downloaded once (cached under
~/.cache/huggingface). Default model `all-MiniLM-L6-v2` is small (~90MB), fast
on CPU, and produces 384-dim vectors with genuine semantic similarity — a large
upgrade over the HashingEmbedder placeholder.

Install the optional dependency:  pip install -e ".[local]"
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported lazily so the package stays importable without the heavy dep.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LocalSentenceTransformerEmbedder:
    """Wraps a sentence-transformers model behind the Embedder protocol."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model
        self.dim = dim
        self._model = _load_model(model)
        # Trust the model's real dimensionality over config if they disagree.
        # Method was renamed across sentence-transformers versions; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        actual = get_dim() if callable(get_dim) else None
        if actual:
            self.dim = actual

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # unit vectors -> cosine == dot product
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]
