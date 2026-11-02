"""Vector store: persist + search chunk embeddings (data path write, request path read)."""

from rag.vectorstore.base import VectorStore, get_vectorstore

__all__ = ["VectorStore", "get_vectorstore"]
