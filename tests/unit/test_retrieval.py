from rag.embeddings.embedder import HashingEmbedder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.router import SimpleRouter
from rag.types import Chunk


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def vector_search(self, q, top_k):
        from rag.types import ScoredChunk
        return [ScoredChunk(c, 1.0) for c in self._chunks[:top_k]]

    def all_chunks(self):
        return self._chunks

    def upsert(self, chunks):
        self._chunks = chunks


def _chunk(cid, text):
    emb = HashingEmbedder(32).embed([text])[0]
    return Chunk(cid, text, f"{cid}.md", cid, embedding=emb)


def test_hybrid_retriever_ranks_keyword_match_high():
    emb = HashingEmbedder(32)
    chunks = [_chunk("a", "hybrid retrieval fuses vector and keyword search"),
              _chunk("b", "cats are small mammals that enjoy sleeping")]
    r = HybridRetriever(_FakeStore(chunks), emb, alpha=0.5)
    out = r.retrieve("what is hybrid keyword retrieval", top_k=2)
    assert out[0].chunk.chunk_id == "a"


def test_router_short_circuits_greetings():
    assert SimpleRouter().route("hello").needs_retrieval is False
    assert SimpleRouter().route("what is RAG").needs_retrieval is True
