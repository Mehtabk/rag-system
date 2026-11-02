from rag.chunking.strategies import RecursiveChunker
from rag.types import Document


def test_recursive_chunker_splits_and_tags_provenance():
    doc = Document(doc_id="d1", text="\n\n".join(f"Para {i} " * 40 for i in range(5)),
                   source="d1")
    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=50).split(doc)
    assert len(chunks) > 1
    assert all(c.source == "d1" and c.doc_id == "d1" for c in chunks)
    assert all(c.chunk_id.startswith("d1#") for c in chunks)


def test_empty_document_yields_no_chunks():
    assert RecursiveChunker().split(Document(doc_id="e", text="   ", source="e")) == []
