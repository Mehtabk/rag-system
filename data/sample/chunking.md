Chunking breaks large documents into retrieval-friendly sections before embedding.

Most RAG quality problems trace back to the chunking and ingestion layer rather than the language model. If chunks are too large they dilute relevance; if too small they lose context. A recursive chunker splits on natural boundaries like paragraphs and keeps a small overlap between adjacent chunks so that context is not cut mid-thought.

Each chunk should carry metadata about its source, section, and access permissions so retrieval can filter and cite correctly.
