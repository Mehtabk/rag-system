Hybrid retrieval combines dense vector search with sparse keyword search such as BM25.

Dense vector search captures semantic similarity and handles paraphrases, while keyword search captures exact term matches that embeddings often miss. A production RAG system fuses both signals, typically with a weighting factor alpha, because neither method alone is reliable across all queries.

After hybrid retrieval returns a candidate pool, a cross-encoder reranker re-scores each query-chunk pair to push the most relevant passages to the top before generation.
