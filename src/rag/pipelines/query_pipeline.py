"""REQUEST PATH: route -> input guard -> retrieve -> rerank -> generate -> output guard.

A single `QueryPipeline` object is constructed once (reused across requests) and
exposes `.answer(question) -> Answer`. Each stage is injected, so any piece can
be swapped or mocked in tests.
"""

from __future__ import annotations

import logging

from rag.config import load_config
from rag.embeddings import get_embedder
from rag.generation.llm import build_prompt, get_llm, load_prompt_template
from rag.guardrails.input import InputGuardrail
from rag.guardrails.output import OutputGuardrail
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import get_reranker
from rag.retrieval.router import SimpleRouter
from rag.types import Answer
from rag.vectorstore import get_vectorstore

logger = logging.getLogger("rag.query")


class QueryPipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.router = SimpleRouter()
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail(
            min_grounding_score=self.cfg.guardrails.min_grounding_score
        )
        embedder = get_embedder(self.cfg)
        store = get_vectorstore(self.cfg)
        self.retriever = HybridRetriever(
            store=store, embedder=embedder, alpha=self.cfg.retrieval.hybrid_alpha
        )
        self.reranker = get_reranker(self.cfg)
        self.llm = get_llm(self.cfg)
        self.prompt_template = load_prompt_template(self.cfg)

    def answer(self, question: str) -> Answer:
        # 1. route
        route = self.router.route(question)
        if not route.needs_retrieval:
            return Answer(text="Hello! Ask me a question about the indexed documents.",
                          sources=[], reason=route.reason)

        # 2. input guardrail
        if self.cfg.guardrails.check_input:
            verdict = self.input_guard.check(question)
            if not verdict.allowed:
                return Answer(text="This request was blocked.", sources=[],
                              blocked=True, reason=verdict.reason)

        # 3. retrieve (hybrid) -> 4. rerank
        candidates = self.retriever.retrieve(question, top_k=self.cfg.retrieval.top_k)
        if self.cfg.retrieval.rerank:
            candidates = self.reranker.rerank(
                question, candidates, top_n=self.cfg.retrieval.rerank_top_n
            )

        # 5. generate
        prompt = build_prompt(self.prompt_template, question, candidates)
        text = self.llm.generate(prompt)

        # 6. output guardrail (faithfulness / grounding gate)
        grounding = 1.0
        if self.cfg.guardrails.check_output:
            ov = self.output_guard.check(text, candidates)
            grounding = ov.grounding_score
            if not ov.passed:
                return Answer(
                    text="I don't have enough grounded information to answer that confidently.",
                    sources=candidates, grounding_score=grounding,
                    blocked=True, reason="failed grounding gate",
                )

        return Answer(text=text, sources=candidates, grounding_score=grounding)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipe = QueryPipeline()
    ans = pipe.answer("What is hybrid retrieval?")
    print(ans.text)
    print("sources:", [s.chunk.source for s in ans.sources])
