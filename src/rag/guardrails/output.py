"""Output guardrail: a faithfulness / grounding gate BEFORE the answer ships.

Computes a simple lexical grounding score (fraction of answer terms supported
by the retrieved context). If it falls below the configured threshold, the
answer is blocked and replaced with a safe fallback. Real systems use an
NLI/LLM-based faithfulness judge here — same gate, stronger judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.types import ScoredChunk

_TOKEN = re.compile(r"\w+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on", "for",
    "it", "this", "that", "with", "as", "by", "be", "i", "you", "based",
}


def _content_terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


@dataclass
class OutputVerdict:
    grounding_score: float
    passed: bool


class OutputGuardrail:
    def __init__(self, min_grounding_score: float = 0.3):
        self.min_grounding_score = min_grounding_score

    def check(self, answer: str, chunks: list[ScoredChunk]) -> OutputVerdict:
        ans_terms = _content_terms(answer)
        if not ans_terms:
            return OutputVerdict(grounding_score=0.0, passed=False)
        ctx_terms: set[str] = set()
        for c in chunks:
            ctx_terms |= _content_terms(c.chunk.text)
        supported = len(ans_terms & ctx_terms) / len(ans_terms)
        return OutputVerdict(
            grounding_score=round(supported, 3),
            passed=supported >= self.min_grounding_score,
        )
