"""Input guardrail: screen the query BEFORE retrieval/generation.

Catches obvious prompt-injection patterns and flags likely PII. This is a gate
in the code path, not an instruction in the prompt — a prompt instruction can
be argued around; a gate cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"disregard (the )?(system|above)",
    r"you are now",
    r"reveal (your )?(system )?prompt",
]
_PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
}


@dataclass
class InputVerdict:
    allowed: bool
    reason: str = ""
    pii_found: list[str] = None  # type: ignore[assignment]


class InputGuardrail:
    def check(self, query: str) -> InputVerdict:
        low = query.lower()
        for pat in _INJECTION_PATTERNS:
            if re.search(pat, low):
                return InputVerdict(allowed=False, reason="possible prompt injection")
        pii = [name for name, pat in _PII_PATTERNS.items() if re.search(pat, query)]
        return InputVerdict(allowed=True, pii_found=pii or [])
