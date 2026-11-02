"""Query router. A router-first design classifies intent before retrieving so
the system can short-circuit (e.g. greetings), pick an index, or decline.

The default is a minimal heuristic. Real systems use a small classifier or LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Route:
    needs_retrieval: bool
    reason: str = ""


class SimpleRouter:
    GREETINGS: ClassVar[set[str]] = {"hi", "hello", "hey", "thanks", "thank you"}

    def route(self, query: str) -> Route:
        q = query.strip().lower()
        if not q:
            return Route(needs_retrieval=False, reason="empty query")
        if q in self.GREETINGS:
            return Route(needs_retrieval=False, reason="greeting / small talk")
        return Route(needs_retrieval=True)
