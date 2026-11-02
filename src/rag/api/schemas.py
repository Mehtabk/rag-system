"""Pydantic request/response models — the API's typed contract."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from rag.settings import get_settings


class QueryRequest(BaseModel):
    # min_length is static; the max is enforced dynamically against current
    # settings so the limit stays configurable at runtime (not frozen at import).
    question: str = Field(..., min_length=1, examples=["What is hybrid retrieval?"])

    @field_validator("question")
    @classmethod
    def _bound_length(cls, v: str) -> str:
        max_chars = get_settings().max_question_chars
        if len(v) > max_chars:
            raise ValueError(f"question exceeds max length of {max_chars} characters")
        return v


class SourceOut(BaseModel):
    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    grounding_score: float
    blocked: bool
    reason: str | None = None
    sources: list[SourceOut]
