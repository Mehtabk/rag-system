"""Runtime/security settings, sourced from environment variables.

Separated from config.yaml on purpose:
  - config.yaml holds *non-secret* tuning (chunk size, top_k, provider names)
    and is safe to commit.
  - settings.py holds *operational/security* values (API keys, CORS origins,
    rate limits, admin token) sourced from the environment, never committed.

Secrets are read from the environment only. They are never logged or echoed —
`repr()` of this object redacts them.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    # --- Environment ---
    env: str = Field(default="local")  # local | staging | production
    log_level: str = Field(default="INFO")

    # --- API surface controls ---
    # Comma-separated allowed origins. Empty => no cross-origin access (safe default).
    cors_origins: str = Field(default="")
    # Expose interactive docs (/docs, /redoc). Off by default in production.
    enable_docs: bool = Field(default=True)
    # Max characters accepted in a question (defense against oversized/costly input).
    max_question_chars: int = Field(default=2000)
    # Per-client request budget (requests per window) and window length in seconds.
    rate_limit_requests: int = Field(default=60)
    rate_limit_window_seconds: int = Field(default=60)

    # --- Admin / privileged operations ---
    # Token required to call privileged endpoints like /ingest. If unset AND
    # env == production, those endpoints are disabled entirely (fail closed).
    admin_token: SecretStr | None = Field(default=None)

    # --- Provider credentials (only needed when using cloud providers) ---
    openai_api_key: SecretStr | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    cohere_api_key: SecretStr | None = Field(default=None)
    pinecone_api_key: SecretStr | None = Field(default=None)

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def __repr__(self) -> str:  # never leak secret values in logs/tracebacks
        return (
            f"Settings(env={self.env!r}, enable_docs={self.enable_docs}, "
            f"cors_origins={self.cors_origin_list()!r}, "
            f"rate_limit={self.rate_limit_requests}/{self.rate_limit_window_seconds}s, "
            f"admin_token={'set' if self.admin_token else 'unset'})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
