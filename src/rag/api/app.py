"""FastAPI entrypoint. `uvicorn rag.api.app:app`.

Security posture:
  - Docs (/docs, /redoc) are toggleable and OFF by default in production.
  - CORS is deny-by-default; set RAG_CORS_ORIGINS to opt specific origins in.
  - Per-IP rate limiting and conservative security headers on every response.
  - A catch-all handler returns a generic 500 so internal errors/stack traces
    never leak to clients (details go to logs instead).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from rag.api.routes import router
from rag.api.security import RateLimitMiddleware, SecurityHeadersMiddleware
from rag.settings import get_settings

logger = logging.getLogger("rag.api")
settings = get_settings()

app = FastAPI(
    title="rag-system",
    version="0.1.0",
    description="Production-grade RAG skeleton: hybrid retrieval, reranking, guardrails, evals.",
    # Disable interactive docs/schema unless explicitly enabled (off in prod).
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

# --- Middleware (order matters: headers outermost, then rate limit) ---
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)

# CORS: deny-by-default. Only the explicitly listed origins are allowed.
_origins = settings.cors_origin_list()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the real error server-side; return a generic message to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/")
def root() -> dict:
    return {"name": "rag-system", "health": "/health"}
