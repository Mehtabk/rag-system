"""API security: rate limiting, security headers, and admin auth.

Deliberately dependency-free (in-memory fixed-window limiter) so the skeleton
needs no Redis. For multi-instance production, swap the limiter for a shared
store (Redis) — the interface stays the same.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from rag.settings import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client-IP rate limiter. Fails *closed* on abuse."""

    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # Honor a single proxy hop if present, else the socket peer.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        key = self._client_key(request)
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry = int(self.window - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(retry)},
            )
        bucket.append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds conservative security headers to every response.

    The default CSP is strict (`default-src 'none'`) — correct for a JSON API.
    The interactive docs pages (/docs, /redoc) load Swagger/ReDoc assets from a
    CDN and need a relaxed CSP, so those paths get a scoped exception. Every
    other route keeps the strict policy.
    """

    # Swagger UI / ReDoc pull assets + fonts from jsdelivr; ReDoc also needs
    # inline styles and a worker blob. Scope the relaxation to the docs routes.
    _DOCS_PATHS = ("/docs", "/redoc")
    _DOCS_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://cdn.jsdelivr.net https://fastapi.tiangolo.com; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "worker-src 'self' blob:; "
        "frame-ancestors 'none'"
    )
    _STRICT_CSP = "default-src 'none'; frame-ancestors 'none'"

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        is_docs = request.url.path in self._DOCS_PATHS
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy", self._DOCS_CSP if is_docs else self._STRICT_CSP
        )
        # Swagger UI must not be framed either, but it doesn't need DENY to work;
        # keep DENY everywhere for clickjacking protection.
        response.headers.setdefault("X-Frame-Options", "DENY")
        # Only assert HSTS when actually served over TLS.
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Dependency guarding privileged endpoints.

    - production + no admin token configured  -> endpoint disabled (fail closed)
    - admin token configured                  -> constant-time compare required
    - non-production + no token configured    -> allowed (dev convenience)
    """
    import hmac

    settings = get_settings()
    configured = settings.admin_token.get_secret_value() if settings.admin_token else None

    if configured is None:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="privileged endpoint disabled (no admin token configured)",
            )
        return  # dev convenience

    if not x_admin_token or not hmac.compare_digest(x_admin_token, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing admin token",
        )
