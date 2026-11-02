"""Security-surface tests: input bounds, admin auth, rate limiting, headers,
and error handling that doesn't leak internals.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Permissive-but-present limits for deterministic tests.
    monkeypatch.setenv("RAG_ENV", "local")
    monkeypatch.setenv("RAG_RATE_LIMIT_REQUESTS", "1000")
    monkeypatch.setenv("RAG_MAX_QUESTION_CHARS", "50")
    # Reload settings + app so the env takes effect.
    from rag import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    import rag.api.app as app_mod
    import rag.api.schemas as schemas_mod

    importlib.reload(schemas_mod)
    importlib.reload(app_mod)

    from rag.pipelines.index_pipeline import run_index

    run_index()
    return TestClient(app_mod.app)


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    # Non-docs routes keep the strict, locked-down CSP.
    assert r.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"


def test_docs_csp_allows_cdn_assets(client):
    # The docs page needs a relaxed CSP so Swagger UI assets can load.
    r = client.get("/docs")
    assert r.status_code == 200
    csp = r.headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" in csp
    assert "default-src 'none'" not in csp


def test_oversized_question_rejected_422(client):
    r = client.post("/query", json={"question": "x" * 500})
    assert r.status_code == 422  # exceeds RAG_MAX_QUESTION_CHARS


def test_empty_question_rejected_422(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_ingest_requires_admin_token_outside_dev(client, monkeypatch):
    # With a token configured, anonymous /ingest must be rejected.
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "s3cret-token")
    from rag import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    # No header -> 401
    assert client.post("/ingest").status_code == 401
    # Correct header -> 200
    ok = client.post("/ingest", headers={"X-Admin-Token": "s3cret-token"})
    assert ok.status_code == 200
    # Wrong header -> 401
    assert client.post("/ingest", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_rate_limit_triggers_429(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "local")
    monkeypatch.setenv("RAG_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RAG_RATE_LIMIT_WINDOW_SECONDS", "60")
    from rag import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    import rag.api.app as app_mod

    importlib.reload(app_mod)
    c = TestClient(app_mod.app)

    codes = [c.get("/health").status_code for _ in range(5)]
    assert 429 in codes, f"expected a 429 within the burst, got {codes}"
