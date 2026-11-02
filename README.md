# rag-system

[![CI](https://github.com/Mehtabk/rag-system/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mehtabk/rag-system/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A production-grade **Retrieval-Augmented Generation** skeleton.

It encodes the architecture the 2026 best-practice consensus converged on:

- **Two decoupled paths**, not one pipeline:
  - **Data path** (offline): `ingest → chunk → embed → write`
  - **Request path** (online): `gateway → embed query → retrieve → rerank → generate`
- **Hybrid retrieval** (vector + keyword) followed by **cross-encoder reranking**.
- **Guardrails** on both input (PII / injection) and output (faithfulness / grounding).
- **First-class evals** (`evals/`) as a peer to `tests/`, with a CI regression gate.
- **Everything behind interfaces** so each stage is independently testable and swappable.
- **Observability** (tracing, latency, token cost) baked in.

This repo ships **working placeholder implementations** — an in-memory vector store, a
hashing "embedder", and an echo generator — so the whole thing runs end-to-end with zero
external services or API keys. Swap in real providers under each `providers/` folder.

## Quickstart

```bash
# 1. install (editable, with dev tools)
pip install -e ".[dev]"

# 2. index the sample documents (data path)
make ingest

# 3. serve the API (request path)
make serve           # -> http://localhost:8000/docs

# 4. ask a question
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "What is hybrid retrieval?"}' | python3 -m json.tool

# 5. run the eval harness + regression gate
make eval

# 6. tests
make test
```

## Layout

```
config/            models, chunk sizes, top_k, prompts, logging — centralized
src/rag/
  ingestion/       loaders + layout-aware parsing + the offline index run   ─┐ data path
  chunking/        splitters + metadata (source, section, ACLs)              │
  embeddings/      provider-agnostic embedder + providers/                   │
  vectorstore/     abstract store + providers (pinecone/pgvector/...)       ─┘
  retrieval/       hybrid (vector+BM25) + rerank + query router             ─┐ request path
  generation/      llm interface + providers + citation enforcement          │
  guardrails/      input (PII/injection) + output (faithfulness) gates        │
  pipelines/       index_pipeline.py / query_pipeline.py wire it together    ─┘
  api/             FastAPI app, routes, pydantic schemas
  observability/   tracing, metrics, logging
  utils/           caching, shared helpers
evals/             golden datasets, metrics, eval runner, CI regression gate
tests/             unit + integration
scripts/           reindex, seed
```

## Choosing an embedder

The embedder is swappable via `config/config.yaml` (`embeddings.provider`). Three options,
trading off setup cost vs. quality:

| Provider | Config value | Deps | Keys | Quality | When to use |
|----------|--------------|------|------|---------|-------------|
| Hashing  | `hashing`    | none | no   | low (lexical only) | fast CI, unit tests, zero-footprint demos |
| Local ST | `local`      | `.[local]` (torch) | no | good (real semantic) | offline dev, privacy-sensitive, no API budget |
| OpenAI   | `openai`     | `.[providers]` | yes | best | production with an API budget |

**Local (recommended default for real use, offline):**
```bash
pip install -e ".[local]"
# config.yaml -> embeddings.provider: local, model: all-MiniLM-L6-v2, dim: 384
python scripts/reindex.py      # re-embed the corpus with the new model
```
First run downloads the model (~90 MB) to `~/.cache/huggingface`; afterwards it is fully
offline. `all-MiniLM-L6-v2` is small and CPU-fast; swap `model:` for a larger one
(`bge-base-en-v1.5`, `e5-large-v2`) if you want more accuracy.

> Changing the embedder changes the vector dimensionality, so **always reindex** after a
> switch — a store built with one model cannot be searched with another.

## Security

Hardened defaults, all configurable via `RAG_*` env vars (see `.env.example`):

- **Deny-by-default CORS** — set `RAG_CORS_ORIGINS` to opt specific origins in.
- **Docs off in production** — `/docs`, `/redoc`, `/openapi.json` disabled unless
  `RAG_ENABLE_DOCS=true`.
- **Per-IP rate limiting** (`RAG_RATE_LIMIT_*`) → `429` on abuse.
- **Security headers** on every response (`X-Content-Type-Options`, `X-Frame-Options`,
  CSP, HSTS over TLS).
- **Bounded input** — questions over `RAG_MAX_QUESTION_CHARS` are rejected with `422`.
- **Admin-gated `/ingest`** — requires `X-Admin-Token` (constant-time compared). In
  production with no token configured, the endpoint is **disabled (fail closed)**.
- **Input guardrail** screens queries for prompt injection / PII *before* retrieval;
  **output guardrail** blocks ungrounded answers.
- **Generic 500s** — internal errors are logged server-side, never leaked to clients.
- **Secrets** come from the environment only (`SecretStr`), are never logged, and `.env`
  is gitignored. Generate the admin token with:
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- **Defensive ingestion/index I/O** — oversized files are skipped, symlinks escaping the
  ingestion root are refused, and a corrupted index file degrades gracefully to empty.

## Swapping in real providers

Each pluggable stage exposes an abstract base + a `providers/` folder:

| Stage        | Interface                     | Example real provider           |
|--------------|-------------------------------|---------------------------------|
| Embeddings   | `embeddings/embedder.py`      | `providers/openai_embedder.py`  |
| Vector store | `vectorstore/base.py`         | `providers/pinecone_store.py`   |
| Generation   | `generation/llm.py`           | `providers/anthropic_llm.py`    |

Point `config/config.yaml` at the provider you want; nothing else changes.

> This is a skeleton for teaching/bootstrapping. For enterprise scale, split the index and
> query paths into separately deployed services behind an async queue, add multi-tenant
> isolation, a caching tier, and permission-aware retrieval.

## License

MIT — see [LICENSE](./LICENSE). Built by [DeeplyAgentic](https://www.deeplyagentic.com).
