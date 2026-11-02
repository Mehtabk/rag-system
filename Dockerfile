FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY data ./data

RUN pip install --no-cache-dir -e .

EXPOSE 8000
# Build the index at startup, then serve. In production the data path would be
# a separate job, not inline — this is for single-container convenience.
CMD ["sh", "-c", "python -m rag.pipelines.index_pipeline && uvicorn rag.api.app:app --host 0.0.0.0 --port 8000"]
