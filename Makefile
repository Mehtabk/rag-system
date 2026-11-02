.PHONY: install ingest serve eval test lint fmt clean

install:
	pip install -e ".[dev]"

ingest:
	python -m rag.pipelines.index_pipeline

serve:
	uvicorn rag.api.app:app --reload --port 8000

eval:
	python -m evals.run_eval

test:
	pytest -q

lint:
	ruff check src tests evals

fmt:
	ruff format src tests evals

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
