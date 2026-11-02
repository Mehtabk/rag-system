"""Offline eval runner. Indexes the sample corpus, runs the golden set through
the query pipeline, and reports per-stage metrics. Exits non-zero if aggregate
scores fall below thresholds (used as a CI regression gate).

Usage:
    python -m evals.run_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.metrics import must_include_score, reciprocal_rank, retrieval_recall_at_k
from rag.config import load_config
from rag.pipelines.index_pipeline import run_index
from rag.pipelines.query_pipeline import QueryPipeline

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "datasets" / "golden.json"

# Regression thresholds — tune as the system matures.
THRESHOLDS = {"recall_at_k": 0.9, "mrr": 0.8, "answer": 0.9}


def main() -> int:
    cfg = load_config()
    run_index(cfg)  # ensure the index exists
    pipe = QueryPipeline(cfg)

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    recalls, rrs, answers = [], [], []

    print(f"\nRunning {len(cases)} eval cases\n" + "-" * 48)
    for case in cases:
        ans = pipe.answer(case["question"])
        recall = retrieval_recall_at_k(ans.sources, case["expected_sources"])
        rr = reciprocal_rank(ans.sources, case["expected_sources"])
        aq = must_include_score(ans.text, case.get("must_include", []))
        recalls.append(recall)
        rrs.append(rr)
        answers.append(aq)
        print(f"Q: {case['question']}")
        print(f"   recall@k={recall:.2f}  rr={rr:.2f}  answer={aq:.2f}  grounding={ans.grounding_score:.2f}")

    agg = {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "answer": sum(answers) / len(answers),
    }
    print("-" * 48)
    print("Aggregate:", {k: round(v, 3) for k, v in agg.items()})

    failures = [k for k, thr in THRESHOLDS.items() if agg[k] < thr]
    if failures:
        print(f"\nREGRESSION GATE FAILED on: {', '.join(failures)}")
        return 1
    print("\nRegression gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
