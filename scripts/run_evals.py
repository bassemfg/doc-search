"""Offline evaluation helper that logs to LangSmith if configured."""

from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

from app.agents import GRAPH
from app.monitoring import monitor

EVAL_CASES = [
    {
        "query": "Which story covers MongoDB Atlas migration?",
        "expected_title": "Atlas Rising",
    },
    {
        "query": "Document intelligence platform movie",
        "expected_title": "Vector Velocity",
    },
]


def run_case(case: Dict[str, str]):
    initial_state = {
        "operation": "search",
        "query": case["query"],
        "top_k": 3,
        "session_id": str(uuid4()),
    }
    result_state = GRAPH.invoke(initial_state)
    docs = result_state.get("results", [])
    titles = [doc.get("title", "") for doc in docs]
    success = case["expected_title"] in titles
    return {
        "query": case["query"],
        "expected_title": case["expected_title"],
        "retrieved_titles": titles,
        "success": success,
    }


def main() -> None:
    cases: List[Dict[str, str]] = EVAL_CASES
    evaluations = [run_case(case) for case in cases]
    total = len(evaluations)
    hits = sum(1 for case in evaluations if case["success"])
    metrics = {
        "accuracy": hits / total if total else 0.0,
        "total_cases": total,
    }
    print("Evaluation metrics", metrics)
    monitor.log_evaluation("sample_queries", metrics, evaluations)


if __name__ == "__main__":
    main()
