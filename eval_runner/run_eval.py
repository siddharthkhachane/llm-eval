from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lm_eval import evaluator

from model import OllamaLM


TASKS = ["hellaswag", "mmlu"]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "results.json"
DEFAULT_QUICK_LIMIT = 20


def extract_accuracy(metrics: dict) -> float | None:
    priority_keys = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
    ]
    for key in priority_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    for _, value in metrics.items():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval tasks against local Ollama wrapper.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick runs a limited subset for pipeline validation; full runs the full task.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_QUICK_LIMIT,
        help="Number of examples per task in quick mode (ignored in full mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("eval_runner")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = OllamaLM(endpoint="http://localhost:8000/generate")
    summary: dict[str, dict[str, float | None]] = {}
    run_limit = None if args.mode == "full" else args.limit

    if args.mode == "quick":
        logger.info("Mode: quick (limit=%s samples per task)", run_limit)
    else:
        logger.info("Mode: full (no sample limit)")

    for task in TASKS:
        logger.info("Running task: %s", task)
        try:
            evaluation = evaluator.simple_evaluate(
                model=model,
                tasks=[task],
                log_samples=False,
                verbosity="INFO",
                limit=run_limit,
            )
        except Exception as exc:
            logger.error("Task failed: %s (%s) -- skipping", task, exc)
            summary[task] = {"accuracy": None}
            continue
        task_metrics = evaluation.get("results", {}).get(task, {})
        summary[task] = {"accuracy": extract_accuracy(task_metrics)}

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation complete")
    logger.info("Results saved to results.json")


if __name__ == "__main__":
    main()
