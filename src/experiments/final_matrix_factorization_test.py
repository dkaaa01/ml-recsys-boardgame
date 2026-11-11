"""Evaluate the frozen Matrix Factorization winner on held-out test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.loader import load_interactions
from src.experiments.matrix_factorization_evaluation import evaluate, find_relevant_items
from src.experiments.matrix_factorization_outputs import save_final_outputs
from src.models.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig


def load_frozen_config(path: Path) -> tuple[MatrixFactorizationConfig, dict[str, float]]:
    """Restore model parameters and validation metrics from the winner file."""

    winner = json.loads(path.read_text(encoding="utf-8"))
    config = MatrixFactorizationConfig(
        factors=int(winner["factors"]),
        learning_rate=float(winner["learning_rate"]),
        regularization=float(winner["regularization"]),
        epochs=int(winner["epochs"]),
        batch_size=int(winner["batch_size"]),
        seed=int(winner["seed"]),
        min_rating=float(winner["min_rating"]),
        max_rating=float(winner["max_rating"]),
    )
    validation = {
        metric: float(winner[metric])
        for metric in ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    }
    return config, validation


def run_final_test(args: argparse.Namespace) -> dict[str, float]:
    """Fit the frozen winner and evaluate the test split exactly once."""

    config, validation = load_frozen_config(args.best_config)
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape).ratings
    test = load_interactions(args.test, shape=shape).ratings
    relevant = find_relevant_items(test, args.relevant_threshold)
    if not relevant:
        raise ValueError("No test user has a relevant item")

    started = time.perf_counter()
    model = MatrixFactorization(config).fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = evaluate(model, test, relevant, args.top_k)
    evaluation_seconds = time.perf_counter() - started
    save_final_outputs(
        args.output_dir, config, validation, metrics,
        fit_seconds=fit_seconds, evaluation_seconds=evaluation_seconds,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment_10k/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/experiment_10k/test.csv"))
    parser.add_argument("--best-config", type=Path,
                        default=Path("reports/matrix_factorization_10k/best_mf.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/matrix_factorization_10k"))
    parser.add_argument("--total-users", type=int, default=10_000)
    parser.add_argument("--total-items", type=int, default=1_500)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(run_final_test(args), indent=2))


if __name__ == "__main__":
    main()
