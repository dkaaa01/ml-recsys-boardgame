"""Tune Matrix Factorization on validation data without reading test data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.loader import load_interactions, sample_user_ids
from src.experiments.matrix_factorization_evaluation import evaluate, find_relevant_items
from src.experiments.matrix_factorization_outputs import save_search_outputs
from src.models.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig


def build_trial_configs(trials: int, seed: int) -> list[MatrixFactorizationConfig]:
    """Create deterministic random-search trials with one stable reference."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    reference = MatrixFactorizationConfig(
        factors=32, learning_rate=0.01, regularization=0.05,
        epochs=12, batch_size=4096, seed=seed,
    )
    configs = [reference]
    seen = {reference}
    random = np.random.default_rng(seed)
    while len(configs) < trials:
        config = MatrixFactorizationConfig(
            factors=int(random.choice((16, 32, 64))),
            learning_rate=float(random.choice((0.005, 0.01, 0.02))),
            regularization=float(random.choice((0.01, 0.05, 0.10))),
            epochs=int(random.choice((8, 12, 18))),
            batch_size=4096,
            seed=seed,
        )
        if config not in seen:
            seen.add(config)
            configs.append(config)
    return configs


def selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    """Prefer ranking quality, then recall, lower error, and coverage."""

    return (
        float(row["ndcg_at_10"]), float(row["recall_at_10"]),
        -float(row["rmse"]), float(row["coverage"]),
    )


def run_experiment(args: argparse.Namespace) -> list[dict[str, object]]:
    """Train each configuration, evaluate it, and persist the search result."""

    if not 0 < args.sample_users <= args.total_users:
        raise ValueError("sample_users must be between 1 and total_users")
    # Step 1: load the deterministic user cohort from train and validation only.
    users = sample_user_ids(args.total_users, args.sample_users, args.seed)
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape, user_filter=users).ratings
    validation = load_interactions(args.validation, shape=shape, user_filter=users).ratings
    relevant = find_relevant_items(validation, args.relevant_threshold)
    if not relevant:
        raise ValueError("No validation user has a relevant item; increase sample size")

    # Step 2: fit and evaluate every predetermined hyperparameter configuration.
    rows = [
        _run_trial(trial, config, train, validation, relevant, args.top_k)
        for trial, config in enumerate(build_trial_configs(args.trials, args.seed))
    ]
    # Step 3: freeze the validation winner and save metrics plus diagnostics.
    best = max(rows, key=selection_key)
    save_search_outputs(rows, best, args.output_dir)
    return rows


def _run_trial(trial, config, train, validation, relevant, top_k) -> dict[str, object]:
    print(f"Trial {trial}: {config}")
    started = time.perf_counter()
    model = MatrixFactorization(config).fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = evaluate(model, validation, relevant, top_k)
    return {
        "trial": trial, **asdict(config), **metrics,
        "train_rmse": model.training_rmse_[-1],
        "fit_seconds": fit_seconds,
        "evaluation_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/experiment/validation.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/matrix_factorization")
    )
    parser.add_argument("--total-users", type=int, default=20_000)
    parser.add_argument("--total-items", type=int, default=2_000)
    parser.add_argument("--sample-users", type=int, default=20_000)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_experiment(args)
    print(json.dumps(max(rows, key=selection_key), indent=2))


if __name__ == "__main__":
    main()
