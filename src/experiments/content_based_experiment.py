"""Tune and evaluate the description-and-tag content recommender."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

# Keep Matplotlib and font caches inside the project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_interactions
from src.experiments.matrix_factorization_evaluation import find_relevant_items
from src.metrics import catalog_coverage, ndcg_at_k, recall_at_k, rmse, validate_recommendations
from src.models.content_based import (
    ContentBasedRecommender,
    ContentConfig,
    build_item_features,
)


CONFIG_FIELDS = tuple(asdict(ContentConfig()))
METRIC_FIELDS = ("rmse", "recall_at_10", "ndcg_at_10", "coverage")


def build_trial_configs() -> list[ContentConfig]:
    """Compare description-heavy, balanced, and tag-heavy feature combinations."""

    return [
        ContentConfig(
            max_description_features=max_features,
            description_weight=description_weight,
        )
        for max_features in (2_000, 4_000)
        for description_weight in (0.25, 0.50, 0.75)
    ]


def selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    """Prefer ranking quality, then recall, lower error, and coverage."""

    return (
        float(row["ndcg_at_10"]),
        float(row["recall_at_10"]),
        -float(row["rmse"]),
        float(row["coverage"]),
    )


def evaluate(
    model: ContentBasedRecommender,
    held_out,
    relevant: dict[int, set[int]],
    top_k: int,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Measure held-out rating error and unseen Top-K ranking quality."""

    if model.train_ is None:
        raise RuntimeError("model must be fitted before evaluation")
    held_out_coo = held_out.tocoo()
    predictions = model.predict_pairs(held_out_coo.row, held_out_coo.col)
    recommendations, _ = model.recommend(
        sorted(relevant), top_k=top_k, batch_size=batch_size
    )
    seen_items = {
        user_id: set(map(int, model.train_.getrow(user_id).indices))
        for user_id in relevant
    }
    validate_recommendations(recommendations, seen_items, k=top_k)
    return {
        "rmse": rmse(held_out_coo.data, predictions),
        "recall_at_10": recall_at_k(recommendations, relevant, top_k),
        "ndcg_at_10": ndcg_at_k(recommendations, relevant, top_k),
        "coverage": catalog_coverage(
            recommendations, range(held_out.shape[1]), k=top_k
        ),
        "rating_predictions": float(held_out_coo.nnz),
        "ranking_users": float(len(relevant)),
    }


def run_tuning(args: argparse.Namespace) -> list[dict[str, object]]:
    """Tune only on train and validation; never open the test file."""

    # Step 1: load train and validation only; tuning never opens the test file.
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape).ratings
    validation = load_interactions(args.validation, shape=shape).ratings
    relevant = find_relevant_items(validation, args.relevant_threshold)
    if not relevant:
        raise ValueError("No validation user has a relevant item")

    # Step 2: rebuild features and evaluate every predetermined configuration.
    rows: list[dict[str, object]] = []
    for trial, config in enumerate(build_trial_configs()):
        print(f"Trial {trial}: {config}")
        started = time.perf_counter()
        features = build_item_features(
            args.games, config, expected_items=args.total_items
        )
        model = ContentBasedRecommender(config).fit(train, features.matrix)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metrics = evaluate(
            model,
            validation,
            relevant,
            args.top_k,
            batch_size=args.batch_size,
        )
        rows.append(
            {
                "trial": trial,
                **asdict(config),
                "description_features": features.description_features,
                "tag_features": features.tag_features,
                **metrics,
                "fit_seconds": fit_seconds,
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
    # Step 3: freeze the validation winner and persist all comparison evidence.
    best = max(rows, key=selection_key)
    save_tuning_outputs(rows, best, args.output_dir)
    return rows


def run_final_test(args: argparse.Namespace) -> dict[str, float]:
    """Evaluate the frozen validation winner once and export ranked content scores."""

    # Step 1: restore the already-frozen validation winner without searching.
    winner = json.loads(args.best_config.read_text(encoding="utf-8"))
    config = ContentConfig(**{field: winner[field] for field in CONFIG_FIELDS})
    validation = {field: float(winner[field]) for field in METRIC_FIELDS}
    # Step 2: open train and test only after the configuration is fixed.
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape).ratings
    test = load_interactions(args.test, shape=shape).ratings
    relevant = find_relevant_items(test, args.relevant_threshold)
    if not relevant:
        raise ValueError("No test user has a relevant item")

    # Step 3: rebuild the frozen model and evaluate the held-out test once.
    started = time.perf_counter()
    features = build_item_features(args.games, config, expected_items=args.total_items)
    model = ContentBasedRecommender(config).fit(train, features.matrix)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = evaluate(
        model, test, relevant, args.top_k, batch_size=args.batch_size
    )
    evaluation_seconds = time.perf_counter() - started
    # Step 4: save final metrics, comparison plots, and compact handoff scores.
    save_final_outputs(
        args.output_dir,
        config,
        validation,
        metrics,
        fit_seconds=fit_seconds,
        evaluation_seconds=evaluation_seconds,
    )
    export_content_scores(
        model,
        sorted(relevant),
        args.scores_output,
        top_k=args.export_top_k,
        batch_size=args.batch_size,
    )
    return metrics


def export_content_scores(
    model: ContentBasedRecommender,
    user_ids: list[int],
    output_path: Path,
    *,
    top_k: int,
    batch_size: int,
) -> None:
    """Write compact per-user Top-K scores instead of a 40-million-row dense file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("user_id", "item_id", "rank", "content_score"))
        for user_id, ranking in model.iter_recommendations(
            user_ids, top_k=top_k, batch_size=batch_size
        ):
            writer.writerows(
                (user_id, item_id, rank, f"{score:.8f}")
                for rank, (item_id, score) in enumerate(ranking, 1)
            )


def save_tuning_outputs(
    rows: list[dict[str, object]],
    best: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "content_trials.csv", rows)
    _write_json(output_dir / "best_content.json", best)

    positions = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, metric, label, color in (
        (axes[0], "recall_at_10", "Recall@10", "#4C78A8"),
        (axes[1], "ndcg_at_10", "NDCG@10", "#F58518"),
        (axes[2], "coverage", "Coverage", "#54A24B"),
    ):
        axis.bar(positions, [float(row[metric]) for row in rows], color=color)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    labels = [
        f"T{row['trial']}\n{int(row['max_description_features'])} terms\n"
        f"w={float(row['description_weight']):.2f}"
        for row in rows
    ]
    axes[0].set_title("Content Feature Search")
    axes[2].set_xticks(positions, labels)
    figure.tight_layout()
    figure.savefig(output_dir / "ranking_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for row in rows:
        axis.scatter(
            float(row["description_weight"]),
            float(row["ndcg_at_10"]),
            s=60 + int(row["max_description_features"]) / 50,
        )
        axis.annotate(
            f"T{row['trial']}",
            (float(row["description_weight"]), float(row["ndcg_at_10"])),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set(
        xlabel="Description weight",
        ylabel="Validation NDCG@10",
        title="Description and Tag Weighting",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_weight_effects.png", dpi=180)
    plt.close(figure)


def save_final_outputs(
    output_dir: Path,
    config: ContentConfig,
    validation: dict[str, float],
    test: dict[str, float],
    *,
    fit_seconds: float,
    evaluation_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "final_test_metrics.json",
        {
            "protocol": "frozen validation winner evaluated once on held-out test data",
            "config": asdict(config),
            "validation": validation,
            "test": test,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, label in zip(
        axes.ravel(),
        METRIC_FIELDS,
        ("RMSE ↓", "Recall@10 ↑", "NDCG@10 ↑", "Coverage ↑"),
    ):
        axis.bar(
            ("Validation", "Test"),
            (validation[metric], test[metric]),
            color=("#4C78A8", "#F58518"),
        )
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen Content Model: Validation vs Test")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_vs_test.png", dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tune", "final"), default="tune")
    parser.add_argument("--train", type=Path, default=Path("data/experiment/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/experiment/validation.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/experiment/test.csv"))
    parser.add_argument("--games", type=Path, default=Path("data/experiment/games.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/content_based")
    )
    parser.add_argument(
        "--best-config",
        type=Path,
        default=Path("reports/content_based/best_content.json"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("data/experiment/content_scores.csv"),
    )
    parser.add_argument("--total-users", type=int, default=20_000)
    parser.add_argument("--total-items", type=int, default=2_000)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--export-top-k", type=int, default=100)
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "tune":
        rows = run_tuning(args)
        print(json.dumps(max(rows, key=selection_key), indent=2))
    else:
        print(json.dumps(run_final_test(args), indent=2))


if __name__ == "__main__":
    main()
