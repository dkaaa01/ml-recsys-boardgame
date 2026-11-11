"""Save Matrix Factorization metrics and diagnostic plots."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

# Keep Matplotlib and font caches inside the project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from src.models.matrix_factorization import MatrixFactorizationConfig


def save_search_outputs(
    rows: list[dict[str, object]],
    best: dict[str, object],
    output_dir: Path,
) -> None:
    """Save all tuning trials, the winner, and three diagnostic plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "mf_trials.csv", rows)
    _write_json(output_dir / "best_mf.json", best)
    _plot_ranking_metrics(rows, output_dir / "ranking_metrics.png")
    _plot_error_vs_ranking(rows, output_dir / "rmse_vs_ndcg.png")
    _plot_hyperparameters(rows, output_dir / "hyperparameter_effects.png")


def save_final_outputs(
    output_dir: Path,
    config: MatrixFactorizationConfig,
    validation: dict[str, float],
    test: dict[str, float],
    *,
    fit_seconds: float,
    evaluation_seconds: float,
) -> None:
    """Save final test metrics and a validation-versus-test plot."""

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
    _plot_validation_vs_test(validation, test, output_dir / "validation_vs_test.png")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _plot_ranking_metrics(rows: list[dict[str, object]], path: Path) -> None:
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(max(10, len(rows)), 10), sharex=True)
    for axis, metric, label, color in (
        (axes[0], "recall_at_10", "Recall@10", "#4C78A8"),
        (axes[1], "ndcg_at_10", "NDCG@10", "#F58518"),
        (axes[2], "coverage", "Catalog coverage", "#54A24B"),
    ):
        values = [float(row[metric]) for row in rows]
        axis.bar(positions, values, color=color)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Matrix Factorization Hyperparameter Search")
    axes[2].set_xticks(positions, [f"T{row['trial']}" for row in rows])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_error_vs_ranking(rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for row in rows:
        x, y = float(row["rmse"]), float(row["ndcg_at_10"])
        axis.scatter(x, y, s=65)
        axis.annotate(f"T{row['trial']}", (x, y), xytext=(4, 4), textcoords="offset points")
    axis.set(title="Rating Error vs. Ranking Quality", xlabel="RMSE (lower is better)",
             ylabel="NDCG@10 (higher is better)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_hyperparameters(rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        [float(row["learning_rate"]) for row in rows],
        [float(row["ndcg_at_10"]) for row in rows],
        s=[40 + float(row["factors"]) * 2 for row in rows],
        c=[float(row["regularization"]) for row in rows],
        cmap="viridis", edgecolor="black", linewidth=0.5,
    )
    axis.set_xscale("log")
    axis.set(xlabel="Learning rate", ylabel="Validation NDCG@10",
             title="Hyperparameter Effects: marker size = factors")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Regularization")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_validation_vs_test(
    validation: dict[str, float],
    test: dict[str, float],
    path: Path,
) -> None:
    metrics = ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    labels = ("RMSE ↓", "Recall@10 ↑", "NDCG@10 ↑", "Coverage ↑")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        values = (validation[metric], test[metric])
        axis.bar(("Validation", "Test"), values, color=("#4C78A8", "#F58518"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen Matrix Factorization: Validation vs Test")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
