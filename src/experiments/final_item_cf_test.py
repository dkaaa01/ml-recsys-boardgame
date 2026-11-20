"""Evaluate the frozen ItemCF configuration once on the held-out test split.

This module is intentionally separate from hyperparameter search.  It reads the
already-selected validation winner, fits that exact configuration on training
data, and reports the untouched test performance without changing parameters.
"""

from __future__ import annotations

import argparse
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
from src.experiments.item_cf_experiment import evaluate_model, validation_targets
from src.models.item_cf import ItemCFConfig, ItemItemCF


def load_frozen_config(path: Path) -> tuple[ItemCFConfig, dict[str, float]]:
    """Restore the saved validation winner without selecting new settings.

    Args:
        path: best-model JSON containing parameter fields and validation metrics.

    Returns:
        A ItemCFConfig instance and the saved validation-metric mapping."""

    with path.open(encoding="utf-8") as source:
        winner = json.load(source)
    # CSV-derived JSON stores values as strings.  Restore the dataclass types
    # explicitly so comparisons and array slicing behave as they did in search.
    config = ItemCFConfig(
        similarity=str(winner["similarity"]),  # type: ignore[arg-type]
        neighbors=int(winner["neighbors"]),
        min_overlap=int(winner["min_overlap"]),
        shrinkage=float(winner["shrinkage"]),
        min_similarity=float(winner["min_similarity"]),
        positive_threshold=float(winner["positive_threshold"]),
        scoring=str(winner["scoring"]),  # type: ignore[arg-type]
    )
    validation_metrics = {
        metric: float(winner[metric])
        for metric in ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    }
    return config, validation_metrics


def write_final_report(
    output_dir: Path,
    config: ItemCFConfig,
    validation: dict[str, float],
    test: dict[str, float],
    *,
    fit_seconds: float,
    evaluation_seconds: float,
) -> None:
    """Write frozen ItemCF test evidence and the validation/test comparison.

    Args:
        output_dir: Directory for the JSON metrics, plot and Markdown summary.
        config: Frozen ItemCFConfig restored from validation search.
        validation: Previously saved validation metrics.
        test: New held-out test metrics, including evaluated counts.
        fit_seconds: Measured training duration in seconds.
        evaluation_seconds: Measured test-scoring duration in seconds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "frozen validation winner evaluated once on held-out test data",
        "config": asdict(config),
        "validation": validation,
        "test": test,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    metrics = ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    labels = ("RMSE ↓", "Recall@10 ↑", "NDCG@10 ↑", "Coverage ↑")
    # Separate panels are used because RMSE is around 1.4 while ranking metrics
    # are below 0.12; a single y-axis would hide the important ranking changes.
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        values = [validation[metric], test[metric]]
        bars = axis.bar(("Validation", "Test"), values, color=("#4C78A8", "#F58518"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )
    similarity_label = config.similarity.replace("_", " ").title()
    figure.suptitle(f"Frozen {similarity_label} Model: Validation vs Test")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_vs_test.png", dpi=180)
    plt.close(figure)

    summary = f"""# Final ItemCF test result

The validation winner was frozen before this evaluation. No test metric was
used to select or change the model or its hyperparameters.

## Frozen configuration

- Similarity: `{config.similarity}`
- Scoring: `{config.scoring}`
- Neighbors: `{config.neighbors}`
- Minimum overlap: `{config.min_overlap}`
- Shrinkage: `{config.shrinkage}`
- Minimum similarity: `{config.min_similarity}`

## Validation and final test metrics

| Split | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | {validation['rmse']:.4f} | {validation['recall_at_10']:.4f} | {validation['ndcg_at_10']:.4f} | {validation['coverage']:.4f} |
| **Test (final)** | **{test['rmse']:.4f}** | **{test['recall_at_10']:.4f}** | **{test['ndcg_at_10']:.4f}** | **{test['coverage']:.4f}** |

- Test rating predictions: {int(test['rating_predictions']):,}
- Test users with at least one relevant game: {int(test['ranking_users']):,}
- Fit time: {fit_seconds:.2f} seconds
- Test evaluation time: {evaluation_seconds:.2f} seconds
"""
    (output_dir / "final_test_summary.md").write_text(summary, encoding="utf-8")


def run_final_test(args: argparse.Namespace) -> dict[str, float]:
    """Fit the frozen validation winner and evaluate held-out test data.

    Args:
        args: CLI namespace with train, test, best_config, output_dir, total_users, total_items, top_k and relevant_threshold.

    Returns:
        The test metric dictionary; final JSON/plots are also written.

    Train and test share an explicit matrix shape. Test metrics are reported after the configuration is fixed and are not used by this function to alter it."""

    # Step 1: restore the validation winner without changing any parameter.
    config, validation_metrics = load_frozen_config(args.best_config)
    # Step 2: load train and the held-out test split using the fixed matrix shape.
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape).ratings
    test = load_interactions(args.test, shape=shape).ratings
    relevant, test_users = validation_targets(test, args.relevant_threshold)
    if not relevant:
        raise ValueError("No test user has a relevant item at the selected threshold")

    # Step 3: fit once, evaluate once, and persist the final comparison evidence.
    fit_started = time.perf_counter()
    model = ItemItemCF(config).fit(train)
    fit_seconds = time.perf_counter() - fit_started
    evaluation_started = time.perf_counter()
    test_metrics, _ = evaluate_model(model, test, relevant, test_users, args.top_k)
    evaluation_seconds = time.perf_counter() - evaluation_started

    write_final_report(
        args.output_dir,
        config,
        validation_metrics,
        test_metrics,
        fit_seconds=fit_seconds,
        evaluation_seconds=evaluation_seconds,
    )
    return test_metrics


def main() -> None:
    """Parse final-test paths and dimensions, then print the test metrics.

    --train and --test select the fixed split CSVs; --best-config restores the validation winner; --output-dir receives results; --total-users/--total-items define matrix shape; --top-k is 10; --relevant-threshold defaults to 8."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/experiment/test.csv"))
    parser.add_argument(
        "--best-config",
        type=Path,
        default=Path("reports/item_cf/best_item_cf.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/item_cf"))
    parser.add_argument("--total-users", type=int, default=20_000)
    parser.add_argument("--total-items", type=int, default=2_000)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(run_final_test(args), indent=2))


if __name__ == "__main__":
    main()
