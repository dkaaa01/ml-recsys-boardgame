"""Tune ItemCF on the 20,000-user experiment dataset."""

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
from matplotlib.patches import Patch

from src.data.loader import load_interactions, sample_user_ids
from src.models.item_cf import ItemCFConfig, ItemItemCF
from src.metrics import (
    catalog_coverage,
    ndcg_at_k,
    recall_at_k,
    rmse,
    validate_recommendations,
)


SIMILARITIES = ("cosine", "adjusted_cosine", "pearson", "jaccard")


def build_trial_configs(trials: int, seed: int, sample_users: int, total_users: int) -> list[ItemCFConfig]:
    """Create conditional random-search trials while guaranteeing all 4 methods.

    Scikit-learn's ``RandomizedSearchCV`` assumes a conventional estimator and a
    single row-wise score.  Recommendation evaluation instead needs per-user
    top-K lists and conditional parameters (``positive_threshold`` matters only
    to Jaccard), so the small search scheduler is explicit here.
    """

    if trials < len(SIMILARITIES):
        raise ValueError(f"trials must be at least {len(SIMILARITIES)}")
    # A user subsample produces fewer co-raters per item pair, so the smoke run
    # deliberately searches smaller support thresholds than the full run.
    support_values = (1, 2, 3, 5) if sample_users < total_users else (3, 5, 10, 20)
    shrinkage_values = (0.0, 10.0, 25.0, 100.0) if sample_users < total_users else (0.0, 25.0, 100.0, 250.0)
    rng = np.random.default_rng(seed)

    configs = [
        ItemCFConfig(
            similarity=name,
            neighbors=40,
            min_overlap=support_values[0],
            shrinkage=shrinkage_values[1],
            positive_threshold=8.0,
            scoring="baseline_residual",
        )
        for name in SIMILARITIES
    ]
    seen = {tuple(asdict(config).items()) for config in configs}
    while len(configs) < trials:
        # Round-robin assignment gives every similarity family the same search
        # budget (difference at most one when trials is not divisible by four).
        similarity = SIMILARITIES[(len(configs) - len(SIMILARITIES)) % len(SIMILARITIES)]
        config = ItemCFConfig(
            similarity=similarity,  # type: ignore[arg-type]
            neighbors=int(rng.choice((20, 40, 80, 120))),
            min_overlap=int(rng.choice(support_values)),
            shrinkage=float(rng.choice(shrinkage_values)),
            min_similarity=float(rng.choice((0.0, 0.05, 0.10))),
            positive_threshold=float(rng.choice((7.0, 8.0))) if similarity == "jaccard" else 8.0,
            scoring=str(rng.choice(("weighted_mean", "baseline_residual"))),  # type: ignore[arg-type]
        )
        key = tuple(asdict(config).items())
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def build_local_grid(
    best: ItemCFConfig,
    *,
    sample_users: int,
    total_users: int,
    max_trials: int,
) -> list[ItemCFConfig]:
    """Create a small grid around the best random-search configuration.

    Full Cartesian grid search repeats many obviously distant combinations.
    This local grid changes only neighboring values of K, minimum support, and
    shrinkage, then caps the number of refinements for predictable runtime.
    """

    if max_trials <= 0:
        return []
    neighbor_space = (20, 40, 80, 120)
    support_space = (1, 2, 3, 5) if sample_users < total_users else (3, 5, 10, 20)
    shrinkage_space = (0.0, 10.0, 25.0, 100.0) if sample_users < total_users else (0.0, 25.0, 100.0, 250.0)

    def nearby(space, value):
        index = space.index(value)
        return space[max(0, index - 1) : min(len(space), index + 2)]

    candidates: list[ItemCFConfig] = []
    for neighbors in nearby(neighbor_space, best.neighbors):
        for min_overlap in nearby(support_space, best.min_overlap):
            for shrinkage in nearby(shrinkage_space, best.shrinkage):
                config = ItemCFConfig(
                    similarity=best.similarity,
                    neighbors=neighbors,
                    min_overlap=min_overlap,
                    shrinkage=shrinkage,
                    min_similarity=best.min_similarity,
                    positive_threshold=best.positive_threshold,
                    scoring=best.scoring,
                )
                if config != best:
                    candidates.append(config)
    # Prefer one-step changes before combinations of two or three changes.
    candidates.sort(
        key=lambda config: (
            (config.neighbors != best.neighbors)
            + (config.min_overlap != best.min_overlap)
            + (config.shrinkage != best.shrinkage),
            config.neighbors,
            config.min_overlap,
            config.shrinkage,
        )
    )
    return candidates[:max_trials]


def validation_targets(validation, relevant_threshold: float) -> tuple[dict[int, set[int]], list[int]]:
    """Return binary relevance sets and users that have validation interactions."""

    relevant: dict[int, set[int]] = {}
    users: list[int] = []
    for user_id in np.flatnonzero(validation.getnnz(axis=1)):
        row = validation.getrow(int(user_id))
        users.append(int(user_id))
        liked = row.indices[row.data >= relevant_threshold]
        if liked.size:
            relevant[int(user_id)] = set(map(int, liked))
    return relevant, users


def evaluate_model(
    model: ItemItemCF,
    validation,
    relevant: dict[int, set[int]],
    validation_users: list[int],
    top_k: int,
) -> tuple[dict[str, float], dict[int, list[int]]]:
    """Evaluate rating error and top-K quality without recomputing user scores."""

    actual: list[float] = []
    predicted: list[float] = []
    recommendations: dict[int, list[int]] = {}
    ranking_users = set(relevant)
    train = model.train_
    assert train is not None

    for user_id in validation_users:
        scores = model.score_user(user_id)
        validation_row = validation.getrow(user_id)
        actual.extend(map(float, validation_row.data))
        predicted.extend(map(float, scores[validation_row.indices]))
        if user_id not in ranking_users:
            continue

        # The validation positives stay eligible.  Only training-seen games are
        # masked, matching what would be known at recommendation time.
        scores[train.getrow(user_id).indices] = -np.inf
        item_ids = np.arange(scores.size)
        order = np.lexsort((item_ids, -scores))
        recommendations[user_id] = [
            int(item_id) for item_id in order[np.isfinite(scores[order])][:top_k]
        ]

    validate_recommendations(
        recommendations,
        {user_id: set(map(int, train.getrow(user_id).indices)) for user_id in recommendations},
        k=top_k,
    )

    metrics = {
        "rmse": rmse(actual, predicted),
        "recall_at_10": recall_at_k(recommendations, relevant, top_k),
        "ndcg_at_10": ndcg_at_k(recommendations, relevant, top_k),
        "coverage": catalog_coverage(recommendations, range(validation.shape[1])),
        "rating_predictions": float(len(actual)),
        "ranking_users": float(len(relevant)),
    }
    return metrics, recommendations


def bayesian_popularity_result(
    train, validation, relevant, validation_users, top_k: int
) -> dict[str, object]:
    """Evaluate a train-only Bayesian popularity control under the same protocol."""

    counts = np.asarray(train.getnnz(axis=0), dtype=float)
    sums = np.asarray(train.sum(axis=0)).ravel()
    global_mean = float(train.data.mean())
    means = np.divide(sums, counts, out=np.full_like(sums, global_mean), where=counts > 0)
    prior = float(np.quantile(counts[counts > 0], 0.75))
    scores = (counts * means + prior * global_mean) / np.maximum(counts + prior, 1.0)
    item_ids = np.arange(train.shape[1])

    actual: list[float] = []
    predicted: list[float] = []
    recommendations: dict[int, list[int]] = {}
    for user_id in validation_users:
        row = validation.getrow(user_id)
        actual.extend(map(float, row.data))
        predicted.extend(map(float, scores[row.indices]))
        if user_id not in relevant:
            continue
        user_scores = scores.copy()
        user_scores[train.getrow(user_id).indices] = -np.inf
        order = np.lexsort((item_ids, -user_scores))
        recommendations[user_id] = [
            int(item_id) for item_id in order[np.isfinite(user_scores[order])][:top_k]
        ]
    validate_recommendations(
        recommendations,
        {user_id: set(map(int, train.getrow(user_id).indices)) for user_id in recommendations},
        k=top_k,
    )
    return {
        "trial": -1,
        "stage": "baseline",
        "model": "bayesian_popularity",
        "rmse": rmse(actual, predicted),
        "recall_at_10": recall_at_k(recommendations, relevant, top_k),
        "ndcg_at_10": ndcg_at_k(recommendations, relevant, top_k),
        "coverage": catalog_coverage(recommendations, range(train.shape[1])),
        "fit_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "rating_predictions": len(actual),
        "ranking_users": len(relevant),
    }


def write_results(
    rows: list[dict[str, object]], output_dir: Path, *, full_validation: bool = False
) -> None:
    """Write machine-readable search results and three presentation-ready plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "item_cf_trials.csv").open("w", encoding="utf-8", newline="") as output:
        # Force LF so generated reports produce clean, cross-platform Git diffs.
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row for row in rows if row["model"] == "item_cf"]
    best = max(
        candidates,
        key=lambda row: (float(row["ndcg_at_10"]), float(row["recall_at_10"]), float(row["coverage"])),
    )
    with (output_dir / "best_item_cf.json").open("w", encoding="utf-8") as output:
        json.dump(best, output, indent=2)
        output.write("\n")

    # Large searches remain fully available in CSV, but plots show a compact
    # comparison: baseline, each family's best trial, and the overall top five.
    plot_rows = rows
    if len(rows) > 20:
        selected = [
            next(row for row in rows if row["model"] == "bayesian_popularity")
        ]
        selected.extend(
            max(
                (row for row in candidates if row["similarity"] == similarity),
                key=lambda row: float(row["ndcg_at_10"]),
            )
            for similarity in SIMILARITIES
        )
        # Several local-grid settings can produce exactly the same retained
        # neighbors. Keep only one such outcome in compact plots while the CSV
        # still preserves every trial.
        unique_top: list[dict[str, object]] = []
        seen_outcomes: set[tuple[object, ...]] = set()
        for row in sorted(candidates, key=lambda row: float(row["ndcg_at_10"]), reverse=True):
            outcome = (
                row["similarity"],
                round(float(row["rmse"]), 6),
                round(float(row["ndcg_at_10"]), 6),
                round(float(row["coverage"]), 6),
            )
            if outcome in seen_outcomes:
                continue
            seen_outcomes.add(outcome)
            unique_top.append(row)
            if len(unique_top) == 5:
                break
        selected.extend(unique_top)
        plot_rows = list({int(row["trial"]): row for row in selected}.values())

    labels = [
        "Bayesian popularity"
        if row["model"] == "bayesian_popularity"
        else f"T{row['trial']} {row['similarity']}"
        for row in plot_rows
    ]
    positions = np.arange(len(plot_rows))
    # Separate axes keep the much larger coverage values from visually hiding
    # small but meaningful differences in Recall and NDCG.
    figure, axes = plt.subplots(3, 1, figsize=(max(11, len(plot_rows) * 1.2), 11), sharex=True)
    for axis, metric, label, color in (
        (axes[0], "recall_at_10", "Recall@10", "#4C78A8"),
        (axes[1], "ndcg_at_10", "NDCG@10", "#F58518"),
        (axes[2], "coverage", "Catalog coverage", "#54A24B"),
    ):
        values = [float(row[metric]) for row in plot_rows]
        axis.bar(positions, values, color=color)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        for x, value in zip(positions, values):
            axis.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Ranking Quality by Similarity and Search Trial")
    axes[2].set_xticks(positions, labels, rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(output_dir / "ranking_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for row, label in zip(plot_rows, labels):
        axis.scatter(float(row["rmse"]), float(row["ndcg_at_10"]), s=65)
        axis.annotate(label, (float(row["rmse"]), float(row["ndcg_at_10"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_title("Rating Error vs. Ranking Quality")
    axis.set_xlabel("RMSE (lower is better)")
    axis.set_ylabel("NDCG@10 (higher is better)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rmse_vs_ndcg.png", dpi=180)
    plt.close(figure)

    trial_rows = [row for row in rows if row["model"] == "item_cf"]
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = {name: color for name, color in zip(SIMILARITIES, ("#4C78A8", "#F58518", "#54A24B", "#E45756"))}
    axis.bar(
        [int(row["trial"]) for row in trial_rows],
        [float(row["ndcg_at_10"]) for row in trial_rows],
        color=[colors[str(row["similarity"])] for row in trial_rows],
    )
    axis.set_title("Seeded Random Search + Local Grid Refinement")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Validation NDCG@10")
    axis.legend(
        handles=[Patch(facecolor=colors[name], label=name) for name in SIMILARITIES],
        title="Similarity",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "search_trials.png", dpi=180)
    plt.close(figure)

    # One panel per similarity exposes how support, shrinkage, and neighbor K
    # affect ranking quality. Marker area represents K; color represents the
    # minimum number of common raters required by that trial.
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for axis, similarity in zip(axes.ravel(), SIMILARITIES):
        family = [row for row in trial_rows if row["similarity"] == similarity]
        scatter = axis.scatter(
            [float(row["shrinkage"]) for row in family],
            [float(row["ndcg_at_10"]) for row in family],
            s=[35 + float(row["neighbors"]) * 1.2 for row in family],
            c=[float(row["min_overlap"]) for row in family],
            cmap="viridis",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )
        label_counts: dict[tuple[float, float], int] = {}
        for row in family:
            point = (float(row["shrinkage"]), round(float(row["ndcg_at_10"]), 8))
            label_index = label_counts.get(point, 0)
            label_counts[point] = label_index + 1
            axis.annotate(
                f"T{row['trial']}",
                (float(row["shrinkage"]), float(row["ndcg_at_10"])),
                xytext=(4, 4 + label_index * 9),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(similarity)
        axis.set_xscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)
    figure.supxlabel("Shrinkage (symmetric log scale)")
    figure.supylabel("Validation NDCG@10")
    figure.suptitle("Hyperparameter Effects: marker size = neighbors, color = minimum overlap")
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label="Minimum common raters")
    figure.savefig(output_dir / "hyperparameter_effects.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    baseline = next(row for row in rows if row["model"] == "bayesian_popularity")
    comparison = "higher" if float(best["ndcg_at_10"]) > float(baseline["ndcg_at_10"]) else "lower"
    report_name = (
        "ItemCF reduced-dataset validation summary"
        if full_validation
        else "ItemCF smoke experiment summary"
    )
    scope_note = (
        "This report uses all users in the reduced assignment dataset for training and validation."
        if full_validation
        else "This report is a validation smoke run on a deterministic user sample."
    )
    next_step = (
        "The winning configuration can now be frozen before one final test evaluation."
        if full_validation
        else "The next step is a larger validation run and then a targeted grid around the stable winner."
    )
    comparison_scope = "reduced validation dataset" if full_validation else "sample"
    summary = f"""# {report_name}

{scope_note} The test split was not read. {len(candidates)} ItemCF configurations
were evaluated in addition to the Bayesian popularity baseline.

## Best ItemCF configuration

- Similarity: `{best['similarity']}`
- Scoring: `{best['scoring']}`
- Neighbors: `{best['neighbors']}`
- Minimum overlap: `{best['min_overlap']}`
- Shrinkage: `{best['shrinkage']}`
- RMSE: {float(best['rmse']):.4f}
- Recall@10: {float(best['recall_at_10']):.4f}
- NDCG@10: {float(best['ndcg_at_10']):.4f}
- Catalog coverage: {float(best['coverage']):.4f}

The best ItemCF NDCG@10 is {comparison} than the Bayesian popularity control in this
{comparison_scope} ({float(baseline['ndcg_at_10']):.4f}). This is a useful diagnostic, not a
reason to inspect the test set. {next_step}

## Visual outputs

- `ranking_metrics.png`: Recall, NDCG, and coverage on separate axes.
- `rmse_vs_ndcg.png`: rating-prediction error versus ranking quality.
- `search_trials.png`: random-search and local-grid validation trials.
- `hyperparameter_effects.png`: shrinkage, neighbor K, and support effects by similarity.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> list[dict[str, object]]:
    if not 0 < args.sample_users <= args.total_users:
        raise ValueError("sample_users must be between 1 and total_users")
    # Step 1: load the deterministic user cohort from train and validation only.
    sampled_users = sample_user_ids(args.total_users, args.sample_users, args.seed)
    shape = (args.total_users, args.total_items)
    print(f"Loading {args.sample_users:,} sampled users from complete CSV files...")
    train = load_interactions(args.train, shape=shape, user_filter=sampled_users).ratings
    validation = load_interactions(args.validation, shape=shape, user_filter=sampled_users).ratings
    relevant, validation_users = validation_targets(validation, args.relevant_threshold)
    if not relevant:
        raise ValueError("No sampled validation user has a relevant item; increase sample size")

    # Step 2: evaluate the Bayesian control under the same candidate protocol.
    rows: list[dict[str, object]] = [
        bayesian_popularity_result(
            train, validation, relevant, validation_users, args.top_k
        )
    ]
    # Keep one rectangular CSV schema even though the Bayesian baseline has no CF options.
    for field in asdict(ItemCFConfig()):
        rows[0][field] = ""

    # Step 3: run the balanced similarity search and record every validation trial.
    configs = build_trial_configs(args.trials, args.seed, args.sample_users, args.total_users)

    def run_config(trial: int, stage: str, config: ItemCFConfig) -> None:
        print(f"{stage.title()} trial {trial}: {config}")
        fit_started = time.perf_counter()
        model = ItemItemCF(config).fit(train)
        fit_seconds = time.perf_counter() - fit_started
        evaluation_started = time.perf_counter()
        metrics, _ = evaluate_model(
            model, validation, relevant, validation_users, args.top_k
        )
        row: dict[str, object] = {
            "trial": trial,
            "stage": stage,
            "model": "item_cf",
            **asdict(config),
            **metrics,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - evaluation_started,
        }
        rows.append(row)

    for trial, config in enumerate(configs):
        run_config(trial, "random", config)

    # Step 4: refine the random-search winner with a small local parameter grid.
    random_winner_row = max(
        (row for row in rows if row.get("stage") == "random"),
        key=lambda row: (float(row["ndcg_at_10"]), float(row["recall_at_10"]), float(row["coverage"])),
    )
    random_winner = ItemCFConfig(
        **{field: random_winner_row[field] for field in asdict(ItemCFConfig())}
    )
    refinements = build_local_grid(
        random_winner,
        sample_users=args.sample_users,
        total_users=args.total_users,
        max_trials=args.grid_trials,
    )
    for offset, config in enumerate(refinements, start=len(configs)):
        run_config(offset, "grid", config)

    # Step 5: normalize the result schema and persist the frozen validation winner.
    # Dict insertion differs for the first baseline row; normalize order here.
    ordered_fields = [
        "trial", "stage", "model", *asdict(ItemCFConfig()).keys(), "rmse", "recall_at_10",
        "ndcg_at_10", "coverage", "fit_seconds", "evaluation_seconds",
        "rating_predictions", "ranking_users",
    ]
    normalized = [{field: row.get(field, "") for field in ordered_fields} for row in rows]
    write_results(
        normalized,
        args.output_dir,
        full_validation=args.sample_users == args.total_users,
    )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/experiment/validation.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/item_cf"))
    parser.add_argument("--total-users", type=int, default=20_000)
    parser.add_argument("--total-items", type=int, default=2_000)
    parser.add_argument("--sample-users", type=int, default=20_000)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--grid-trials", type=int, default=6)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        choices=(10,),
        help="Fixed at 10 because the assignment reports Recall@10 and NDCG@10.",
    )
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_experiment(args)
    best = max(
        (row for row in rows if row["model"] == "item_cf"),
        key=lambda row: float(row["ndcg_at_10"]),
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
