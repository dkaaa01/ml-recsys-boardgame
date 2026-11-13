"""Compare frozen Top-10 rankings on the shared experiment test set.

Run from the repository root with python -m src.experiments.final_model_comparison.
The two outputs go to reports/comparison/.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from itertools import islice
from pathlib import Path

from src.metrics import catalog_coverage, ndcg_at_k, recall_at_k, validate_recommendations


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "experiment"
RANKINGS = ROOT / "outputs" / "top10_test.csv"
REPORT = ROOT / "reports" / "comparison"
TOP_K = 10
LIKED_RATING = 8.0
LABELS = {
    "bayesian_popularity": "Bayesian popularity",
    "hybrid": "Hybrid",
    "item_cf": "ItemCF",
    "content": "Content-based",
    "mf": "Matrix factorization",
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def load_test_protocol():
    """Check the cohort and holdout before scoring any recommendations."""
    catalog = {int(row["item_id"]): row["bgg_id"] for row in read_rows(DATA / "games.csv")}
    expected_positives = {
        int(row["user_id"]): int(row["test_relevant_count"])
        for row in read_rows(DATA / "eval_users.csv")
    }
    seen = defaultdict(set)
    counts = Counter()
    for row in read_rows(DATA / "train.csv"):
        user, item = int(row["user_id"]), int(row["item_id"])
        if item not in catalog or item in seen[user]:
            raise ValueError(f"Invalid or duplicate training pair: {(user, item)}")
        seen[user].add(item)
        counts[item] += 1

    test_items = defaultdict(set)
    relevant = defaultdict(set)
    for row in read_rows(DATA / "test.csv"):
        user, item = int(row["user_id"]), int(row["item_id"])
        if user not in expected_positives or item not in catalog:
            raise ValueError(f"Test pair outside the common evaluation data: {(user, item)}")
        if item in seen[user] or item in test_items[user]:
            raise ValueError(f"Overlapping or duplicate test pair: {(user, item)}")
        test_items[user].add(item)
        if float(row["rating"]) >= LIKED_RATING:
            relevant[user].add(item)

    if not expected_positives or {u: len(items) for u, items in relevant.items()} != expected_positives:
        raise ValueError("Test positives do not match eval_users.csv.")
    return catalog, seen, counts, relevant


def load_model_rankings():
    expected_models = set(LABELS) - {"bayesian_popularity"}
    ranked = defaultdict(lambda: defaultdict(dict))
    for row in read_rows(RANKINGS):
        model = row["model"]
        user, item, rank = (int(row[key]) for key in ("user_id", "item_id", "rank"))
        if model not in expected_models or not 1 <= rank <= TOP_K or rank in ranked[model][user]:
            raise ValueError(f"Invalid or duplicate recommendation rank: {(model, user, rank)}")
        ranked[model][user][rank] = item

    if set(ranked) != expected_models:
        raise ValueError("The recommendation file must contain all four personalized models.")
    return {
        model: {user: [items[rank] for rank in sorted(items)] for user, items in users.items()}
        for model, users in ranked.items()
    }


def popularity_rankings(catalog, seen, counts, users):
    """Reuse the team's training-only Bayesian ranking and apply the common seen-item filter."""
    rows = sorted(read_rows(DATA / "popularity.csv"), key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in rows] != list(range(1, len(catalog) + 1)):
        raise ValueError("The popularity list does not rank the complete common catalog.")
    if {int(row["item_id"]): row["bgg_id"] for row in rows} != catalog:
        raise ValueError("The popularity file uses a different game catalog.")
    if any(int(row["rating_count"]) != counts[int(row["item_id"])] for row in rows):
        raise ValueError("Popularity rating counts do not match the training split.")

    ordered = [int(row["item_id"]) for row in rows]
    return {
        user: list(islice((item for item in ordered if item not in seen[user]), TOP_K))
        for user in users
    }


def evaluate():
    catalog, seen, counts, relevant = load_test_protocol()
    rankings = load_model_rankings()
    rankings["bayesian_popularity"] = popularity_rankings(catalog, seen, counts, relevant)

    results = []
    for model, label in LABELS.items():
        recommendations = rankings[model]
        if recommendations.keys() != relevant.keys():
            raise ValueError(f"{label} does not cover the shared evaluation cohort.")
        if any(len(items) != TOP_K for items in recommendations.values()):
            raise ValueError(f"{label} does not have exactly {TOP_K} results per user.")
        validate_recommendations(recommendations, seen, k=TOP_K)
        results.append({
            "model": label,
            "recall_at_10": recall_at_k(recommendations, relevant, TOP_K),
            "ndcg_at_10": ndcg_at_k(recommendations, relevant, TOP_K),
            "coverage": catalog_coverage(recommendations, catalog, k=TOP_K),
        })
    return results, len(relevant), len(catalog)


def format_coverage(value):
    percent = Decimal(str(value)) * 100
    return f"{percent.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def save_chart(results, users, games, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.4), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.195, right=.975, top=.77, bottom=.16, wspace=.37)
    fig.text(.035, .925, "Top-10 recommendation performance", fontsize=23, weight="bold", color="#172730")
    fig.text(.036, .857, f"Held-out test  |  {users:,} users  |  {games:,} games  |  Relevant ratings ≥ 8",
             fontsize=12, color="#475B65")

    labels = [row["model"] for row in results]
    colors = ["#485E73", "#087C72", "#A4B1BF", "#A4B1BF", "#A4B1BF"]
    panels = (("recall_at_10", "Recall@10"), ("ndcg_at_10", "NDCG@10"), ("coverage", "Catalog coverage"))
    for column, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        values = [row[metric] for row in results]
        ax.barh(range(len(labels)), values, color=colors, height=.52)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.34)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", color="#172730", pad=10)
        ax.set_yticks(range(len(labels)), labels if column == 0 else [""] * len(labels), fontsize=11)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for rank, value in enumerate(values):
            display = format_coverage(value) if metric == "coverage" else f"{value:.4f}"
            ax.text(value + max(values) * .035, rank, display, va="center", fontsize=10.5, color="#172730")
    fig.text(.036, .085, "Coverage is the share of the game catalog shown at least once across all users.",
             fontsize=10.5, color="#475B65")
    fig.savefig(path, dpi=180)


def main():
    results, users, games = evaluate()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "test_metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_chart(results, users, games, REPORT / "test_comparison.png")

    print(f"Test cohort: {users:,} users, {games:,} games")
    print(f"{'Model':23} {'Recall@10':>10} {'NDCG@10':>10} {'Coverage':>10}")
    for row in results:
        print(f"{row['model']:23} {row['recall_at_10']:10.4f} {row['ndcg_at_10']:10.4f} {format_coverage(row['coverage']):>10}")
    print(f"Saved comparison to {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
