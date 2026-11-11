"""Create EDA summaries and plots from the processed dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path

# Keep Matplotlib and font caches inside the project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt

from src.models.popularity import rank_popular_games


def to_float(value: str) -> float | None:
    """Convert a CSV value to float and treat zero as missing metadata."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def list_length(value: str) -> int:
    """Count values stored as a Python-style list string."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def collect_interaction_stats(processed_dir: Path) -> dict[str, object]:
    """Collect counts without loading all interactions into memory."""
    user_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    rating_counts: Counter[float] = Counter()
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        count = 0
        with (processed_dir / f"{split}.csv").open("r", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                user_counts[int(row["user_id"])] += 1
                item_counts[int(row["item_id"])] += 1
                rating_counts[float(row["rating"])] += 1
                count += 1
        split_counts[split] = count
    possible = len(user_counts) * len(item_counts)
    total = sum(split_counts.values())
    return {
        "user_counts": user_counts,
        "item_counts": item_counts,
        "rating_counts": rating_counts,
        "split_counts": split_counts,
        "users": len(user_counts),
        "items": len(item_counts),
        "interactions": total,
        "matrix_sparsity": 1 - total / possible,
    }


def collect_metadata_stats(games_path: Path) -> dict[str, list[float]]:
    """Collect numeric metadata used by EDA plots."""
    result = {
        "year_published": [],
        "complexity": [],
        "max_playtime": [],
        "category_count": [],
        "mechanic_count": [],
    }
    with games_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            for name in ("year_published", "complexity", "max_playtime"):
                value = to_float(row[name])
                if value is not None:
                    result[name].append(value)
            result["category_count"].append(float(list_length(row["categories"])))
            result["mechanic_count"].append(float(list_length(row["mechanics"])))
    return result


def save_plots(
    interaction_stats: dict[str, object],
    metadata_stats: dict[str, list[float]],
    popular_ranking: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Save the main EDA figures as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(list(interaction_stats["user_counts"].values()), bins=60, color="#4C78A8")
    axes[0].set(title="Ratings per User", xlabel="Number of ratings", ylabel="Users", yscale="log")
    axes[1].hist(list(interaction_stats["item_counts"].values()), bins=60, color="#F58518")
    axes[1].set(title="Ratings per Game", xlabel="Number of ratings", ylabel="Games", yscale="log")
    fig.tight_layout()
    fig.savefig(output_dir / "interaction_count_distributions.png", dpi=160)
    plt.close(fig)

    ratings = interaction_stats["rating_counts"]
    # Group decimal ratings into 0.5-point bins for a readable chart.
    rating_bins: Counter[float] = Counter()
    for rating, count in ratings.items():
        rating_bins[round(float(rating) * 2) / 2] += count
    x = sorted(rating_bins)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, [rating_bins[value] for value in x], width=0.42, color="#54A24B")
    ax.set(title="Rating Distribution", xlabel="Rating", ylabel="Interactions")
    fig.tight_layout()
    fig.savefig(output_dir / "rating_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    valid_years = [year for year in metadata_stats["year_published"] if 1950 <= year <= 2025]
    axes[0, 0].hist(valid_years, bins=30, range=(1950, 2025), color="#4C78A8")
    axes[0, 0].set(title="Year Published", xlabel="Year", ylabel="Games", xlim=(1950, 2025))
    axes[0, 1].hist(metadata_stats["complexity"], bins=30, color="#E45756")
    axes[0, 1].set(title="Game Complexity", xlabel="Complexity (1-5)", ylabel="Games")
    axes[1, 0].hist(metadata_stats["max_playtime"], bins=50, range=(0, 300), color="#72B7B2")
    axes[1, 0].set(title="Maximum Playtime", xlabel="Minutes (up to 300)", ylabel="Games")
    axes[1, 1].scatter(
        metadata_stats["category_count"], metadata_stats["mechanic_count"],
        s=10, alpha=0.35, color="#B279A2",
    )
    axes[1, 1].set(title="Categories and Mechanics", xlabel="Category count", ylabel="Mechanic count")
    fig.tight_layout()
    fig.savefig(output_dir / "game_metadata_distributions.png", dpi=160)
    plt.close(fig)

    top = list(reversed(popular_ranking[:10]))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([str(row["name"]) for row in top], [float(row["popularity_score"]) for row in top], color="#4C78A8")
    ax.set(title="Top 10 Popular Games (Training Data Only)", xlabel="Bayesian popularity score")
    fig.tight_layout()
    fig.savefig(output_dir / "top_10_popular_games.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    interaction_stats = collect_interaction_stats(args.processed_dir)
    metadata_stats = collect_metadata_stats(args.processed_dir / "games.csv")
    popular_ranking, _ = rank_popular_games(
        args.processed_dir / "train.csv", args.processed_dir / "games.csv"
    )
    save_plots(interaction_stats, metadata_stats, popular_ranking, args.output_dir)

    summary = {
        key: value
        for key, value in interaction_stats.items()
        if key not in {"user_counts", "item_counts", "rating_counts"}
    }
    summary["average_ratings_per_user"] = (
        interaction_stats["interactions"] / interaction_stats["users"]
    )
    summary["average_ratings_per_game"] = (
        interaction_stats["interactions"] / interaction_stats["items"]
    )
    with (args.output_dir / "eda_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

