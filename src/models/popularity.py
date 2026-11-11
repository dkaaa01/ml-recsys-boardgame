"""Popularity baseline based on Bayesian weighted ratings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def percentile(values: list[int], quantile: float) -> float:
    """Calculate a percentile without an external dependency."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_game_names(games_path: Path) -> dict[int, tuple[str, str]]:
    """Map the internal item ID to the BGG ID and game name."""
    games: dict[int, tuple[str, str]] = {}
    with games_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            games[int(row["item_id"])] = (row["bgg_id"], row["name"])
    return games


def rank_popular_games(
    train_path: Path,
    games_path: Path,
    prior_quantile: float = 0.75,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Rank games using only training ratings.

    The Bayesian score lowers the rank of games with very few ratings while
    keeping highly rated and well-supported games near the top.
    """
    rating_sum: dict[int, float] = defaultdict(float)
    rating_count: dict[int, int] = defaultdict(int)
    global_sum = 0.0
    global_count = 0
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            item_id = int(row["item_id"])
            rating = float(row["rating"])
            rating_sum[item_id] += rating
            rating_count[item_id] += 1
            global_sum += rating
            global_count += 1

    if not global_count:
        raise ValueError("The training file has no ratings.")
    global_mean = global_sum / global_count
    prior_count = percentile(list(rating_count.values()), prior_quantile)
    games = load_game_names(games_path)
    ranking: list[dict[str, object]] = []
    for item_id, count in rating_count.items():
        mean_rating = rating_sum[item_id] / count
        weighted_rating = (
            count / (count + prior_count) * mean_rating
            + prior_count / (count + prior_count) * global_mean
        )
        bgg_id, name = games[item_id]
        ranking.append(
            {
                "item_id": item_id,
                "bgg_id": bgg_id,
                "name": name,
                "rating_count": count,
                "mean_rating": mean_rating,
                "popularity_score": weighted_rating,
            }
        )
    ranking.sort(
        key=lambda row: (-float(row["popularity_score"]), -int(row["rating_count"]), str(row["name"]))
    )
    metadata = {
        "global_mean_rating": global_mean,
        "prior_rating_count": prior_count,
        "prior_quantile": prior_quantile,
    }
    return ranking, metadata


def write_popularity_outputs(
    ranking: list[dict[str, object]],
    metadata: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "popular_games.csv").open("w", encoding="utf-8", newline="") as output:
        fieldnames = [
            "rank",
            "item_id",
            "bgg_id",
            "name",
            "rating_count",
            "mean_rating",
            "popularity_score",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranking, 1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                    "mean_rating": f"{float(row['mean_rating']):.6f}",
                    "popularity_score": f"{float(row['popularity_score']):.6f}",
                }
            )
    with (output_dir / "popularity_report.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")


def recommend_unseen_games(
    ranking: list[dict[str, object]],
    seen_item_ids: set[int],
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Return the highest-ranked games that the user has not rated."""
    recommendations: list[dict[str, object]] = []
    for row in ranking:
        if int(row["item_id"]) in seen_item_ids:
            continue
        recommendations.append(row)
        if len(recommendations) == top_k:
            break
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--games", type=Path, default=Path("data/processed/games.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--prior-quantile", type=float, default=0.75)
    args = parser.parse_args()
    ranking, metadata = rank_popular_games(args.train, args.games, args.prior_quantile)
    write_popularity_outputs(ranking, metadata, args.output_dir)
    for rank, row in enumerate(ranking[:10], 1):
        print(
            f"{rank:2d}. {row['name']} | score={row['popularity_score']:.3f} "
            f"| mean={row['mean_rating']:.3f} | n={row['rating_count']}"
        )


if __name__ == "__main__":
    main()

