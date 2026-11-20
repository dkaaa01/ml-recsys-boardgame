"""Popularity baseline based on Bayesian weighted ratings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def percentile(values: list[int], quantile: float) -> float:
    """Interpolate a training-count quantile between adjacent sorted values.

    Args:
        values: Non-empty list of observed item rating counts.
        quantile: Fraction in [0, 1], such as 0.75 for the 75th percentile.
    Returns:
        Linearly interpolated count, which can be non-integral.
    Raises:
        ValueError: If values is empty."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_game_names(games_path: Path) -> dict[int, tuple[str, str]]:
    """Read the metadata identifiers and display names used in a ranking.

    Args:
        games_path: UTF-8 CSV containing item_id, bgg_id and name.
    Returns:
        Mapping from each internal integer item ID to (BGG ID, game name)."""
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
    """Rank games by a training-only Bayesian weighted mean.

    For count n, item mean R, training global mean C and prior count m, the
    score is n*R/(n+m) + m*C/(n+m). Here m is the specified quantile of
    training item counts. This is a shrinkage baseline, not BGG's official rank.

    Args:
        train_path: CSV with user_id, item_id and rating; only training is read.
        games_path: CSV containing the metadata for the same item IDs.
        prior_quantile: Quantile in [0, 1] used to set the prior count m.
    Returns:
        Ordered ranking rows and the global mean/prior settings. Only games
        with a training rating are ranked. Ties use count, then game name.
    Raises:
        ValueError: If train_path contains no ratings."""
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
    *,
    csv_output: Path | None = None,
) -> None:
    """Save a readable ranking and the constants used to calculate it.

    Args:
        ranking: Ordered rows returned by rank_popular_games.
        metadata: Global mean, prior count and quantile for those rows.
        output_dir: Destination for popular_games.csv and popularity_report.json.
        csv_output: Optional second path for the same ranking CSV.
    Returns:
        None. Parent directories are created when required."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_popularity_csv(ranking, output_dir / "popular_games.csv")
    if csv_output is not None:
        _write_popularity_csv(ranking, csv_output)
    with (output_dir / "popularity_report.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")


def _write_popularity_csv(
    ranking: list[dict[str, object]], output_path: Path
) -> None:
    """Write a ranking with one-based ranks and six-decimal score columns.

    Args:
        ranking: Ordered dictionaries containing the popularity output fields.
        output_path: CSV destination; its parent directory is created.
    Returns:
        None. The file is overwritten with a header and all ranking rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = [
            "rank",
            "item_id",
            "bgg_id",
            "name",
            "rating_count",
            "mean_rating",
            "popularity_score",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
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


def recommend_unseen_games(
    ranking: list[dict[str, object]],
    seen_item_ids: set[int],
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Select the highest-ranked games absent from the user's training history.

    Args:
        ranking: Global popularity rows already in descending score order.
        seen_item_ids: Internal IDs already rated by this user in training.
        top_k: Positive maximum number of rows to return.
    Returns:
        Ranking rows in their original order, up to top_k when available."""
    recommendations: list[dict[str, object]] = []
    for row in ranking:
        if int(row["item_id"]) in seen_item_ids:
            continue
        recommendations.append(row)
        if len(recommendations) == top_k:
            break
    return recommendations


def main() -> None:
    """Read command-line paths and prior quantile, write the ranking, and print its top 10.

    No Python arguments are accepted. Run python -m src.models.popularity
    --help for the CSV input and output options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=Path("data/experiment/train.csv")
    )
    parser.add_argument(
        "--games", type=Path, default=Path("data/experiment/games.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/experiment/popularity.csv"),
        help="Team handoff CSV written from the same experiment ranking.",
    )
    parser.add_argument("--prior-quantile", type=float, default=0.75)
    args = parser.parse_args()
    ranking, metadata = rank_popular_games(args.train, args.games, args.prior_quantile)
    write_popularity_outputs(
        ranking, metadata, args.output_dir, csv_output=args.csv_output
    )
    for rank, row in enumerate(ranking[:10], 1):
        print(
            f"{rank:2d}. {row['name']} | score={row['popularity_score']:.3f} "
            f"| mean={row['mean_rating']:.3f} | n={row['rating_count']}"
        )


if __name__ == "__main__":
    main()

