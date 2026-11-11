"""Streaming preprocessing for the BoardGameGeek Reviews dataset.

The implementation uses only Python's standard library. It reads the large CSV
sequentially and keeps only counters and ID maps in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


USER_ALIASES = ("user", "username")
ITEM_ALIASES = ("id", "bgg_id", "game_id")
RATING_ALIASES = ("rating", "score")
NAME_ALIASES = ("name", "game_name", "primary")


def normalized_header(name: str | None) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def resolve_column(fieldnames: Iterable[str | None], aliases: tuple[str, ...]) -> str:
    lookup = {normalized_header(field): field for field in fieldnames}
    for alias in aliases:
        if alias in lookup and lookup[alias] is not None:
            return str(lookup[alias])
    raise ValueError(f"Missing required column; expected one of {aliases}, got {list(fieldnames)}")


def stable_split(user: str, item: str, seed: int) -> str:
    key = f"{seed}\0{user}\0{item}".encode("utf-8")
    bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def locate_ratings(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("bgg-*-reviews.csv"))
    if not candidates:
        candidates = sorted(raw_dir.glob("*review*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No ratings CSV found under {raw_dir}")
    return candidates[-1]


def count_items(ratings_path: Path) -> tuple[Counter[str], dict[str, int]]:
    """Count valid ratings for each game."""
    items: Counter[str] = Counter()
    stats = {"raw_rows": 0, "invalid_rows": 0}
    with ratings_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {ratings_path}")
        user_col = resolve_column(reader.fieldnames, USER_ALIASES)
        item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
        rating_col = resolve_column(reader.fieldnames, RATING_ALIASES)
        for row in reader:
            stats["raw_rows"] += 1
            user = (row.get(user_col) or "").strip()
            item = (row.get(item_col) or "").strip()
            try:
                rating = float(row.get(rating_col) or "nan")
            except ValueError:
                rating = float("nan")
            if not user or not item or not 1 <= rating <= 10:
                stats["invalid_rows"] += 1
                continue
            items[item] += 1
    return items, stats


def load_metadata_ids(raw_dir: Path) -> set[str]:
    """Return the BGG IDs that have a metadata row."""
    metadata_path = raw_dir / "games_detailed_info.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Game metadata not found: {metadata_path}")
    metadata_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {metadata_path}")
        item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
        for row in reader:
            item = (row.get(item_col) or "").strip()
            if item:
                metadata_ids.add(item)
    return metadata_ids


def select_top_items(
    item_counts: Counter[str],
    metadata_ids: set[str],
    min_item_ratings: int,
    max_items: int,
) -> list[str]:
    """Select the most-rated games after metadata and count filters."""
    candidates = (
        (item, count)
        for item, count in item_counts.items()
        if count >= min_item_ratings and item in metadata_ids
    )
    ranked = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
    return [item for item, _ in ranked[:max_items]]


def count_users_for_items(ratings_path: Path, selected_items: set[str]) -> Counter[str]:
    """Count user ratings only inside the selected game catalog."""
    users: Counter[str] = Counter()
    with ratings_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {ratings_path}")
        user_col = resolve_column(reader.fieldnames, USER_ALIASES)
        item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
        rating_col = resolve_column(reader.fieldnames, RATING_ALIASES)
        for row in reader:
            item = (row.get(item_col) or "").strip()
            if item not in selected_items:
                continue
            user = (row.get(user_col) or "").strip()
            try:
                rating = float(row.get(rating_col) or "nan")
            except ValueError:
                continue
            if user and 1 <= rating <= 10:
                users[user] += 1
    return users


def write_interactions(
    ratings_path: Path,
    output_dir: Path,
    user_counts: Counter[str],
    item_counts: Counter[str],
    min_user_ratings: int,
    selected_items: list[str],
    seed: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_users = {user for user, count in user_counts.items() if count >= min_user_ratings}
    eligible_items = set(selected_items)
    user_ids: dict[str, int] = {}
    item_ids: dict[str, int] = {}
    split_counts = Counter()

    handles = {
        split: (output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
        for split in ("train", "validation", "test")
    }
    try:
        writers = {split: csv.writer(handle) for split, handle in handles.items()}
        for writer in writers.values():
            writer.writerow(("user_id", "item_id", "rating"))

        with ratings_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            assert reader.fieldnames is not None
            user_col = resolve_column(reader.fieldnames, USER_ALIASES)
            item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
            rating_col = resolve_column(reader.fieldnames, RATING_ALIASES)
            for row in reader:
                user = (row.get(user_col) or "").strip()
                item = (row.get(item_col) or "").strip()
                if user not in eligible_users or item not in eligible_items:
                    continue
                try:
                    rating = float(row.get(rating_col) or "nan")
                except ValueError:
                    continue
                if not 1 <= rating <= 10:
                    continue
                user_id = user_ids.setdefault(user, len(user_ids))
                item_id = item_ids.setdefault(item, len(item_ids))
                split = stable_split(user, item, seed)
                writers[split].writerow((user_id, item_id, f"{rating:g}"))
                split_counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    with (output_dir / "users.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("user_id", "username", "raw_rating_count"))
        for username, user_id in sorted(user_ids.items(), key=lambda pair: pair[1]):
            writer.writerow((user_id, username, user_counts[username]))

    with (output_dir / "items.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("item_id", "bgg_id", "raw_rating_count"))
        for bgg_id, item_id in sorted(item_ids.items(), key=lambda pair: pair[1]):
            writer.writerow((item_id, bgg_id, item_counts[bgg_id]))
    return user_ids, item_ids, dict(split_counts)


def clean_game_metadata(raw_dir: Path, output_dir: Path, item_ids: dict[str, int]) -> int:
    metadata_path = raw_dir / "games_detailed_info.csv"
    if not metadata_path.exists():
        return 0
    selected_aliases = {
        "name": ("primary", "name"),
        "year_published": ("yearpublished", "year_published"),
        "min_players": ("minplayers", "min_players"),
        "max_players": ("maxplayers", "max_players"),
        "min_playtime": ("minplaytime", "min_playtime"),
        "max_playtime": ("maxplaytime", "max_playtime"),
        "min_age": ("minage", "min_age"),
        "average_rating": ("average", "average_rating"),
        "bayes_average_rating": ("bayesaverage", "bayes_average_rating"),
        "complexity": ("averageweight", "complexity"),
        "users_rated": ("usersrated", "users_rated"),
        "categories": ("boardgamecategory", "categories"),
        "mechanics": ("boardgamemechanic", "mechanics"),
    }
    written = 0
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source, (
        output_dir / "games.csv"
    ).open("w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            return 0
        item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
        normalized = {normalized_header(field): str(field) for field in reader.fieldnames}
        columns = {
            output_name: next((normalized[a] for a in aliases if a in normalized), None)
            for output_name, aliases in selected_aliases.items()
        }
        writer = csv.writer(output)
        writer.writerow(("item_id", "bgg_id", *columns.keys()))
        for row in reader:
            bgg_id = (row.get(item_col) or "").strip()
            if bgg_id not in item_ids:
                continue
            writer.writerow(
                (item_ids[bgg_id], bgg_id, *(row.get(column, "") if column else "" for column in columns.values()))
            )
            written += 1
    return written


def preprocess(
    raw_dir: Path,
    output_dir: Path,
    min_user_ratings: int = 10,
    min_item_ratings: int = 100,
    max_items: int = 3000,
    seed: int = 42,
) -> dict[str, object]:
    ratings_path = locate_ratings(raw_dir)
    item_counts, stats = count_items(ratings_path)
    metadata_ids = load_metadata_ids(raw_dir)
    selected_items = select_top_items(item_counts, metadata_ids, min_item_ratings, max_items)
    if len(selected_items) < max_items:
        raise ValueError(
            f"Only {len(selected_items)} games satisfy the filters; expected {max_items}."
        )
    selected_item_set = set(selected_items)
    user_counts = count_users_for_items(ratings_path, selected_item_set)
    user_ids, item_ids, split_counts = write_interactions(
        ratings_path,
        output_dir,
        user_counts,
        item_counts,
        min_user_ratings,
        selected_items,
        seed,
    )
    metadata_rows = clean_game_metadata(raw_dir, output_dir, item_ids)
    report: dict[str, object] = {
        "source": str(ratings_path),
        **stats,
        "users_in_selected_catalog": len(user_counts),
        "raw_items": len(item_counts),
        "metadata_items": len(metadata_ids),
        "items_with_min_ratings": sum(
            count >= min_item_ratings for count in item_counts.values()
        ),
        "items_removed_for_missing_metadata": sum(
            count >= min_item_ratings and item not in metadata_ids
            for item, count in item_counts.items()
        ),
        "selected_top_items": len(selected_items),
        "processed_users": len(user_ids),
        "processed_items": len(item_ids),
        "processed_game_metadata_rows": metadata_rows,
        "split_counts": split_counts,
        "min_user_ratings": min_user_ratings,
        "min_item_ratings": min_item_ratings,
        "max_items": max_items,
        "split_seed": seed,
        "split_method": "deterministic 80/10/10 hash by (user, item)",
    }
    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-user-ratings", type=int, default=10)
    parser.add_argument("--min-item-ratings", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = preprocess(
        args.raw_dir,
        args.output_dir,
        args.min_user_ratings,
        args.min_item_ratings,
        args.max_items,
        args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

