"""Create a compact, leakage-safe dataset for recommendation experiments.

Selection statistics are computed from the training split only.  The selected
user and item IDs are then applied unchanged to validation and test, so future
information cannot influence which entities enter the experiment dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def count_train_items(train_path: Path) -> Counter[int]:
    """Count training interactions per item without loading the CSV into memory."""

    counts: Counter[int] = Counter()
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            counts[int(row["item_id"])] += 1
    return counts


def count_train_users(train_path: Path, selected_items: set[int]) -> Counter[int]:
    """Count each user's training ratings inside the reduced game catalog."""

    counts: Counter[int] = Counter()
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            item_id = int(row["item_id"])
            if item_id in selected_items:
                counts[int(row["user_id"])] += 1
    return counts


def activity_bin(rating_count: int) -> str:
    """Group users by activity so sampling does not retain only heavy users."""

    if rating_count < 40:
        return "20-39"
    if rating_count < 80:
        return "40-79"
    return "80+"


def stratified_user_sample(
    eligible_counts: dict[int, int], sample_size: int, seed: int
) -> tuple[list[int], dict[str, dict[str, int]]]:
    """Sample users proportionally from activity bins with deterministic rounding."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(eligible_counts) < sample_size:
        raise ValueError(
            f"Only {len(eligible_counts):,} users satisfy the filter; "
            f"cannot sample {sample_size:,}."
        )

    groups: dict[str, list[int]] = defaultdict(list)
    for user_id, count in eligible_counts.items():
        groups[activity_bin(count)].append(user_id)
    for users in groups.values():
        users.sort()

    total = len(eligible_counts)
    exact = {name: sample_size * len(users) / total for name, users in groups.items()}
    allocations = {name: int(value) for name, value in exact.items()}
    remaining = sample_size - sum(allocations.values())
    # Largest-remainder allocation preserves the source distribution while
    # guaranteeing that allocations sum exactly to the requested sample size.
    order = sorted(groups, key=lambda name: (-(exact[name] - allocations[name]), name))
    for name in order:
        if remaining == 0:
            break
        if allocations[name] < len(groups[name]):
            allocations[name] += 1
            remaining -= 1

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    report: dict[str, dict[str, int]] = {}
    for name in sorted(groups):
        users = np.asarray(groups[name], dtype=np.int64)
        chosen = rng.choice(users, allocations[name], replace=False)
        selected.extend(map(int, chosen))
        report[name] = {"eligible": len(users), "sampled": allocations[name]}
    selected.sort()
    return selected, report


def write_filtered_split(
    source_path: Path,
    output_path: Path,
    user_mapping: dict[int, int],
    item_mapping: dict[int, int],
) -> int:
    """Filter one split and compact source IDs to contiguous experiment IDs."""

    written = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        reader = csv.DictReader(source)
        writer = csv.writer(output)
        writer.writerow(("user_id", "item_id", "rating"))
        for row in reader:
            old_user = int(row["user_id"])
            old_item = int(row["item_id"])
            if old_user not in user_mapping or old_item not in item_mapping:
                continue
            writer.writerow(
                (user_mapping[old_user], item_mapping[old_item], row["rating"])
            )
            written += 1
    return written


def write_games(
    source_path: Path, output_path: Path, item_mapping: dict[int, int]
) -> int:
    """Write selected metadata and retain the original item ID for traceability."""

    written = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No header in {source_path}")
        fields = ["item_id", "source_item_id", *[f for f in reader.fieldnames if f != "item_id"]]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            old_item = int(row["item_id"])
            if old_item not in item_mapping:
                continue
            writer.writerow(
                {
                    "item_id": item_mapping[old_item],
                    "source_item_id": old_item,
                    **{field: row[field] for field in reader.fieldnames if field != "item_id"},
                }
            )
            written += 1
    return written


def write_users(
    output_path: Path,
    selected_users: list[int],
    user_mapping: dict[int, int],
    train_counts: Counter[int],
) -> None:
    """Write the compact-to-source user mapping and activity information."""

    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ("user_id", "source_user_id", "train_rating_count", "activity_bin")
        )
        for old_user in selected_users:
            count = train_counts[old_user]
            writer.writerow((user_mapping[old_user], old_user, count, activity_bin(count)))


def create_experiment_subset(
    processed_dir: Path,
    output_dir: Path,
    *,
    max_items: int = 1_500,
    max_users: int = 10_000,
    min_user_train_ratings: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    """Create filtered splits, compact mappings, metadata, and a JSON report."""

    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"
    games_path = processed_dir / "games.csv"
    for path in (train_path, validation_path, test_path, games_path):
        if not path.exists():
            raise FileNotFoundError(path)

    item_counts = count_train_items(train_path)
    ranked_items = sorted(item_counts, key=lambda item: (-item_counts[item], item))
    selected_items = ranked_items[:max_items]
    if len(selected_items) < max_items:
        raise ValueError(f"Only {len(selected_items)} training items are available")

    train_user_counts = count_train_users(train_path, set(selected_items))
    eligible = {
        user_id: count
        for user_id, count in train_user_counts.items()
        if count >= min_user_train_ratings
    }
    selected_users, strata = stratified_user_sample(eligible, max_users, seed)

    # Item IDs follow training popularity rank; user IDs follow source ID order.
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_items)}
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_users)}
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        split: write_filtered_split(
            processed_dir / f"{split}.csv",
            output_dir / f"{split}.csv",
            user_mapping,
            item_mapping,
        )
        for split in ("train", "validation", "test")
    }
    metadata_rows = write_games(games_path, output_dir / "games.csv", item_mapping)
    write_users(output_dir / "users.csv", selected_users, user_mapping, train_user_counts)

    report: dict[str, object] = {
        "selection_source": "train split only",
        "source_processed_dir": str(processed_dir),
        "users": len(selected_users),
        "items": len(selected_items),
        "min_user_train_ratings": min_user_train_ratings,
        "seed": seed,
        "eligible_users": len(eligible),
        "activity_strata": strata,
        "split_counts": split_counts,
        "interactions": sum(split_counts.values()),
        "metadata_rows": metadata_rows,
        "matrix_sparsity": 1 - sum(split_counts.values()) / (max_users * max_items),
    }
    with (output_dir / "subset_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiment_10k"))
    parser.add_argument("--max-items", type=int, default=1_500)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--min-user-train-ratings", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = create_experiment_subset(
        args.processed_dir,
        args.output_dir,
        max_items=args.max_items,
        max_users=args.max_users,
        min_user_train_ratings=args.min_user_train_ratings,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
