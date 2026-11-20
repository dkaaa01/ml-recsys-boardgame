"""Create aligned subsets for warm-start recommendation experiments.

Training counts rank games and define activity strata. If an evaluation-user
file exists, positive counts in every split also determine user eligibility.
The same compact user/item mappings are applied to all splits. These choices
condition the evaluated cohort; test ratings never enter model fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def count_train_items(train_path: Path) -> Counter[int]:
    """Count ratings per game in the input training CSV.

    train_path contains user_id/item_id/rating rows. Return a Counter keyed by
    integer item IDs. Only train rows determine the reduced catalog order.
    """

    counts: Counter[int] = Counter()
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            counts[int(row["item_id"])] += 1
    return counts


def count_train_users(train_path: Path, selected_items: set[int]) -> Counter[int]:
    """Count each user's training history within a fixed game catalog.

    train_path is the training CSV and selected_items is the allowed item-ID set.
    Return a Counter mapping original user IDs to their retained training counts.
    """

    counts: Counter[int] = Counter()
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            item_id = int(row["item_id"])
            if item_id in selected_items:
                counts[int(row["user_id"])] += 1
    return counts


def load_eval_users(path: Path) -> set[int] | None:
    """Load the source dataset's ranking-evaluation user IDs.

    path is eval_users.csv. Return its integer user-ID set, or None when the file
    is absent and no per-split positive eligibility restriction can be applied.
    """

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as source:
        return {int(row["user_id"]) for row in csv.DictReader(source)}


def count_relevant_by_split(
    processed_dir: Path,
    selected_items: set[int],
    users: set[int],
    threshold: float,
) -> dict[str, Counter[int]]:
    """Count positive ratings per user inside a selected game catalog.

    Parameters
    ----------
    processed_dir : Path
        Directory containing train.csv, validation.csv and test.csv.
    selected_items : set[int]
        Allowed original game IDs.
    users : set[int]
        Original user IDs whose relevant counts are required.
    threshold : float
        Inclusive minimum rating considered positive, 8.0 in this project.

    Returns
    -------
    dict[str, Counter[int]]
        Per-user positive counts for each split. Validation/test labels are read
        only to define users with an evaluable positive in each split; these
        counts do not become model input or determine game popularity ranks.
    """

    result: dict[str, Counter[int]] = {}
    for split in ("train", "validation", "test"):
        counts: Counter[int] = Counter()
        with (processed_dir / f"{split}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            for row in csv.DictReader(source):
                user_id = int(row["user_id"])
                if (
                    user_id in users
                    and int(row["item_id"]) in selected_items
                    and float(row["rating"]) >= threshold
                ):
                    counts[user_id] += 1
        result[split] = counts
    return result


def activity_bin(rating_count: int) -> str:
    """Map a retained training-rating count to a sampling stratum label.

    rating_count is the number of train ratings inside the selected catalog.
    Return '20-39' below 40, '40-79' below 80, and '80+' otherwise. The first
    label reflects the main experiment's minimum of 20; smaller execution samples
    may include users below 20 under the same label.
    """

    if rating_count < 40:
        return "20-39"
    if rating_count < 80:
        return "40-79"
    return "80+"


def stratified_user_sample(
    eligible_counts: dict[int, int], sample_size: int, seed: int
) -> tuple[list[int], dict[str, dict[str, int]]]:
    """Sample users in proportion to their training-activity strata.

    Parameters
    ----------
    eligible_counts : dict[int, int]
        Original eligible user IDs and training-rating counts.
    sample_size : int
        Requested number of distinct users; must not exceed the eligible count.
    seed : int
        Seed for NumPy's deterministic random generator.

    Returns
    -------
    tuple[list[int], dict]
        Sorted chosen IDs and per-stratum eligible/sample counts. Floors of the
        proportional allocations are filled by largest fractional remainders so
        the total is exactly sample_size. Each stratum uses sampling without
        replacement through Generator.choice(..., replace=False).
    """

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
    """Write one split restricted to selected users and games.

    source_path is a processed split CSV and output_path is its destination.
    user_mapping and item_mapping map original integer IDs to compact IDs shared
    by every split. Return the number of rows written, preserving rating values.
    """

    written = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        reader = csv.DictReader(source)
        writer = csv.writer(output, lineterminator="\n")
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
    """Write metadata for the selected games and retain their source IDs.

    source_path is games.csv, output_path is its destination, and item_mapping
    maps source item IDs to compact experiment IDs. Return the number of rows
    written. source_item_id records the preceding dataset's ID for traceability.
    """

    written = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No header in {source_path}")
        fields = ["item_id", "source_item_id", *[f for f in reader.fieldnames if f != "item_id"]]
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
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
    """Write user-ID provenance and training activity for the selected sample.

    output_path is users.csv. selected_users lists source IDs, user_mapping maps
    those IDs to compact IDs, and train_counts gives their retained training
    rating counts. The function returns None after writing the mapping/activity CSV.
    """

    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ("user_id", "source_user_id", "train_rating_count", "activity_bin")
        )
        for old_user in selected_users:
            count = train_counts[old_user]
            writer.writerow((user_mapping[old_user], old_user, count, activity_bin(count)))


def write_eval_users(
    output_path: Path,
    selected_users: list[int],
    user_mapping: dict[int, int],
    relevant_counts: dict[str, Counter[int]] | None,
) -> None:
    """Write positive counts for the shared ranking-evaluation cohort.

    output_path is eval_users.csv; selected_users and user_mapping identify the
    selected source IDs and their compact replacements. relevant_counts maps
    split names to source-user positive-count Counters. Return None and write
    nothing if those counts are unavailable; otherwise write all three counts.
    """

    if relevant_counts is None:
        return
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "user_id",
                "train_relevant_count",
                "validation_relevant_count",
                "test_relevant_count",
            )
        )
        for old_user in selected_users:
            writer.writerow(
                (
                    user_mapping[old_user],
                    relevant_counts["train"][old_user],
                    relevant_counts["validation"][old_user],
                    relevant_counts["test"][old_user],
                )
            )


def create_experiment_subset(
    processed_dir: Path,
    output_dir: Path,
    *,
    max_items: int = 1_500,
    max_users: int = 10_000,
    min_user_train_ratings: int = 20,
    relevant_threshold: float = 8.0,
    seed: int = 42,
) -> dict[str, object]:
    """Create a smaller catalog, user sample, aligned splits and selection report.

    Parameters
    ----------
    processed_dir : Path
        Input splits and games.csv, optionally including eval_users.csv.
    output_dir : Path
        Destination for filtered CSVs, ID mappings and subset_report.json.
    max_items : int
        Number of games selected by descending training-rating count.
    max_users : int
        Exact number of eligible users to sample across training-activity strata.
    min_user_train_ratings : int
        Minimum train history inside the selected catalog, 20 in the main run.
    relevant_threshold : float
        Inclusive positive-rating cutoff, used only for evaluation eligibility.
    seed : int
        Seed controlling the user sample, 42 in the reported experiment.

    Returns
    -------
    dict[str, object]
        Counts, sampling rules, strata and sparsity also written as JSON.

    Game order and activity counts use train only. When eval_users.csv exists,
    users must also have a positive rating in each reduced split. This eligibility
    condition inspects validation/test labels and conditions the evaluated cohort;
    it does not fit model scores or tune parameters on test ratings. The stored
    metrics therefore describe this retained warm-start cohort.
    """

    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"
    games_path = processed_dir / "games.csv"
    for path in (train_path, validation_path, test_path, games_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # Step 1: rank games using train counts only and keep the requested catalog.
    item_counts = count_train_items(train_path)
    ranked_items = sorted(item_counts, key=lambda item: (-item_counts[item], item))
    selected_items = ranked_items[:max_items]
    if len(selected_items) < max_items:
        raise ValueError(f"Only {len(selected_items)} training items are available")

    # Step 2: find sufficiently active users inside that selected catalog.
    train_user_counts = count_train_users(train_path, set(selected_items))
    eligible = {
        user_id: count
        for user_id, count in train_user_counts.items()
        if count >= min_user_train_ratings
    }
    # Ranking metrics require a held-out positive. This cohort filter reads
    # labels in each split; model fitting and hyperparameter selection do not
    # receive test ratings. Metrics apply to the retained warm-start users.
    source_eval_users = load_eval_users(processed_dir / "eval_users.csv")
    relevant_counts: dict[str, Counter[int]] | None = None
    if source_eval_users is not None:
        candidates = set(eligible).intersection(source_eval_users)
        relevant_counts = count_relevant_by_split(
            processed_dir, set(selected_items), candidates, relevant_threshold
        )
        eligible = {
            user_id: count
            for user_id, count in eligible.items()
            if user_id in candidates
            and all(relevant_counts[split][user_id] > 0 for split in relevant_counts)
        }
    # Step 4: preserve light, medium, and heavy user proportions while sampling.
    selected_users, strata = stratified_user_sample(eligible, max_users, seed)

    # Step 5: remap IDs and write filtered splits plus metadata. Item IDs follow
    # training popularity rank; user IDs follow source ID order.
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
    write_eval_users(
        output_dir / "eval_users.csv", selected_users, user_mapping, relevant_counts
    )

    # Step 6: record the exact cohort, split sizes, and matrix sparsity.
    report: dict[str, object] = {
        "selection_source": "training catalog/activity counts; per-split positive eligibility when eval_users exists",
        "source_processed_dir": str(processed_dir),
        "users": len(selected_users),
        "items": len(selected_items),
        "min_user_train_ratings": min_user_train_ratings,
        "relevant_threshold": relevant_threshold,
        "evaluation_users": len(selected_users) if relevant_counts is not None else None,
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
    """Build a subset from command-line paths, sizes and eligibility settings.

    The no-argument command parses --processed-dir and --output-dir, --max-items
    and --max-users, --min-user-train-ratings, --relevant-threshold, and --seed.
    These names map directly to create_experiment_subset parameters. Print its
    JSON report and return None; the CLI defaults select 2,000 games/20,000 users.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/experiment")
    )
    parser.add_argument("--max-items", type=int, default=2_000)
    parser.add_argument("--max-users", type=int, default=20_000)
    parser.add_argument("--min-user-train-ratings", type=int, default=20)
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = create_experiment_subset(
        args.processed_dir,
        args.output_dir,
        max_items=args.max_items,
        max_users=args.max_users,
        min_user_train_ratings=args.min_user_train_ratings,
        relevant_threshold=args.relevant_threshold,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
