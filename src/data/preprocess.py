"""Prepare common evaluation splits from the 2025 BoardGameGeek Reviews data.

The raw review file is larger than memory, so duplicate and conflicting ratings
are resolved in a temporary SQLite database.  Catalog selection uses provisional
training statistics only; the final split then guarantees that eligible ranking
users retain an 8+ rating in train, validation, and test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


USER_ALIASES = ("user", "username")
ITEM_ALIASES = ("id", "bgg_id", "game_id")
RATING_ALIASES = ("rating", "score")

METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("primary", "name"),
    "description": ("description",),
    "year_published": ("yearpublished", "year_published"),
    "min_players": ("minplayers", "min_players"),
    "max_players": ("maxplayers", "max_players"),
    "min_playtime": ("minplaytime", "min_playtime"),
    "max_playtime": ("maxplaytime", "max_playtime", "playingtime"),
    "min_age": ("minage", "min_age"),
    "average_rating": ("average", "average_rating"),
    "bayes_average_rating": ("bayesaverage", "bayes_average_rating"),
    "complexity": ("averageweight", "complexity"),
    "users_rated": ("usersrated", "users_rated"),
    "categories": ("boardgamecategory", "categories"),
    "mechanics": ("boardgamemechanic", "mechanics"),
}


def normalized_header(name: str | None) -> str:
    """Normalize an optional CSV header for alias matching.

    name is a header string or None. Return lowercase text with outer whitespace
    removed and spaces changed to underscores; None becomes an empty string.
    """

    return (name or "").strip().lower().replace(" ", "_")


def resolve_column(fieldnames: Iterable[str | None], aliases: tuple[str, ...]) -> str:
    """Find an original CSV column name by its normalized aliases.

    fieldnames contains actual header values and aliases is an ordered tuple of
    accepted normalized names. Return the first matching original column name;
    raise ValueError when no required alias is present.
    """

    lookup = {normalized_header(field): field for field in fieldnames}
    for alias in aliases:
        if alias in lookup and lookup[alias] is not None:
            return str(lookup[alias])
    raise ValueError(
        f"Missing required column; expected one of {aliases}, got {list(fieldnames)}"
    )


@dataclass(frozen=True)
class Interaction:
    """Hold one user's observation before integer IDs are assigned.

    item is the original game-ID string and rating is the observed 1-10 score.
    The frozen dataclass prevents accidental edits during deterministic splitting.
    """

    item: str
    rating: float


def stable_bucket(user: str, item: str, seed: int, salt: str = "split") -> int:
    """Assign a reproducible bucket to a user/game pair.

    user and item are original string IDs, seed fixes the experiment, and salt
    separates split/positive/repair ordering. Return an integer from 0 through 9
    from an eight-byte BLAKE2b digest. Buckets below 8 form provisional training,
    8 validation and 9 test; this is an approximately 80/10/10 random split,
    with explicit per-user repairs performed later.
    """

    payload = f"{seed}\0{salt}\0{user}\0{item}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % 10


def sha256sum(path: Path) -> str:
    """Return the SHA-256 hexadecimal fingerprint of path.

    path is an existing input or generated CSV. Read four MiB at a time so source
    and output fingerprints can be recorded without loading the files into RAM.
    """

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_source_files(raw_dir: Path) -> tuple[Path, Path]:
    """Locate the rating and metadata CSV pair beneath raw_dir.

    Return (ratings_path, metadata_path), taking the lexicographically last match
    for bgg-*-reviews.csv and games_detailed_info*.csv. Raise FileNotFoundError
    when either pattern has no match. Reproduction uses the version-4 archive.
    """

    ratings = sorted(raw_dir.glob("bgg-*-reviews.csv"))
    metadata = sorted(raw_dir.glob("games_detailed_info*.csv"))
    if not ratings:
        raise FileNotFoundError(f"No bgg-*-reviews.csv found under {raw_dir}")
    if not metadata:
        raise FileNotFoundError(f"No games_detailed_info*.csv found under {raw_dir}")
    return ratings[-1], metadata[-1]


def _numeric_rating(value: str | None) -> float | None:
    """Parse a raw CSV rating and enforce the inclusive 1-10 range.

    value may be missing or nonnumeric. Return a finite float for a valid rating,
    or None for missing, unparseable, nonfinite or out-of-range cells.
    """

    try:
        rating = float(value or "nan")
    except ValueError:
        return None
    return rating if math.isfinite(rating) and 1.0 <= rating <= 10.0 else None


def _connect(database: Path) -> sqlite3.Connection:
    """Open the disposable SQLite cleaning database at database.

    Return a Connection with an observations table keyed by (user, item, rating).
    WITHOUT ROWID avoids a second row-ID index. File-backed temporary storage and
    a 256 MiB page cache limit RAM use. synchronous=OFF speeds this rebuildable
    work database, which is never treated as the permanent source of truth.
    """

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            user TEXT NOT NULL,
            item TEXT NOT NULL,
            rating REAL NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (user, item, rating)
        ) WITHOUT ROWID
        """
    )
    return connection


def ingest_ratings(
    ratings_path: Path,
    database_path: Path,
    *,
    batch_size: int = 50_000,
) -> dict[str, int]:
    """Validate and deduplicate the raw observations in a disk-backed database.

    Parameters
    ----------
    ratings_path : Path
        Raw CSV with user, game and rating columns resolved through aliases.
    database_path : Path
        Disposable SQLite file, replaced when it already exists.
    batch_size : int
        Number of valid rows per executemany transaction, default 50,000.

    Returns
    -------
    dict[str, int]
        Raw/invalid/duplicate/conflicting/clean row counts. The database also
        receives clean_ratings with one observation per surviving user/game pair.

    Bound SQL parameters supply values safely. ON CONFLICT counts repeated equal
    ratings, while GROUP BY/HAVING rejects all observations for a user/game pair
    with multiple different ratings. An exact duplicate alone is retained once.
    """

    if database_path.exists():
        database_path.unlink()
    connection = _connect(database_path)
    statement = """
        INSERT INTO observations(user, item, rating, occurrences)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user, item, rating)
        DO UPDATE SET occurrences = occurrences + 1
    """
    raw_rows = invalid_rows = 0
    batch: list[tuple[str, str, float]] = []
    try:
        with ratings_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError(f"No CSV header in {ratings_path}")
            user_col = resolve_column(reader.fieldnames, USER_ALIASES)
            item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
            rating_col = resolve_column(reader.fieldnames, RATING_ALIASES)
            for row in reader:
                raw_rows += 1
                user = (row.get(user_col) or "").strip()
                item = (row.get(item_col) or "").strip()
                rating = _numeric_rating(row.get(rating_col))
                if not user or not item or rating is None:
                    invalid_rows += 1
                    continue
                batch.append((user, item, rating))
                if len(batch) >= batch_size:
                    connection.executemany(statement, batch)
                    connection.commit()
                    batch.clear()
            if batch:
                connection.executemany(statement, batch)
                connection.commit()

        unique_observations, duplicate_rows = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(occurrences - 1), 0) FROM observations"
        ).fetchone()
        conflicting_pairs, conflicting_input_rows = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(input_rows), 0)
            FROM (
                SELECT user, item, SUM(occurrences) AS input_rows
                FROM observations
                GROUP BY user, item
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        connection.execute(
            """
            CREATE TABLE clean_ratings AS
            SELECT user, item, MIN(rating) AS rating
            FROM observations
            GROUP BY user, item
            HAVING COUNT(*) = 1
            """
        )
        connection.execute("CREATE UNIQUE INDEX clean_user_item ON clean_ratings(user, item)")
        connection.execute("CREATE INDEX clean_item ON clean_ratings(item)")
        connection.commit()
        clean_rows = connection.execute("SELECT COUNT(*) FROM clean_ratings").fetchone()[0]
    finally:
        connection.close()
    return {
        "raw_rows": int(raw_rows),
        "invalid_rows": int(invalid_rows),
        "exact_duplicate_rows_removed": int(duplicate_rows),
        "unique_valid_observations": int(unique_observations),
        "conflicting_user_item_pairs_removed": int(conflicting_pairs),
        "conflicting_input_rows_removed": int(conflicting_input_rows),
        "clean_user_item_pairs": int(clean_rows),
    }


def load_metadata(metadata_path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Resolve game metadata headers and index the retained rows by original ID.

    metadata_path is the raw metadata CSV. Return (metadata, missing_optional),
    where metadata maps original game IDs to normalized field dictionaries and
    missing_optional lists absent fields. Require name, description, player
    limits and maximum playtime columns. Empty IDs and repeated IDs are skipped;
    the first row of each game is retained and absent optional cells become ''.
    """

    metadata: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {metadata_path}")
        item_col = resolve_column(reader.fieldnames, ITEM_ALIASES)
        normalized = {normalized_header(field): field for field in reader.fieldnames}
        columns = {
            output: next((normalized[a] for a in aliases if a in normalized), None)
            for output, aliases in METADATA_ALIASES.items()
        }
        missing = [name for name, column in columns.items() if column is None]
        required = {"name", "description", "min_players", "max_players", "max_playtime"}
        required_missing = sorted(required.intersection(missing))
        if required_missing:
            raise ValueError(
                f"Missing required metadata fields {required_missing}; actual header={reader.fieldnames}"
            )
        for row in reader:
            item = (row.get(item_col) or "").strip()
            if not item or item in metadata:
                continue
            metadata[item] = {
                output: (row.get(column) or "").strip() if column else ""
                for output, column in columns.items()
            }
    return metadata, missing


def _valid_positive_number(value: str) -> float | None:
    """Return a finite positive float parsed from the CSV cell value.

    Return None for an unparseable/nonfinite/non-positive cell, including the
    zero markers used for missing player-count and playtime metadata.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def condition_validity(row: dict[str, str]) -> tuple[bool, str]:
    """Check whether metadata supports player-count and playtime filtering.

    row supplies min_players/max_players/min_playtime/max_playtime strings.
    Return (eligible, reason), with semicolon-separated invalid_player_range or
    invalid_playtime_range reasons. Both interval endpoints must be positive and
    the maximum cannot be smaller than the minimum. Invalid games can remain in
    offline experiments while condition_eligible excludes them from the demo.
    """

    min_players = _valid_positive_number(row["min_players"])
    max_players = _valid_positive_number(row["max_players"])
    min_time = _valid_positive_number(row["min_playtime"])
    max_time = _valid_positive_number(row["max_playtime"])
    reasons = []
    if min_players is None or max_players is None or max_players < min_players:
        reasons.append("invalid_player_range")
    if min_time is None or max_time is None or max_time < min_time:
        reasons.append("invalid_playtime_range")
    return not reasons, ";".join(reasons)


def select_catalog(
    database_path: Path,
    metadata_ids: set[str],
    *,
    min_item_train_ratings: int,
    max_items: int,
    seed: int,
) -> tuple[list[str], Counter[str], dict[str, int]]:
    """Choose a metadata-backed catalog using provisional training counts.

    Parameters
    ----------
    database_path : Path
        SQLite database containing deduplicated clean_ratings.
    metadata_ids : set[str]
        Game IDs with a metadata row available for joining.
    min_item_train_ratings : int
        Minimum provisional train observations per game, 100 in the main run.
    max_items : int
        Catalog cap, 3,000 before the later experiment reduction.
    seed : int
        Seed used by stable_bucket to identify provisional train rows.

    Returns
    -------
    tuple[list[str], Counter[str], dict[str, int]]
        Selected original IDs, all eligible train-count entries and audit counts.
        Games sort by decreasing count, breaking ties by original ID. These counts
        precede the positive-preserving per-user split repair.
    """

    train_counts: Counter[str] = Counter()
    missing_metadata_rows = 0
    connection = sqlite3.connect(database_path)
    try:
        for user, item in connection.execute("SELECT user, item FROM clean_ratings"):
            if item not in metadata_ids:
                missing_metadata_rows += 1
                continue
            if stable_bucket(user, item, seed) < 8:
                train_counts[item] += 1
    finally:
        connection.close()
    eligible = [
        (item, count)
        for item, count in train_counts.items()
        if count >= min_item_train_ratings
    ]
    eligible.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = [item for item, _ in eligible[:max_items]]
    if not selected:
        raise ValueError("No games satisfy the metadata and training-count filters")
    return selected, train_counts, {
        "clean_rows_missing_metadata": missing_metadata_rows,
        "metadata_games_with_training_ratings": len(train_counts),
        "games_with_minimum_training_ratings": len(eligible),
    }


def select_users(
    database_path: Path,
    selected_items: list[str],
    *,
    min_user_train_ratings: int,
    seed: int,
) -> tuple[set[str], Counter[str]]:
    """Choose users with enough provisional train history in the fixed catalog.

    database_path contains clean_ratings and selected_items lists original game
    IDs. min_user_train_ratings is the required provisional count and seed fixes
    the hash split. Return (eligible_user_IDs, train_counts). Temporary SQL tables
    join selected games without reading the complete ratings table into memory.
    """

    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TEMP TABLE selected_items(item TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO selected_items(item) VALUES (?)", ((x,) for x in selected_items))
    train_counts: Counter[str] = Counter()
    try:
        rows = connection.execute(
            """
            SELECT c.user, c.item
            FROM clean_ratings c
            JOIN selected_items s ON s.item = c.item
            """
        )
        for user, item in rows:
            if stable_bucket(user, item, seed) < 8:
                train_counts[user] += 1
    finally:
        connection.close()
    users = {user for user, count in train_counts.items() if count >= min_user_train_ratings}
    return users, train_counts


def _split_user(
    user: str,
    interactions: list[Interaction],
    *,
    seed: int,
    relevant_threshold: float,
    min_train_ratings: int,
) -> dict[str, list[Interaction]]:
    """Create reproducible train/validation/test lists for one user.

    Parameters
    ----------
    user : str
        Original user ID for stable hashing.
    interactions : list[Interaction]
        This user's deduplicated observations in the selected catalog.
    seed : int
        Hash seed for split assignment and deterministic repair ordering.
    relevant_threshold : float
        Inclusive cutoff for a positive observation, 8.0 in the experiment.
    min_train_ratings : int
        Minimum training count sought by moving non-protected holdout rows.

    Returns
    -------
    dict[str, list[Interaction]]
        Item-sorted train/validation/test observations. When at least three
        positives exist, deterministically place one in every split, then repair
        short training histories while protecting a held-out positive per split.

    Positive labels are inspected when constructing this stratified holdout.
    Consequently the split is neither chronological nor a completely independent
    random assignment, and its 80/10/10 proportions can change during repair.
    """

    result = {"train": [], "validation": [], "test": []}
    for interaction in interactions:
        bucket = stable_bucket(user, interaction.item, seed)
        split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        result[split].append(interaction)

    positives = [x for x in interactions if x.rating >= relevant_threshold]
    if len(positives) >= 3:
        positives.sort(key=lambda x: (stable_bucket(user, x.item, seed, "positive"), x.item))
        forced = {
            "validation": positives[0],
            "test": positives[1],
            "train": positives[2],
        }
        for target, interaction in forced.items():
            for source in result:
                if interaction in result[source]:
                    result[source].remove(interaction)
                    break
            result[target].append(interaction)

    if len(result["train"]) < min_train_ratings:
        protected = {
            split: next(
                (x for x in result[split] if x.rating >= relevant_threshold), None
            )
            for split in ("validation", "test")
        }
        movable = [
            (split, interaction)
            for split in ("validation", "test")
            for interaction in result[split]
            if interaction != protected[split]
        ]
        movable.sort(key=lambda pair: (stable_bucket(user, pair[1].item, seed, "repair"), pair[1].item))
        for source, interaction in movable:
            if len(result["train"]) >= min_train_ratings:
                break
            result[source].remove(interaction)
            result["train"].append(interaction)

    for values in result.values():
        values.sort(key=lambda x: x.item)
    return result


def _iter_selected_user_rows(
    database_path: Path,
    selected_items: list[str],
    selected_users: set[str],
) -> Iterator[tuple[str, list[Interaction]]]:
    """Yield selected observations grouped by original user ID.

    database_path contains clean_ratings; selected_items and selected_users are
    the original IDs to retain. Yield (username, list[Interaction]) in user/game
    order. SQL temporary-table joins perform the filtering and only one user's
    interactions are accumulated at a time. The connection closes on completion.
    """

    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TEMP TABLE selected_items(item TEXT PRIMARY KEY)")
    connection.execute("CREATE TEMP TABLE selected_users(user TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO selected_items(item) VALUES (?)", ((x,) for x in selected_items))
    connection.executemany("INSERT INTO selected_users(user) VALUES (?)", ((x,) for x in selected_users))
    cursor = connection.execute(
        """
        SELECT c.user, c.item, c.rating
        FROM clean_ratings c
        JOIN selected_items i ON i.item = c.item
        JOIN selected_users u ON u.user = c.user
        ORDER BY c.user, c.item
        """
    )
    current_user: str | None = None
    interactions: list[Interaction] = []
    try:
        for user, item, rating in cursor:
            if current_user is not None and user != current_user:
                yield current_user, interactions
                interactions = []
            current_user = user
            interactions.append(Interaction(str(item), float(rating)))
        if current_user is not None:
            yield current_user, interactions
    finally:
        connection.close()


def write_outputs(
    database_path: Path,
    output_dir: Path,
    selected_items: list[str],
    selected_users: set[str],
    provisional_train_counts: Counter[str],
    metadata: dict[str, dict[str, str]],
    *,
    seed: int,
    relevant_threshold: float,
    min_user_train_ratings: int,
) -> dict[str, object]:
    """Write aligned cleaned splits, mapping files and evaluation eligibility.

    Parameters
    ----------
    database_path : Path
        SQLite clean_ratings input.
    output_dir : Path
        Directory created for train/validation/test, user/item/game mappings and
        eval_users.csv. Existing files with these names are replaced.
    selected_items : list[str]
        Catalog IDs in fixed order, defining compact item IDs.
    selected_users : set[str]
        Original users satisfying the provisional activity requirement.
    provisional_train_counts : Counter[str]
        Pre-repair game counts recorded in items.csv for traceability.
    metadata : dict[str, dict[str, str]]
        Normalized game fields keyed by original game ID.
    seed : int
        Seed for per-user holdout assignment and repairs.
    relevant_threshold : float
        Positive cutoff used to record eligibility/counts in eval_users.csv.
    min_user_train_ratings : int
        Required final training count; users below it after repair are skipped.

    Returns
    -------
    dict[str, object]
        Written user/item/evaluation counts, split sizes and invalid-condition
        reason counts. Evaluation users have at least one positive in each split;
        all models share the resulting cohort and ID mappings.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    item_ids = {item: item_id for item_id, item in enumerate(selected_items)}
    handles = {
        split: (output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="")
        for split in ("train", "validation", "test")
    }
    users_handle = (output_dir / "users.csv").open("w", encoding="utf-8", newline="")
    eval_handle = (output_dir / "eval_users.csv").open("w", encoding="utf-8", newline="")
    split_counts: Counter[str] = Counter()
    eval_users = 0
    written_users = 0
    try:
        writers = {split: csv.writer(handle) for split, handle in handles.items()}
        for writer in writers.values():
            writer.writerow(("user_id", "item_id", "rating"))
        users_writer = csv.writer(users_handle)
        users_writer.writerow(
            ("user_id", "username", "interaction_count", "train_rating_count")
        )
        eval_writer = csv.writer(eval_handle)
        eval_writer.writerow(
            (
                "user_id",
                "train_relevant_count",
                "validation_relevant_count",
                "test_relevant_count",
            )
        )

        for username, interactions in _iter_selected_user_rows(
            database_path, selected_items, selected_users
        ):
            split = _split_user(
                username,
                interactions,
                seed=seed,
                relevant_threshold=relevant_threshold,
                min_train_ratings=min_user_train_ratings,
            )
            if len(split["train"]) < min_user_train_ratings:
                continue
            user_id = written_users
            written_users += 1
            users_writer.writerow(
                (user_id, username, len(interactions), len(split["train"]))
            )
            relevant_counts = {
                name: sum(x.rating >= relevant_threshold for x in values)
                for name, values in split.items()
            }
            if all(relevant_counts[name] > 0 for name in ("train", "validation", "test")):
                eval_writer.writerow(
                    (
                        user_id,
                        relevant_counts["train"],
                        relevant_counts["validation"],
                        relevant_counts["test"],
                    )
                )
                eval_users += 1
            for name, values in split.items():
                for interaction in values:
                    writers[name].writerow(
                        (user_id, item_ids[interaction.item], f"{interaction.rating:g}")
                    )
                    split_counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()
        users_handle.close()
        eval_handle.close()

    condition_reasons: Counter[str] = Counter()
    game_fields = [
        "item_id",
        "bgg_id",
        *METADATA_ALIASES,
        "condition_eligible",
        "condition_exclusion_reason",
    ]
    with (output_dir / "games.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=game_fields)
        writer.writeheader()
        for item in selected_items:
            valid, reason = condition_validity(metadata[item])
            if reason:
                condition_reasons.update(reason.split(";"))
            writer.writerow(
                {
                    "item_id": item_ids[item],
                    "bgg_id": item,
                    **metadata[item],
                    "condition_eligible": int(valid),
                    "condition_exclusion_reason": reason,
                }
            )

    with (output_dir / "items.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("item_id", "bgg_id", "provisional_train_rating_count"))
        for item in selected_items:
            writer.writerow((item_ids[item], item, provisional_train_counts[item]))

    return {
        "processed_users": written_users,
        "processed_items": len(selected_items),
        "evaluation_users": eval_users,
        "split_counts": dict(split_counts),
        "condition_exclusions": dict(condition_reasons),
    }


def preprocess(
    raw_dir: Path,
    output_dir: Path,
    *,
    work_dir: Path | None = None,
    min_user_train_ratings: int = 10,
    min_item_train_ratings: int = 100,
    max_items: int = 3_000,
    relevant_threshold: float = 8.0,
    seed: int = 42,
    keep_database: bool = False,
) -> dict[str, object]:
    """Run validation, metadata joins, entity selection and common splitting.

    Parameters
    ----------
    raw_dir : Path
        Directory with the extracted rating and game-metadata CSVs.
    output_dir : Path
        Destination for cleaned CSVs and preprocessing_report.json.
    work_dir : Path or None
        Dedicated disposable workspace, default output_dir/.work. The routine
        creates dedup.sqlite3 here and removes this entire directory on success
        unless keep_database is True; use an empty project-owned directory.
    min_user_train_ratings : int
        Provisional/final minimum history per user, default 10.
    min_item_train_ratings : int
        Provisional train support per game, default 100.
    max_items : int
        Cap on selected games, default 3,000.
    relevant_threshold : float
        Inclusive positive cutoff for split repair and evaluation, default 8.0.
    seed : int
        Hash seed, default 42.
    keep_database : bool
        Preserve work_dir for inspecting cleaning tables when True.

    Returns
    -------
    dict[str, object]
        Cleaning counts, settings, split sizes, timings and source/output hashes,
        also written as JSON. This pipeline reads labels for positive-preserving
        split construction; subsequent model fitting uses the emitted train split.
    """

    started = time.perf_counter()

    # Step 1: resolve the two source CSVs and prepare disk-backed work storage.
    ratings_path, metadata_path = locate_source_files(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "dedup.sqlite3"

    # Step 2: validate and deduplicate the large rating file without loading it
    # all into memory, then load the much smaller game metadata table.
    ingest_started = time.perf_counter()
    cleaning = ingest_ratings(ratings_path, database_path)
    ingest_seconds = time.perf_counter() - ingest_started
    metadata, missing_metadata_fields = load_metadata(metadata_path)
    # Step 3: choose the initial catalog and activity-qualified users using
    # provisional training counts. Positive-preserving split construction and
    # the later ranking-evaluation eligibility check also inspect rating labels.
    selected_items, item_train_counts, catalog_stats = select_catalog(
        database_path,
        set(metadata),
        min_item_train_ratings=min_item_train_ratings,
        max_items=max_items,
        seed=seed,
    )
    selected_users, train_user_counts = select_users(
        database_path,
        selected_items,
        min_user_train_ratings=min_user_train_ratings,
        seed=seed,
    )
    # Step 4: create stable IDs, repair per-user splits, and write every model's
    # common train/validation/test files plus the shared evaluation cohort.
    outputs = write_outputs(
        database_path,
        output_dir,
        selected_items,
        selected_users,
        item_train_counts,
        metadata,
        seed=seed,
        relevant_threshold=relevant_threshold,
        min_user_train_ratings=min_user_train_ratings,
    )
    output_files = [
        output_dir / name
        for name in (
            "train.csv",
            "validation.csv",
            "test.csv",
            "users.csv",
            "items.csv",
            "games.csv",
            "eval_users.csv",
        )
    ]
    # Step 5: persist counts, rules, timings, and checksums for reproducibility.
    report: dict[str, object] = {
        "dataset_version": 4,
        "ratings_source": str(ratings_path),
        "metadata_source": str(metadata_path),
        "source_sha256": {
            ratings_path.name: sha256sum(ratings_path),
            metadata_path.name: sha256sum(metadata_path),
        },
        **cleaning,
        **catalog_stats,
        "metadata_rows": len(metadata),
        "missing_optional_metadata_fields": missing_metadata_fields,
        "provisional_catalog_rule": "train-only counts from deterministic 80/10/10 hash",
        "min_user_train_ratings": min_user_train_ratings,
        "min_item_train_ratings": min_item_train_ratings,
        "max_items": max_items,
        "relevant_threshold": relevant_threshold,
        "split_seed": seed,
        **outputs,
        "ingest_seconds": ingest_seconds,
        "total_seconds": time.perf_counter() - started,
        "output_sha256": {path.name: sha256sum(path) for path in output_files},
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Step 6: remove only the reproducible SQLite work directory by default.
    if not keep_database:
        shutil.rmtree(work_dir)
    return report


def main() -> None:
    """Run preprocess from command-line paths and filtering settings.

    This no-argument command parses --raw-dir, --output-dir, --work-dir,
    --min-user-train-ratings, --min-item-train-ratings, --max-items,
    --relevant-threshold, --seed and --keep-database. Each maps to the same-named
    preprocess parameter. Print the JSON audit report and return None.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--min-user-train-ratings", type=int, default=10)
    parser.add_argument("--min-item-train-ratings", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=3_000)
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-database", action="store_true")
    args = parser.parse_args()
    report = preprocess(
        args.raw_dir,
        args.output_dir,
        work_dir=args.work_dir,
        min_user_train_ratings=args.min_user_train_ratings,
        min_item_train_ratings=args.min_item_train_ratings,
        max_items=args.max_items,
        relevant_threshold=args.relevant_threshold,
        seed=args.seed,
        keep_database=args.keep_database,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
