"""Recommend board games that fit a requested group size and playtime.

Existing profiles use saved hybrid scores and exclude training-rated games.
Guests use the number of training ratings as their popularity ranking. The
CSV inputs are games.csv, users.csv, train.csv and an optional scores.csv.

Run the BGG execution sample from the repository root with:
    python recommend_cli.py --data-dir data/sample --scores outputs/scores.csv

--make-sample writes a separate synthetic fixture for input/filter checks.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a headered UTF-8 CSV, accepting an optional byte-order mark.

    Args:
        path: CSV file to read.

    Returns:
        A list of rows keyed by column name."""

    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def to_int(value, default: int = 0) -> int:
    """Convert a numeric metadata value to an integer.

    Args:
        value: String or numeric value; decimal values are truncated by int().
        default: Fallback for a missing or invalid value.

    Returns:
        The converted integer or default."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_games(path: Path) -> dict[int, dict]:
    """Load game metadata used by the condition filter.

    Args:
        path: games.csv containing item_id, name, player and playtime limits.

    Returns:
        A dictionary from item ID to name, limits, and eligibility flag.

    Missing numeric fields become zero and fail the appropriate validity checks."""

    games = {}
    for row in read_csv(path):
        games[int(row["item_id"])] = {
            "name": row.get("name", ""),
            "min_p": to_int(row.get("min_players")),
            "max_p": to_int(row.get("max_players")),
            "min_t": to_int(row.get("min_playtime")),
            "max_t": to_int(row.get("max_playtime")),
            # Older CSVs lack an eligibility flag; the local validity checks still apply.
            "eligible": row.get("condition_eligible", "1").strip() not in ("0", "False", "false"),
        }
    return games


def is_valid(game: dict) -> bool:
    """Check the player range and the recorded maximum playtime.

    Args:
        game: Metadata dictionary returned by load_games.

    Returns:
        True when the player range is ordered and maximum time is positive."""
    return (
        game["min_p"] >= 1
        and game["max_p"] >= game["min_p"]
        and game["max_t"] > 0
        and game["min_t"] <= game["max_t"]
    )


def fits(game: dict, players: int, minutes: int) -> bool:
    # Use maximum recorded playtime when checking the available minutes.
    """Check whether one game satisfies the requested group and time limit.

    Args:
        game: Metadata dictionary with eligibility, player range and playtime.
        players: Requested positive number of players.
        minutes: Maximum allowed playing time in minutes.

    Returns:
        True if metadata are valid, the group fits, and max_t <= minutes.

    The recorded maximum playtime is used as a conservative time estimate."""

    return game["eligible"] and is_valid(game) and game["min_p"] <= players <= game["max_p"] and game["max_t"] <= minutes


def load_users(path: Path) -> dict[str, int]:
    """Index saved profiles by their number and optional name.

    Args:
        path: users.csv containing user_id and optionally username.

    Returns:
        A mapping from lowercase profile keys to integer user IDs."""
    users = {}
    for row in read_csv(path):
        uid = int(row["user_id"])
        users[str(uid)] = uid
        if row.get("username"):
            users[row["username"].strip().lower()] = uid
    return users


def load_train(path: Path) -> tuple[dict[int, set[int]], Counter]:
    """Collect seen games and game-rating counts from training data.

    Args:
        path: train.csv containing user_id, item_id and rating.

    Returns:
        A pair of user-to-seen-game sets and a game-to-rating-count Counter.

    The guest ranking uses these counts; it differs from the Bayesian baseline."""

    seen: dict[int, set[int]] = defaultdict(set)
    counts: Counter = Counter()  # popularity = number of training ratings
    for row in read_csv(path):
        seen[int(row["user_id"])].add(int(row["item_id"]))
        counts[int(row["item_id"])] += 1
    return seen, counts


def load_scores(path: Path) -> dict[int, dict[int, float]]:
    """Read saved hybrid scores for existing profiles.

    Args:
        path: CSV containing user_id, item_id and score.

    Returns:
        Nested dictionaries scores[user_id][item_id]; empty if the file is absent."""

    scores: dict[int, dict[int, float]] = defaultdict(dict)
    if path.exists():
        for row in read_csv(path):
            scores[int(row["user_id"])][int(row["item_id"])] = float(row["score"])
    return scores


def ask_int(prompt: str) -> int:
    """Prompt until a positive whole number is entered.

    Args:
        prompt: Text shown before reading each attempted input.

    Returns:
        The accepted integer."""

    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("Enter a whole number of at least 1.")


def recommend(games, candidates, user_scores, counts, k):
    """Order eligible game IDs by profile score or training popularity.

    Args:
        games: Game dictionary accepted by the CLI interface; filtering occurs before this call.
        candidates: Game IDs remaining after condition and seen-item filtering.
        user_scores: Hybrid item-score mapping, or None for the guest ranking.
        counts: Training rating counts used for popularity and score ties.
        k: Maximum number of games to return.

    Returns:
        Up to k ranked game IDs.

    For personalized output, games missing a saved score follow scored games; ties use training counts and then item ID."""
    if user_scores is None:
        key = lambda i: (-counts[i], i)
    else:
        # Games without a score go after scored ones, ordered by popularity.
        key = lambda i: (i not in user_scores, -user_scores.get(i, 0.0), -counts[i], i)
    return sorted(candidates, key=key)[:k]


def make_sample(folder: Path) -> None:
    """Write a small synthetic dataset for checking CLI input and filtering.

    Args:
        folder: Destination directory for games.csv, users.csv, train.csv and scores.csv.

    The random values use seed 42. This fixture is separate from the BGG sample and final evaluation data."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    games = [
        # item_id, name, min_p, max_p, min_t, max_t
        (0, "Fast Cards", 2, 4, 15, 20),
        (1, "Space Trade", 2, 5, 60, 90),
        (2, "Cooperative Escape", 2, 4, 30, 45),
        (3, "Party Words", 4, 8, 20, 30),
        (4, "Quick Tiles", 2, 4, 20, 30),
        (5, "Epic Empire", 3, 5, 120, 240),
        (6, "Farm Builder", 1, 5, 45, 60),
        (7, "Duel Chess-like", 2, 2, 20, 30),
        (8, "Broken Data A", 2, 4, 0, 0),      # time 0 -> excluded
        (9, "Broken Data B", 5, 2, 30, 60),    # bad player range -> excluded
        (10, "Deck Fighters", 2, 4, 30, 60),
        (11, "Island Race", 3, 4, 45, 60),
    ]
    with (folder / "games.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(("item_id", "name", "min_players", "max_players", "min_playtime", "max_playtime"))
        w.writerows(games)
    with (folder / "users.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(("user_id", "username"))
        w.writerows([(0, "James"), (1, "NoHistory")])
    with (folder / "train.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(("user_id", "item_id", "rating"))
        for item in (0, 1, 2):
            w.writerow((0, item, 8))              # James already rated games 0-2
        for user in range(2, 12):                 # filler users build popularity
            for item in rng.sample(range(12), 6):
                w.writerow((user, item, rng.randint(5, 10)))
    with (folder / "scores.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(("user_id", "item_id", "score"))
        for item in range(12):
            w.writerow((0, item, round(rng.random(), 3)))  # only James has scores


def main() -> None:
    """Run the profile, player-count and playtime prompts.

    CLI options select --data-dir, --scores and --top-k. The optional --make-sample writes a synthetic fixture and exits. An unknown profile, absent history or absent scores falls back to the guest ranking with a message."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--scores", type=Path, default=None, help="default: <data-dir>/scores.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--make-sample", type=Path, default=None, help="create tiny sample files and exit")
    args = parser.parse_args()

    if args.make_sample:
        make_sample(args.make_sample)
        print(f"Sample files saved to {args.make_sample}.")
        return

    games = load_games(args.data_dir / "games.csv")
    users = load_users(args.data_dir / "users.csv")
    seen, counts = load_train(args.data_dir / "train.csv")
    scores = load_scores(args.scores or args.data_dir / "scores.csv")

    name = input("Profile (user number or name, Enter for guest): ").strip()
    user_id, mode = None, "Guest popularity"
    if name:
        uid = users.get(name.lower())
        if uid is None or not seen.get(uid):
            print(f"No training history for '{name}'. Using guest popularity.")
        elif uid not in scores:
            print(f"No saved scores for '{name}'. Using guest popularity.")
        else:
            user_id, mode = uid, "Personalized recommendation"

    players = ask_int("Players: ")
    minutes = ask_int("Maximum playtime (minutes): ")

    candidates = [i for i, g in games.items() if fits(g, players, minutes)]
    if user_id is not None:
        candidates = [i for i in candidates if i not in seen[user_id]]  # already rated
    ranked = recommend(games, candidates, scores[user_id] if user_id is not None else None, counts, args.top_k)

    print(f"\n[{mode}] {players} players | up to {minutes} minutes")
    if not ranked:
        print("No games match these conditions.")
        return
    for rank, item in enumerate(ranked, start=1):
        g = games[item]
        print(f"{rank:>2}. {g['name']}  | {g['min_p']}-{g['max_p']} players | {g['max_t']} min")


if __name__ == "__main__":
    main()
