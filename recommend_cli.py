"""Terminal board-game recommender (skeleton).

Flow (team plan, page 1):
  1. profile name (Enter = guest)
  2. number of players and maximum play time
  3. Top-K list: personalized (from a score file) or guest popularity

Input files (columns follow the repo's preprocess.py output)
  games.csv   item_id, name, min_players, max_players, min_playtime, max_playtime
  users.csv   user_id, username
  train.csv   user_id, item_id, rating
  scores.csv  user_id, item_id, score      <- hybrid scores (optional for now)

Try it without any real data:
  python recommend_cli.py --make-sample sample_data
  python recommend_cli.py --data-dir sample_data
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_games(path: Path) -> dict[int, dict]:
    games = {}
    for row in read_csv(path):
        games[int(row["item_id"])] = {
            "name": row.get("name", ""),
            "min_p": to_int(row.get("min_players")),
            "max_p": to_int(row.get("max_players")),
            "min_t": to_int(row.get("min_playtime")),
            "max_t": to_int(row.get("max_playtime")),
            # team column: 1 = usable game (absent in older files -> use our own check)
            "eligible": row.get("condition_eligible", "1").strip() not in ("0", "False", "false"),
        }
    return games


def is_valid(game: dict) -> bool:
    """Plan rule: drop games with 0 time or an impossible player/time range."""
    return (
        game["min_p"] >= 1
        and game["max_p"] >= game["min_p"]
        and game["max_t"] > 0
        and game["min_t"] <= game["max_t"]
    )


def fits(game: dict, players: int, minutes: int) -> bool:
    # Assumption: "recorded play time" = max_playtime, and it must fit the limit.
    return game["eligible"] and is_valid(game) and game["min_p"] <= players <= game["max_p"] and game["max_t"] <= minutes


def load_users(path: Path) -> dict[str, int]:
    """Profile key -> user_id. Uses `username` if the file has it, else the user_id number."""
    users = {}
    for row in read_csv(path):
        uid = int(row["user_id"])
        users[str(uid)] = uid
        if row.get("username"):
            users[row["username"].strip().lower()] = uid
    return users


def load_train(path: Path) -> tuple[dict[int, set[int]], Counter]:
    seen: dict[int, set[int]] = defaultdict(set)
    counts: Counter = Counter()  # popularity = number of training ratings
    for row in read_csv(path):
        seen[int(row["user_id"])].add(int(row["item_id"]))
        counts[int(row["item_id"])] += 1
    return seen, counts


def load_scores(path: Path) -> dict[int, dict[int, float]]:
    scores: dict[int, dict[int, float]] = defaultdict(dict)
    if path.exists():
        for row in read_csv(path):
            scores[int(row["user_id"])][int(row["item_id"])] = float(row["score"])
    return scores


def ask_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("1 이상의 숫자를 입력하세요.")


def recommend(games, candidates, user_scores, counts, k):
    """Rank candidates. Personalized if user_scores is given, else popularity."""
    if user_scores is None:
        key = lambda i: (-counts[i], i)
    else:
        # Games without a score go after scored ones, ordered by popularity.
        key = lambda i: (i not in user_scores, -user_scores.get(i, 0.0), -counts[i], i)
    return sorted(candidates, key=key)[:k]


def make_sample(folder: Path) -> None:
    """Write a tiny dataset in the same format as the real files."""
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--scores", type=Path, default=None, help="default: <data-dir>/scores.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--make-sample", type=Path, default=None, help="create tiny sample files and exit")
    args = parser.parse_args()

    if args.make_sample:
        make_sample(args.make_sample)
        print(f"샘플 데이터를 {args.make_sample}에 만들었어요.")
        return

    games = load_games(args.data_dir / "games.csv")
    users = load_users(args.data_dir / "users.csv")
    seen, counts = load_train(args.data_dir / "train.csv")
    scores = load_scores(args.scores or args.data_dir / "scores.csv")

    name = input("프로필(사용자 번호 또는 이름)을 입력하세요. 게스트는 Enter를 누르세요. > ").strip()
    user_id, mode = None, "게스트 인기순"
    if name:
        uid = users.get(name.lower())
        if uid is None or not seen.get(uid):
            print(f"'{name}' 프로필은 학습 이력이 없어 게스트 추천으로 안내합니다.")
        elif uid not in scores:
            print(f"'{name}' 프로필의 개인화 점수가 아직 없어 게스트 추천으로 안내합니다.")
        else:
            user_id, mode = uid, "개인화 추천"

    players = ask_int("인원수를 입력하세요. > ")
    minutes = ask_int("최대 플레이 시간(분)을 입력하세요. > ")

    candidates = [i for i, g in games.items() if fits(g, players, minutes)]
    if user_id is not None:
        candidates = [i for i in candidates if i not in seen[user_id]]  # already rated
    ranked = recommend(games, candidates, scores[user_id] if user_id is not None else None, counts, args.top_k)

    print(f"\n[{mode}] {players}명 · {minutes}분 이내")
    if not ranked:
        print("조건에 맞는 게임이 없어요.")
        return
    for rank, item in enumerate(ranked, start=1):
        g = games[item]
        print(f"{rank:>2}. {g['name']}  | {g['min_p']}~{g['max_p']}명 | {g['max_t']}분")


if __name__ == "__main__":
    main()
