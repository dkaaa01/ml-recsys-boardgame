"""Make top10_test.csv for ALL models (item_cf, mf, content, hybrid).

Run from the repo root:
    python make_top10.py --data-dir data/sample

- Each model uses its own frozen settings from reports/<model>/best_*.json.
- Hybrid weight w is chosen on validation NDCG@10, then used once for the test users.
- Top-10 excludes games the user rated in train. No player/time condition here.
Outputs (in --out-dir): top10_test.csv (model,user_id,item_id,rank), scores.csv (hybrid)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from hybrid import hybrid_scores, top_k, tune_w
from hybrid_pipeline import build_user_data, final_top10, itemcf_scores_with_evidence
from src.data.loader import load_interactions
from src.metrics import ndcg_at_k
from src.models.content_based import (
    ContentBasedRecommender, ContentConfig, build_item_features)
from src.models.item_cf import ItemCFConfig, ItemItemCF
from src.experiments.matrix_factorization_evaluation import _top_items as mf_team_top_items
from src.models.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig

LIKED = 8.0


def load_config(cls, path: Path):
    """Build a config dataclass from a best_*.json, ignoring metric columns."""
    if not path.exists():
        print(f"[warn] {path} not found -> using {cls.__name__} defaults")
        return cls()
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in raw.items():
        if key in names:
            kind = names[key].type
            kind = kind if isinstance(kind, str) else getattr(kind, "__name__", "")
            if "int" in kind and "float" not in kind:
                value = int(float(value))
            elif "float" in kind:
                value = float(value)
            kwargs[key] = value
    return cls(**kwargs)


def count_rows(path: Path, column: str) -> int:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return max(int(r[column]) for r in csv.DictReader(f)) + 1


def relevant_items(matrix, users):
    out = {}
    for u in users:
        row = matrix.getrow(u)
        liked = row.indices[row.data >= LIKED]
        if liked.size:
            out[u] = set(int(i) for i in liked)
    return out


def read_eval_users(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        col = "user_id" if reader.fieldnames and "user_id" in reader.fieldnames else None
        if col is None:
            return None
        return {int(r[col]) for r in reader}


def single_model_rows(name, score_fn, users, seen_fn, k=10):
    rows = []
    for u in users:
        scores = np.asarray(score_fn(u), dtype=float)
        cand = np.arange(scores.size)
        for rank, (g, _) in enumerate(top_k(scores, cand, seen_fn(u), k), start=1):
            rows.append((name, u, g, rank))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--max-users", type=int, default=20000)
    args = ap.parse_args()
    d, rep = args.data_dir, args.reports_dir

    shape = (count_rows(d / "users.csv", "user_id"), count_rows(d / "games.csv", "item_id"))
    train = load_interactions(d / "train.csv", shape=shape).ratings
    valid = load_interactions(d / "validation.csv", shape=shape).ratings
    test = load_interactions(d / "test.csv", shape=shape).ratings
    print(f"users={shape[0]} games={shape[1]} train ratings={train.nnz}")

    cf = ItemItemCF(load_config(ItemCFConfig, rep / "item_cf/best_item_cf.json")).fit(train)
    cconf = load_config(ContentConfig, rep / "content_based/best_content.json")
    feats = build_item_features(d / "games.csv", cconf, expected_items=shape[1])
    content = ContentBasedRecommender(cconf).fit(train, feats.matrix)
    mf = MatrixFactorization(
        load_config(MatrixFactorizationConfig, rep / "matrix_factorization/best_mf.json")).fit(train)

    # 1) choose w on validation
    val_rel = relevant_items(valid, range(shape[0]))
    val_users = sorted(val_rel)[: args.max_users]
    best_w, results = tune_w(build_user_data(cf, content, val_users), val_rel, ndcg_at_k)
    print("validation NDCG@10 by w:", {w: round(v, 4) for w, v in results.items()}, "-> w =", best_w)

    # 2) test users: the team's eval_users.csv if present, else users with liked test items
    test_rel = relevant_items(test, range(shape[0]))
    eval_users = read_eval_users(d / "eval_users.csv")
    pool = sorted(eval_users & set(range(shape[0]))) if eval_users else sorted(test_rel)
    users = pool[: args.max_users]
    print(f"test users: {len(users)} ({'eval_users.csv' if eval_users else 'users with liked test items'})")

    seen = lambda u: set(train.getrow(u).indices.tolist())
    tdata = build_user_data(cf, content, users)
    rows = []
    rows += single_model_rows("item_cf", cf.score_user, users, seen)
    # MF scores are clipped to 1..10, so many games tie at the maximum. Use the team's own
    # ranking (_top_items) so the tie handling matches the team's reported MF test result.
    for u in users:
        for rank, g in enumerate(mf_team_top_items(mf, u, 10), start=1):
            rows.append(("mf", u, int(g), rank))
    rows += single_model_rows("content", lambda u: np.asarray(
        (content.user_profiles_[u] @ content.item_features_.T).todense()).ravel(), users, seen)
    rows += final_top10(tdata, best_w, model_name="hybrid")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "top10_test.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model", "user_id", "item_id", "rank"))
        w.writerows(rows)
    with (args.out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("user_id", "item_id", "score"))
        for u, dd in tdata.items():
            cand = np.asarray(dd["candidates"])
            keep = ~np.isin(cand, list(dd["exclude"]))
            s = hybrid_scores(np.asarray(dd["knn"])[keep], np.asarray(dd["content"])[keep],
                              np.asarray(dd["has_evidence"])[keep], best_w)
            for g, v in zip(cand[keep], s):
                w.writerow((u, int(g), round(float(v), 6)))

    # quick sanity report (test NDCG@10 per model, on users with liked test items)
    by_model = {}
    for m, u, g, _ in rows:
        by_model.setdefault(m, {}).setdefault(u, []).append(g)
    rel = {u: r for u, r in test_rel.items() if u in set(users)}
    for m, rec in by_model.items():
        print(f"  {m:8s} test NDCG@10 = {ndcg_at_k(rec, rel, 10):.4f}")
    print(f"saved {len(rows)} rows -> {args.out_dir}/top10_test.csv")


if __name__ == "__main__":
    main()
