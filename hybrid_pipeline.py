"""Glue: team ItemItemCF + ContentBasedRecommender -> hybrid Top-10.

Does NOT modify the teammates' files. Put this next to hybrid.py.

Assumptions (check against the real code):
- itemcf:  fitted ItemItemCF (src/models/item_cf.py), user_id is a row of its train matrix
- content: fitted ContentBasedRecommender with attributes
           user_profiles_ (users x features) and item_features_ (items x features)
- both use the same zero-based user_id / item_id
"""
from __future__ import annotations

import csv
import numpy as np
from scipy import sparse

from hybrid import hybrid_scores, top_k, tune_w


def itemcf_scores_with_evidence(itemcf, user_id: int):
    """Same scores as itemcf.score_user, plus a flag: did similar games exist?
    (score_user silently replaces 'no evidence' with popularity; the plan wants 0.)"""
    train, sim = itemcf._require_fit()
    scores = itemcf.score_user(user_id)            # unchanged original logic
    row = train.getrow(user_id)
    if row.nnz == 0:
        return scores, np.zeros(scores.size, dtype=bool)
    observed = row.copy()
    observed.data = np.ones_like(observed.data)
    absolute = sim.copy()
    absolute.data = np.abs(absolute.data)
    denominator = (absolute @ observed.T).toarray().ravel()
    return scores, denominator > 0


def content_scores_all(content, user_id: int) -> np.ndarray:
    """Cosine score for EVERY game (recommend() only returns the Top-10)."""
    profile = content.user_profiles_[user_id]
    out = profile @ content.item_features_.T
    out = out.toarray() if sparse.issparse(out) else np.asarray(out)
    return np.asarray(out, dtype=float).ravel()


def build_user_data(itemcf, content, user_ids):
    train, _ = itemcf._require_fit()
    data = {}
    for u in user_ids:
        knn, has_ev = itemcf_scores_with_evidence(itemcf, u)
        data[u] = {
            "candidates": np.arange(knn.size),
            "knn": knn,
            "content": content_scores_all(content, u),
            "has_evidence": has_ev,
            "exclude": set(train.getrow(u).indices.tolist()),  # rated in train
        }
    return data


def final_top10(data, w, k=10, model_name="hybrid"):
    rows = []
    for u, d in data.items():
        cand = np.asarray(d["candidates"])
        keep = ~np.isin(cand, list(d["exclude"]))
        s = hybrid_scores(np.asarray(d["knn"])[keep], np.asarray(d["content"])[keep],
                          np.asarray(d["has_evidence"])[keep], w)
        for rank, (g, _) in enumerate(top_k(s, cand[keep], (), k), start=1):
            rows.append((model_name, u, g, rank))
    return rows


def save_top10(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model", "user_id", "item_id", "rank"))
        w.writerows(rows)


if __name__ == "__main__":
    # Smoke test with random data and the team's real item_cf.py.
    from item_cf import ItemItemCF, ItemCFConfig
    rng = np.random.default_rng(0)
    R = sparse.random(60, 40, density=0.3, random_state=0, format="csr")
    R.data = rng.integers(1, 11, R.nnz).astype(float)
    cf = ItemItemCF(ItemCFConfig(min_overlap=1, shrinkage=0)).fit(R)

    class FakeContent:  # stands in for ContentBasedRecommender
        item_features_ = sparse.random(40, 20, density=0.4, random_state=1, format="csr")
        user_profiles_ = sparse.random(60, 20, density=0.4, random_state=2, format="csr")

    users = list(range(10))
    data = build_user_data(cf, FakeContent, users)
    relevant = {u: set(rng.choice(40, 3, replace=False).tolist()) for u in users}
    fake_ndcg = lambda rec, rel, k: float(np.mean(
        [len(set(rec[u]) & rel[u]) / k for u in rec]))
    best, results = tune_w(data, relevant, fake_ndcg)
    print("w results:", results, "best:", best)
    rows = final_top10(data, best)
    assert all(g not in data[u]["exclude"] for _, u, g, _ in rows)
    save_top10(rows, "top10_demo.csv")
    print(len(rows), "rows ok")
