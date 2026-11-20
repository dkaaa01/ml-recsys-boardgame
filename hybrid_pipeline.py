"""Connect fitted ItemCF and content models to weighted hybrid rankings.

Both models must use the same compact user and game IDs. Helpers produce the
aligned score arrays consumed by hybrid.py; make_top10.py is the executable
entry point for validation-weight selection and final ranking export.
"""
from __future__ import annotations

import csv
import numpy as np
from scipy import sparse

from hybrid import hybrid_scores, top_k


def itemcf_scores_with_evidence(itemcf, user_id: int):
    """Return ItemCF scores and identify games supported by similar rated games.

    Args:
        itemcf: Fitted ItemItemCF sharing its item ordering with the content model.
        user_id: Zero-based training-matrix row to score.

    Returns:
        Two item-length arrays: raw ItemCF scores and a Boolean evidence mask.

    The mask is based on a nonzero absolute-similarity denominator. Hybrid normalization assigns zero CF contribution to games with no such evidence, rather than treating the popularity fallback as collaborative evidence."""
    train, sim = itemcf._require_fit()
    scores = itemcf.score_user(user_id)
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
    """Compute a fitted content profile's cosine scores for all games.

    Args:
        content: Fitted content model with normalized user_profiles_ and item_features_.
        user_id: Zero-based row in user_profiles_.

    Returns:
        A one-dimensional float array in item-matrix column order."""
    profile = content.user_profiles_[user_id]
    out = profile @ content.item_features_.T
    out = out.toarray() if sparse.issparse(out) else np.asarray(out)
    return np.asarray(out, dtype=float).ravel()


def build_user_data(itemcf, content, user_ids):
    """Align CF scores, content scores and exclusions for each user.

    Args:
        itemcf: Fitted ItemItemCF whose training matrix supplies seen-game IDs.
        content: Fitted content model using the same user and item IDs.
        user_ids: Iterable of zero-based users to score.

    Returns:
        A mapping per user with candidates, knn, content, has_evidence and exclude fields."""

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
    """Generate hybrid recommendation rows with the chosen fixed weight.

    Args:
        data: Per-user score/exclusion mapping returned by build_user_data.
        w: Weight assigned to normalized ItemCF scores; 1-w weights content.
        k: Maximum recommendations per user.
        model_name: Label written in the model column.

    Returns:
        Tuples (model, user_id, item_id, rank), using one-based ranks.

    Seen games are removed before normalization so both models share the same candidate set."""

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
    """Write model/user/item/rank tuples to a headered CSV.

    Args:
        rows: Recommendation tuples in the desired output order.
        path: Destination CSV; its parent directory must already exist."""

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model", "user_id", "item_id", "rank"))
        w.writerows(rows)
