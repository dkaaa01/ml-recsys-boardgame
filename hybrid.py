"""Weighted hybrid (ItemCF + content) skeleton following the team plan.

One user at a time. All arrays are aligned to the same candidate games.

Plan rules implemented here
- Normalize each model's scores to 0-1 per user, over the same candidates.
- Games with no ItemCF evidence get an ItemCF score of 0 when combined.
- hybrid = w * knn + (1 - w) * content, with w chosen on validation.
- Games the user already rated in train are excluded from the Top-K.
"""

from __future__ import annotations

import numpy as np


def minmax_01(scores, mask=None) -> np.ndarray:
    """Scale scores to 0-1. If `mask` is given, only True entries are scaled
    and the rest stay 0. If max == min, everything is 0 (avoids dividing by 0).
    """
    scores = np.asarray(scores, dtype=float)
    out = np.zeros_like(scores)
    use = np.ones(scores.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not use.any():
        return out
    low, high = scores[use].min(), scores[use].max()
    if high > low:
        out[use] = (scores[use] - low) / (high - low)
    return out


def hybrid_scores(knn, content, has_evidence, w: float) -> np.ndarray:
    """w is the collaborative (ItemCF) weight: w * knn + (1 - w) * content."""
    knn_n = minmax_01(knn, mask=has_evidence)  # no evidence -> stays 0
    content_n = minmax_01(content)
    return w * knn_n + (1.0 - w) * content_n


def top_k(scores, candidates, exclude=(), k: int = 10) -> list[tuple[int, float]]:
    """Return [(game_id, score), ...]. Ties are broken by smaller game id."""
    candidates = np.asarray(candidates)
    scores = np.asarray(scores, dtype=float).copy()
    if len(exclude):
        scores[np.isin(candidates, list(exclude))] = -np.inf
    order = np.lexsort((candidates, -scores))
    order = order[np.isfinite(scores[order])][:k]
    return [(int(candidates[i]), float(scores[i])) for i in order]


def tune_w(users: dict, relevant: dict, ndcg_fn, weights=(0.25, 0.50, 0.75), k: int = 10):
    """Pick the weight with the best validation NDCG@10.

    users:    {user_id: {"candidates", "knn", "content", "has_evidence", "exclude"}}
    relevant: {user_id: set(liked game ids)}
    ndcg_fn:  ndcg_fn(recommendations, relevant, k) -> float
              (same call shape as the repo's metrics.ndcg_at_k usage)
    """
    results = {}
    for w in weights:
        recommendations = {}
        for user_id, data in users.items():
            if user_id not in relevant:
                continue
            # Drop already-rated games first, then normalize (plan order).
            all_candidates = np.asarray(data["candidates"])
            keep = ~np.isin(all_candidates, list(data["exclude"]))
            scores = hybrid_scores(
                np.asarray(data["knn"])[keep],
                np.asarray(data["content"])[keep],
                np.asarray(data["has_evidence"])[keep],
                w,
            )
            ranked = top_k(scores, all_candidates[keep], (), k)
            recommendations[user_id] = [game_id for game_id, _ in ranked]
        results[w] = ndcg_fn(recommendations, relevant, k)
    best = max(results, key=results.get)
    return best, results


if __name__ == "__main__":
    # The A/B/C example from the walkthrough (w = 0.75).
    candidates = [1, 2, 3]                  # A, B, C
    knn = [9.0, 5.0, 7.5]                   # C's 7.5 is a popularity fallback
    content = [0.4, 0.6, 0.5]
    has_evidence = [True, True, False]      # C has no similar-game evidence

    scores = hybrid_scores(knn, content, has_evidence, w=0.75)
    ranked = top_k(scores, candidates)
    print("scores:", dict(zip(candidates, np.round(scores, 3))))
    print("top:", ranked)

    assert np.allclose(scores, [0.75, 0.25, 0.125])
    assert [game for game, _ in ranked] == [1, 2, 3]

    # Excluding an already-rated game removes it from the list.
    assert [g for g, _ in top_k(scores, candidates, exclude=[1])] == [2, 3]
    # All-equal scores do not divide by zero.
    assert np.allclose(minmax_01([3, 3, 3]), [0, 0, 0])
    print("ok")
