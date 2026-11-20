"""Weighted hybrid ranking from aligned ItemCF and content scores.

Scores are min-max normalized separately for each user over the same unseen
candidates. ItemCF scores lacking neighbor evidence contribute zero. The
collaborative weight is selected on validation data before final test scoring.
See docs/code_explanations.md for the library APIs and algorithm references.
"""

from __future__ import annotations

import numpy as np


def minmax_01(scores, mask=None) -> np.ndarray:
    """Normalize eligible scores to [0, 1], with excluded positions left at zero.

    Args:
        scores: One-dimensional score array for a single user's candidates.
        mask: Optional aligned boolean eligibility mask, e.g. ItemCF evidence.
    Returns:
        Float array of the same shape. If no entry is eligible or all eligible
        scores are equal, every output is zero to avoid division by zero."""
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
    """Combine per-user normalized collaborative and content evidence.

    Args:
        knn: ItemCF scores aligned to the current candidate games.
        content: Aligned raw content cosine scores.
        has_evidence: Aligned mask marking ItemCF scores with usable neighbors.
        w: ItemCF weight in [0, 1]; content receives 1 - w.
    Returns:
        Aligned weighted scores w*ItemCF_norm + (1-w)*content_norm. ItemCF
        positions without evidence contribute zero, excluding popularity fallback."""
    knn_n = minmax_01(knn, mask=has_evidence)  # no evidence -> stays 0
    content_n = minmax_01(content)
    return w * knn_n + (1.0 - w) * content_n


def top_k(scores, candidates, exclude=(), k: int = 10) -> list[tuple[int, float]]:
    """Select highest finite candidate scores after optional exclusions.

    Args:
        scores: One-dimensional scores aligned to candidates.
        candidates: Item IDs in the same order as scores.
        exclude: Collection of item IDs already rated or otherwise ineligible.
        k: Positive maximum number of returned games.
    Returns:
        (item_id, score) pairs sorted by descending score, then ascending ID.
        Infinite masked entries are omitted, so fewer than k may be returned."""
    candidates = np.asarray(candidates)
    scores = np.asarray(scores, dtype=float).copy()
    if len(exclude):
        scores[np.isin(candidates, list(exclude))] = -np.inf
    order = np.lexsort((candidates, -scores))
    order = order[np.isfinite(scores[order])][:k]
    return [(int(candidates[i]), float(scores[i])) for i in order]


def tune_w(users: dict, relevant: dict, ndcg_fn, weights=(0.25, 0.50, 0.75), k: int = 10):
    """Select the collaborative weight using validation NDCG@K.

    Args:
        users: Mapping from user ID to aligned candidates, knn, content,
            has_evidence arrays and an exclude collection of train-seen IDs.
        relevant: Validation relevant-item sets keyed by user ID.
        ndcg_fn: Function accepting (recommendations, relevant, k), returning
            the user-averaged validation NDCG. No test labels belong here.
        weights: Candidate ItemCF weights; the first wins an exact score tie.
        k: Positive recommendation cutoff.
    Returns:
        (best_weight, results), where results maps every tried weight to its
        validation NDCG. Seen games are removed before score normalization."""
    results = {}
    for w in weights:
        recommendations = {}
        for user_id, data in users.items():
            if user_id not in relevant:
                continue
            # Remove train-seen games before fitting the per-user score ranges.
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
    # Numerical check for three candidates at w = 0.75.
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
