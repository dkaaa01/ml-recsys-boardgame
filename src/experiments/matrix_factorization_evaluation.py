"""Shared validation and ranking evaluation for Matrix Factorization."""

from __future__ import annotations

import numpy as np

from src.models.matrix_factorization import MatrixFactorization
from src.metrics import catalog_coverage, ndcg_at_k, recall_at_k, rmse, validate_recommendations


def find_relevant_items(ratings, threshold: float) -> dict[int, set[int]]:
    """Find held-out games meeting the binary relevance threshold.

    Args:
        ratings: Sparse user-by-item validation or test rating matrix.
        threshold: Inclusive minimum rating for a relevant game.

    Returns:
        A user-to-item-set mapping containing only users with relevant games."""

    relevant: dict[int, set[int]] = {}
    for user_id in np.flatnonzero(ratings.getnnz(axis=1)):
        row = ratings.getrow(int(user_id))
        liked = row.indices[row.data >= threshold]
        if liked.size:
            relevant[int(user_id)] = set(map(int, liked))
    return relevant


def evaluate(
    model: MatrixFactorization,
    held_out,
    relevant: dict[int, set[int]],
    top_k: int,
) -> dict[str, float]:
    """Measure observed-rating RMSE and unseen Top-K ranking performance.

    Args:
        model: Fitted MatrixFactorization with its training matrix retained.
        held_out: Sparse held-out ratings using the training matrix shape and IDs.
        relevant: Held-out relevant-item sets keyed by user.
        top_k: Recommendation-list length, fixed at 10 by the experiment CLI.

    Returns:
        RMSE, mean recall/NDCG, catalog coverage and evaluated counts.

    RMSE uses all observed held-out ratings. Ranking metrics use only users in relevant and exclude their training-rated games."""

    train = model.train_
    if train is None:
        raise RuntimeError("model must be fitted before evaluation")

    held_out_coo = held_out.tocoo()
    predictions = model.predict_pairs(held_out_coo.row, held_out_coo.col)
    recommendations = {
        user_id: _top_items(model, user_id, top_k)
        for user_id in relevant
    }
    seen_items = {
        user_id: set(map(int, train.getrow(user_id).indices))
        for user_id in relevant
    }
    validate_recommendations(recommendations, seen_items, k=top_k)
    return {
        "rmse": rmse(held_out_coo.data, predictions),
        "recall_at_10": recall_at_k(recommendations, relevant, top_k),
        "ndcg_at_10": ndcg_at_k(recommendations, relevant, top_k),
        "coverage": catalog_coverage(recommendations, range(held_out.shape[1]), k=top_k),
        "rating_predictions": float(held_out_coo.nnz),
        "ranking_users": float(len(relevant)),
    }


def _top_items(model: MatrixFactorization, user_id: int, top_k: int) -> list[int]:
    """Select unseen games using the MF evaluator's fixed partial-sort procedure.

    Args:
        model: Fitted matrix-factorization model supplying one score per game.
        user_id: User row to score and to look up training-seen items.
        top_k: Maximum returned games.

    Returns:
        Ranked integer game IDs.

    Training games are masked with negative infinity. argpartition selects the shortlist, then lexsort orders it by score and ID; this exact selection procedure is reused by make_top10.py to retain the reported MF tie behavior."""

    assert model.train_ is not None
    scores = model.score_user(user_id)
    scores[model.train_.getrow(user_id).indices] = -np.inf
    available = np.flatnonzero(np.isfinite(scores))
    if available.size > top_k:
        available = available[np.argpartition(scores[available], -top_k)[-top_k:]]
    order = np.lexsort((available, -scores[available]))
    return list(map(int, available[order]))
