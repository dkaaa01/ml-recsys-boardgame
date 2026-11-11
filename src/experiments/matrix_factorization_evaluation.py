"""Shared validation and ranking evaluation for Matrix Factorization."""

from __future__ import annotations

import numpy as np

from src.models.matrix_factorization import MatrixFactorization
from src.metrics import catalog_coverage, ndcg_at_k, recall_at_k, rmse, validate_recommendations


def find_relevant_items(ratings, threshold: float) -> dict[int, set[int]]:
    """Return each user's held-out items whose rating meets the threshold."""

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
    """Measure rating error, Top-K quality, and catalog coverage."""

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
    """Return deterministic Top-K unseen items for one user."""

    assert model.train_ is not None
    scores = model.score_user(user_id)
    scores[model.train_.getrow(user_id).indices] = -np.inf
    available = np.flatnonzero(np.isfinite(scores))
    if available.size > top_k:
        available = available[np.argpartition(scores[available], -top_k)[-top_k:]]
    order = np.lexsort((available, -scores[available]))
    return list(map(int, available[order]))
