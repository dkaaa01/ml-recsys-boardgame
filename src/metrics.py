"""Evaluation metrics shared by recommendation models.

The functions in this module intentionally use only the Python standard library.
This keeps the metric definitions easy to inspect and makes them usable by every
model without converting results to a framework-specific data type.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import TypeVar


UserId = TypeVar("UserId", bound=Hashable)
ItemId = TypeVar("ItemId", bound=Hashable)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Calculate root mean squared error over paired rating observations.

    Args:
        actual: Iterable of observed numeric ratings.
        predicted: Equally long iterable of numeric rating estimates.
    Returns:
        sqrt(mean((actual - predicted)**2)) on the original rating scale.
    Raises:
        ValueError: If lengths differ or there are no rating pairs."""
    actual_values = list(actual)
    predicted_values = list(predicted)
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    if not actual_values:
        raise ValueError("RMSE requires at least one prediction")
    squared_error = sum(
        (float(observed) - float(estimate)) ** 2
        for observed, estimate in zip(actual_values, predicted_values)
    )
    return math.sqrt(squared_error / len(actual_values))


def validate_recommendations(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    seen_items: Mapping[UserId, Collection[ItemId]],
    *,
    k: int | None = None,
) -> None:
    """Reject repeated games, train-seen games, or oversized recommendation lists.

    Args:
        recommendations: User ID to ordered item IDs.
        seen_items: User ID to items observed during training.
        k: Optional positive maximum list length; shorter lists are accepted.
    Returns:
        None when every list satisfies the protocol.
    Raises:
        ValueError: For a nonpositive k or any invalid recommendation list."""
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    for user_id, items in recommendations.items():
        if k is not None and len(items) > k:
            raise ValueError(f"user {user_id!r} has more than {k} recommendations")
        if len(set(items)) != len(items):
            raise ValueError(f"user {user_id!r} has duplicate recommendations")
        overlap = set(items).intersection(seen_items.get(user_id, ()))
        if overlap:
            raise ValueError(
                f"user {user_id!r} was recommended seen items: {sorted(overlap, key=str)!r}"
            )


def recall_at_k(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    """Calculate user-averaged Recall@K over users with known relevant games.

    Args:
        recommendations: User ID to ordered recommended game IDs.
        relevant_items: User ID to held-out relevant game IDs; the experiments
            define relevance as test/validation rating at least 8.
        k: Positive recommendation cutoff.
    Returns:
        Mean of |top_K intersect relevant| / |relevant|, in [0, 1]. Users
        with no relevant games are skipped; a missing ranking has no hits.
    Raises:
        ValueError: If k is nonpositive or no eligible users remain."""
    _require_positive_k(k)
    scores = []
    for user_id, relevant in relevant_items.items():
        relevant_set = set(relevant)
        if not relevant_set:
            continue
        recommended = set(recommendations.get(user_id, ())[:k])
        scores.append(len(recommended.intersection(relevant_set)) / len(relevant_set))
    return _mean_user_score(scores, "Recall@K")


def ndcg_at_k(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    """Calculate user-averaged binary NDCG@K with logarithmic rank discounts.

    For a one-based rank r, a relevant hit contributes 1/log2(r+1).
    Divide DCG by the ideal DCG from min(k, relevant_count) hits at the top.
    The binary gain follows the project's fixed rating-at-least-8 definition;
    ratings 8, 9 and 10 receive equal relevance gains.

    Args:
        recommendations: User ID to ordered game IDs without duplicates.
        relevant_items: User ID to held-out relevant game IDs.
        k: Positive recommendation cutoff.
    Returns:
        Mean normalized ranking score in [0, 1] for users with relevant items.
        Users with no relevant games are skipped; missing rankings score zero.
    Raises:
        ValueError: If k is nonpositive or no eligible users remain."""
    _require_positive_k(k)
    scores = []
    for user_id, relevant in relevant_items.items():
        relevant_set = set(relevant)
        if not relevant_set:
            continue
        ranking = recommendations.get(user_id, ())[:k]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item_id in enumerate(ranking)
            if item_id in relevant_set
        )
        ideal_hits = min(k, len(relevant_set))
        ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        scores.append(dcg / ideal_dcg)
    return _mean_user_score(scores, "NDCG@K")


def catalog_coverage(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    catalog_items: Collection[ItemId],
    *,
    k: int | None = None,
) -> float:
    """Calculate the catalog fraction shown at least once across all users.

    Args:
        recommendations: User ID to ordered recommended game IDs.
        catalog_items: Complete nonempty candidate catalog.
        k: Optional positive cutoff applied separately to each user's list.
    Returns:
        Number of distinct exposed games / number of catalog games, in [0, 1].
        This is exposure breadth across users, not within-list dissimilarity.
    Raises:
        ValueError: For an empty catalog, nonpositive k, or an unknown game ID."""
    catalog = set(catalog_items)
    if not catalog:
        raise ValueError("catalog coverage requires a non-empty catalog")
    if k is not None:
        _require_positive_k(k)
    exposed: set[ItemId] = set()
    for items in recommendations.values():
        exposed.update(items if k is None else items[:k])
    unknown = exposed.difference(catalog)
    if unknown:
        raise ValueError(f"recommendations contain items outside the catalog: {unknown!r}")
    return len(exposed) / len(catalog)


def _require_positive_k(k: int) -> None:
    """Require a positive recommendation cutoff.

    Args:
        k: Cutoff to validate.
    Returns:
        None, or raises ValueError if k <= 0."""

    if k <= 0:
        raise ValueError("k must be positive")


def _mean_user_score(scores: list[float], metric_name: str) -> float:
    """Average nonempty eligible-user metric values.

    Args:
        scores: Per-user metric values after skipping users without relevant items.
        metric_name: Label used in the error message if there are no values.
    Returns:
        Arithmetic mean; each eligible user receives equal weight.
    Raises:
        ValueError: If scores is empty."""

    if not scores:
        raise ValueError(f"{metric_name} requires at least one user with relevant items")
    return sum(scores) / len(scores)
