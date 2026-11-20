"""Item-item collaborative filtering for explicit board-game ratings.

The implementation follows the lecture's memory-based CF pipeline:

1. compute similarities;
2. select nearest neighbors;
3. aggregate neighbor ratings;
4. remove games the user has already rated.

The less familiar additions (support shrinkage and baseline-residual scoring)
are explained where they are applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse


SimilarityName = Literal["cosine", "adjusted_cosine", "pearson", "jaccard"]
ScoringName = Literal["weighted_mean", "baseline_residual"]


@dataclass(frozen=True)
class ItemCFConfig:
    """Settings for item-item collaborative filtering.

    Args:
        similarity: cosine, adjusted_cosine, pairwise Pearson or positive Jaccard.
        neighbors: Maximum nonzero neighbors retained for each target game.
        min_overlap: Minimum number of co-raters required for an item pair.
        shrinkage: Nonnegative support penalty lambda in s*n/(n+lambda).
        min_similarity: Similarities below this threshold are discarded.
        positive_threshold: Minimum rating counted as positive by Jaccard.
        scoring: weighted_mean or user-mean-centered baseline_residual.
    The frozen configuration records a trial without being changed by fit()."""

    similarity: SimilarityName = "adjusted_cosine"
    neighbors: int = 40
    min_overlap: int = 5
    shrinkage: float = 25.0
    min_similarity: float = 0.0
    positive_threshold: float = 8.0
    scoring: ScoringName = "baseline_residual"


class ItemItemCF:
    """Fit an item similarity matrix and score unseen games from training ratings.

    Input matrices have users on rows and games on columns; absent entries
    represent unobserved ratings. Learned similarities are stored as a sparse
    item-by-item matrix with at most config.neighbors entries per row.
    The four similarities and the scoring rules are implemented in this file."""

    def __init__(self, config: ItemCFConfig | None = None) -> None:
        """Create an unfitted model with an optional immutable configuration.

        Args:
            config: ItemCFConfig; None uses its declared defaults.
        Returns:
            None. Learned matrices and means are populated by fit()."""

        self.config = config or ItemCFConfig()
        self.train_: sparse.csr_matrix | None = None
        self.similarity_: sparse.csr_matrix | None = None
        self.user_means_: np.ndarray | None = None
        self.item_means_: np.ndarray | None = None
        self.popularity_: np.ndarray | None = None
        self.global_mean_: float | None = None

    @staticmethod
    def _means(matrix: sparse.csr_matrix, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate means from stored observations, excluding absent cells.

        Args:
            matrix: Sparse user-by-item rating matrix.
            axis: 1 for per-user means, 0 for per-item means.
        Returns:
            One-dimensional (means, counts); an empty row or column has mean zero."""

        counts = np.asarray(matrix.getnnz(axis=axis), dtype=np.int64)
        sums = np.asarray(matrix.sum(axis=axis)).ravel()
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
        return means, counts

    @staticmethod
    def _center_rows(matrix: sparse.csr_matrix, means: np.ndarray) -> sparse.csr_matrix:
        """Subtract a user's mean from each of their observed ratings.

        Args:
            matrix: User-by-item CSR matrix with missing entries absent.
            means: One observed-rating mean per user row.
        Returns:
            Float64 CSR copy with centered observations; exact zeros are removed."""

        centered = matrix.astype(np.float64, copy=True)
        centered.data -= np.repeat(means, np.diff(centered.indptr))
        centered.eliminate_zeros()
        return centered

    @staticmethod
    def _center_columns(matrix: sparse.csr_matrix, means: np.ndarray) -> sparse.csr_matrix:
        """Subtract an item mean from each stored rating in that column.

        Args:
            matrix: User-by-item CSR matrix with missing entries absent.
            means: One observed-rating mean per game column.
        Returns:
            Float64 CSR copy with centered observations; exact zeros are removed."""

        centered = matrix.astype(np.float64, copy=True)
        centered.data -= means[centered.indices]
        centered.eliminate_zeros()
        return centered

    @staticmethod
    def _cosine_from_columns(matrix: sparse.csr_matrix) -> np.ndarray:
        """Compute cosine similarity between game columns.

        Args:
            matrix: Sparse user-by-item matrix, optionally already centered.
        Returns:
            Dense item-by-item similarity array. A zero-norm column has zero
            similarity because its denominator is masked before division."""

        gram = (matrix.T @ matrix).toarray()
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        denominator = np.outer(norms, norms)
        return np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)

    def fit(self, ratings: sparse.csr_matrix) -> "ItemItemCF":
        """Calculate item similarities and keep each target game's top neighbors.

        Cosine compares rating vectors; adjusted cosine subtracts user means;
        Pearson centers each item pair over common raters; Jaccard compares
        positive-rater sets. All variants use the same overlap and score filters.

        Args:
            ratings: User-by-item sparse matrix containing observed 1-10 ratings.
        Returns:
            This fitted model. Input ratings are copied, not changed.
        Raises:
            ValueError: If there are no observations, neighbors is nonpositive,
                or the similarity name is unsupported."""

        if not sparse.isspmatrix_csr(ratings):
            ratings = ratings.tocsr()
        if ratings.nnz == 0:
            raise ValueError("Cannot fit ItemItemCF on an empty matrix")
        if self.config.neighbors <= 0:
            raise ValueError("neighbors must be positive")

        # Float32 is sufficiently precise for 1-10 ratings and halves both the
        # sparse data and the temporary dense item-pair similarity arrays.
        self.train_ = ratings.astype(np.float32, copy=True)
        self.train_.sort_indices()
        self.user_means_, user_counts = self._means(self.train_, axis=1)
        self.item_means_, item_counts = self._means(self.train_, axis=0)
        self.global_mean_ = float(self.train_.data.mean())

        # Bayesian popularity is used only when a user has no usable similar
        # item.  The prior prevents a rarely rated item from winning by chance.
        prior = float(np.quantile(item_counts[item_counts > 0], 0.75))
        self.popularity_ = (
            item_counts * self.item_means_ + prior * self.global_mean_
        ) / np.maximum(item_counts + prior, 1.0)

        binary = self.train_.copy()
        binary.data = np.ones_like(binary.data, dtype=np.float32)
        overlaps = (binary.T @ binary).toarray()

        if self.config.similarity == "cosine":
            similarities = self._cosine_from_columns(self.train_)
        elif self.config.similarity == "adjusted_cosine":
            # Adjusted cosine removes each user's personal rating scale before
            # comparing games.  A generous rater and a strict rater can then
            # express the same preference pattern on different numeric scales.
            similarities = self._cosine_from_columns(
                self._center_rows(self.train_, self.user_means_)
            )
        elif self.config.similarity == "pearson":
            # Exact Pearson correlation must center *each item pair* using only
            # their common raters.  The formula below derives pairwise sums and
            # squared sums with sparse matrix multiplication; it avoids a slow
            # Python loop over the catalog's item pairs.
            cross_products = (self.train_.T @ self.train_).toarray()
            common_sums = (self.train_.T @ binary).toarray()
            squared = self.train_.copy()
            squared.data **= 2
            common_squared_sums = (squared.T @ binary).toarray()
            safe_overlap = overlaps.copy()
            safe_overlap[safe_overlap == 0] = 1.0
            # Reuse dense buffers in place. Each float32 item-pair array needs
            # 4 * n_items**2 bytes (16 MB for the 2,000-game experiment).
            # Avoiding duplicate covariance/variance arrays reduces peak memory.
            cross_products -= common_sums * common_sums.T / safe_overlap
            common_squared_sums -= common_sums**2 / safe_overlap
            denominator = np.sqrt(
                np.maximum(common_squared_sums, 0.0)
                * np.maximum(common_squared_sums.T, 0.0)
            )
            similarities = np.divide(
                cross_products,
                denominator,
                out=cross_products,
                where=denominator > 0,
            )
        elif self.config.similarity == "jaccard":
            positive = self.train_.copy()
            positive.data = (positive.data >= self.config.positive_threshold).astype(float)
            positive.eliminate_zeros()
            intersection = (positive.T @ positive).toarray()
            positive_counts = np.asarray(positive.getnnz(axis=0), dtype=np.float32)
            union = positive_counts[:, None] + positive_counts[None, :] - intersection
            similarities = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
        else:  # Defensive guard for runtime configuration files.
            raise ValueError(f"Unknown similarity: {self.config.similarity}")

        # Shrinkage is not part of the basic classroom formula.  It discounts a
        # high similarity supported by only a few co-raters: s' = s*n/(n+lambda).
        if self.config.shrinkage > 0:
            similarities *= overlaps / (overlaps + self.config.shrinkage)
        similarities[overlaps < self.config.min_overlap] = 0.0
        similarities[similarities < self.config.min_similarity] = 0.0
        np.fill_diagonal(similarities, 0.0)

        self.similarity_ = self._retain_top_neighbors(similarities)
        return self

    def _retain_top_neighbors(self, similarities: np.ndarray) -> sparse.csr_matrix:
        """Store the largest usable similarities for every target game.

        Args:
            similarities: Dense square array with zero diagonal and filtered pairs.
        Returns:
            Item-by-item CSR matrix retaining up to config.neighbors nonzero
            entries per row. Chosen neighbors are ordered by score, then item ID."""

        item_count = similarities.shape[0]
        keep = min(self.config.neighbors, max(item_count - 1, 1))
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for item_id, row in enumerate(similarities):
            nonzero = np.flatnonzero(row)
            if nonzero.size > keep:
                chosen = nonzero[np.argpartition(row[nonzero], -keep)[-keep:]]
            else:
                chosen = nonzero
            # Stable secondary sorting by item ID makes repeated runs identical.
            order = np.lexsort((chosen, -row[chosen]))
            chosen = chosen[order]
            rows.extend([item_id] * len(chosen))
            columns.extend(chosen.tolist())
            values.extend(row[chosen].tolist())
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=similarities.shape)
        matrix.sort_indices()
        return matrix

    def _require_fit(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Check fitted state before reading the learned matrices.

        Returns:
            The user-by-item training CSR and item-by-item similarity CSR.
        Raises:
            RuntimeError: If fit() has not populated both matrices."""

        if self.train_ is None or self.similarity_ is None:
            raise RuntimeError("Call fit() before prediction")
        return self.train_, self.similarity_

    def score_user(self, user_id: int) -> np.ndarray:
        """Predict scores for all games using one user's observed neighbors.

        Baseline-residual scoring adds weighted deviations to the user's mean.
        Weighted-mean scoring aggregates the raw neighboring ratings. A target
        with no usable neighbor receives the training Bayesian popularity score.

        Args:
            user_id: Zero-based row index in the fitted training matrix.
        Returns:
            One score per game, clipped to [1, 10], including train-seen games.
            Call recommend() to exclude already-rated games.
        Raises:
            RuntimeError: If the model is unfitted.
            IndexError: If user_id is outside the fitted matrix."""

        train, similarities = self._require_fit()
        if not 0 <= user_id < train.shape[0]:
            raise IndexError(f"Unknown user_id {user_id}")
        user_row = train.getrow(user_id)
        assert self.popularity_ is not None
        assert self.user_means_ is not None
        user_mean = self.user_means_[user_id]
        if user_row.nnz == 0:
            return self.popularity_.copy()

        if self.config.scoring == "baseline_residual":
            # Rather than averaging raw ratings, predict how far the user rated
            # neighboring games above/below their own mean, then add that mean.
            residual = user_row.copy()
            residual.data -= user_mean
            numerator = (similarities @ residual.T).toarray().ravel()
        else:
            numerator = (similarities @ user_row.T).toarray().ravel()

        observed = user_row.copy()
        observed.data = np.ones_like(observed.data)
        absolute_similarity = similarities.copy()
        absolute_similarity.data = np.abs(absolute_similarity.data)
        denominator = (absolute_similarity @ observed.T).toarray().ravel()

        fallback = np.full(train.shape[1], user_mean, dtype=float)
        if self.config.scoring == "baseline_residual":
            scores = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            ) + user_mean
            scores[denominator == 0] = self.popularity_[denominator == 0]
        else:
            scores = np.divide(
                numerator,
                denominator,
                out=fallback,
                where=denominator > 0,
            )
            scores[denominator == 0] = self.popularity_[denominator == 0]
        return np.clip(scores, 1.0, 10.0)

    def predict(self, user_id: int, item_id: int) -> float:
        """Return one clipped rating estimate from score_user().

        Args:
            user_id: Zero-based fitted user row.
            item_id: Zero-based fitted game column.
        Returns:
            Predicted rating in [1, 10].
        Raises:
            IndexError: If either ID is outside the fitted matrix."""

        scores = self.score_user(user_id)
        if not 0 <= item_id < scores.size:
            raise IndexError(f"Unknown item_id {item_id}")
        return float(scores[item_id])

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Rank games after removing the user's training-rated items.

        Args:
            user_id: Zero-based fitted user row.
            top_k: Maximum number of results; nonpositive values return an empty list.
        Returns:
            (item_id, score) pairs in descending score order. Item ID breaks ties.
            The list can be shorter than top_k if too few unseen games remain."""

        train, _ = self._require_fit()
        if top_k <= 0:
            return []
        scores = self.score_user(user_id)
        seen = train.getrow(user_id).indices
        scores[seen] = -np.inf
        item_ids = np.arange(scores.size)
        order = np.lexsort((item_ids, -scores))
        chosen = order[np.isfinite(scores[order])][:top_k]
        return [(int(item_id), float(scores[item_id])) for item_id in chosen]
