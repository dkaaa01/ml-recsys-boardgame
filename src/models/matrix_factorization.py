"""Biased Matrix Factorization for explicit board-game ratings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MatrixFactorizationConfig:
    """Hyperparameters for mini-batch SGD training."""

    factors: int = 32
    learning_rate: float = 0.01
    regularization: float = 0.05
    epochs: int = 12
    batch_size: int = 4096
    seed: int = 42
    min_rating: float = 1.0
    max_rating: float = 10.0

    def __post_init__(self) -> None:
        if self.factors <= 0:
            raise ValueError("factors must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.min_rating >= self.max_rating:
            raise ValueError("min_rating must be less than max_rating")


class MatrixFactorization:
    """Learn user/game embeddings plus user and game rating biases.

    The predicted rating is ``global mean + user bias + item bias + dot
    product``. Updates use deterministic shuffled mini-batches. ``np.add.at``
    accumulates repeated user and item gradients without a Python loop.
    """

    def __init__(self, config: MatrixFactorizationConfig | None = None) -> None:
        self.config = config or MatrixFactorizationConfig()
        self.train_: sparse.csr_matrix | None = None
        self.global_mean_: float | None = None
        self.user_bias_: np.ndarray | None = None
        self.item_bias_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.training_rmse_: list[float] = []

    def fit(self, ratings: sparse.csr_matrix) -> "MatrixFactorization":
        """Fit the model to observed cells of a sparse rating matrix."""

        train = ratings.tocsr().astype(np.float64)
        if train.nnz == 0:
            raise ValueError("ratings must contain at least one interaction")
        values = train.data
        if values.min() < self.config.min_rating or values.max() > self.config.max_rating:
            raise ValueError("ratings fall outside the configured rating range")

        coo = train.tocoo()
        users = coo.row.astype(np.int64, copy=False)
        items = coo.col.astype(np.int64, copy=False)
        observed = coo.data.astype(np.float64, copy=False)
        generator = np.random.default_rng(self.config.seed)
        self._initialize(train, float(observed.mean()), generator)
        self.training_rmse_ = []

        order = np.arange(observed.size)
        for _ in range(self.config.epochs):
            generator.shuffle(order)
            for start in range(0, order.size, self.config.batch_size):
                batch = order[start : start + self.config.batch_size]
                self._update_batch(users[batch], items[batch], observed[batch])
            error = observed - self._raw_predict(users, items)
            self.training_rmse_.append(float(np.sqrt(np.mean(error**2))))
        return self

    def _initialize(
        self,
        train: sparse.csr_matrix,
        global_mean: float,
        generator: np.random.Generator,
    ) -> None:
        scale = 0.1 / np.sqrt(self.config.factors)
        self.train_ = train
        self.global_mean_ = global_mean
        self.user_bias_ = np.zeros(train.shape[0], dtype=np.float64)
        self.item_bias_ = np.zeros(train.shape[1], dtype=np.float64)
        self.user_factors_ = generator.normal(
            0.0, scale, (train.shape[0], self.config.factors)
        )
        self.item_factors_ = generator.normal(
            0.0, scale, (train.shape[1], self.config.factors)
        )

    def _update_batch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        assert self.train_ is not None and self.global_mean_ is not None
        assert self.user_bias_ is not None and self.item_bias_ is not None
        assert self.user_factors_ is not None and self.item_factors_ is not None

        user_vectors = self.user_factors_[users].copy()
        item_vectors = self.item_factors_[items].copy()
        predicted = (
            self.global_mean_
            + self.user_bias_[users]
            + self.item_bias_[items]
            + np.einsum("ij,ij->i", user_vectors, item_vectors)
        )
        errors = ratings - predicted
        user_weights = _inverse_frequency(users, self.train_.shape[0])
        item_weights = _inverse_frequency(items, self.train_.shape[1])
        step = self.config.learning_rate
        penalty = self.config.regularization

        np.add.at(
            self.user_bias_, users,
            step * user_weights * (errors - penalty * self.user_bias_[users]),
        )
        np.add.at(
            self.item_bias_, items,
            step * item_weights * (errors - penalty * self.item_bias_[items]),
        )
        np.add.at(
            self.user_factors_, users,
            step * user_weights[:, None]
            * (errors[:, None] * item_vectors - penalty * user_vectors),
        )
        np.add.at(
            self.item_factors_, items,
            step * item_weights[:, None]
            * (errors[:, None] * user_vectors - penalty * item_vectors),
        )

    def _require_fitted(self) -> None:
        if self.train_ is None:
            raise RuntimeError("fit must be called before prediction")

    def _raw_predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.global_mean_ is not None
        assert self.user_bias_ is not None and self.item_bias_ is not None
        assert self.user_factors_ is not None and self.item_factors_ is not None
        return (
            self.global_mean_
            + self.user_bias_[users]
            + self.item_bias_[items]
            + np.einsum("ij,ij->i", self.user_factors_[users], self.item_factors_[items])
        )

    def predict_pairs(self, users, items) -> np.ndarray:
        """Predict aligned user/item arrays and clip to the rating scale."""

        user_ids = np.asarray(users, dtype=np.int64)
        item_ids = np.asarray(items, dtype=np.int64)
        if user_ids.shape != item_ids.shape:
            raise ValueError("users and items must have the same shape")
        self._require_fitted()
        assert self.train_ is not None
        if (
            np.any(user_ids < 0)
            or np.any(user_ids >= self.train_.shape[0])
            or np.any(item_ids < 0)
            or np.any(item_ids >= self.train_.shape[1])
        ):
            raise IndexError("user or item ID is outside the fitted matrix")
        return np.clip(
            self._raw_predict(user_ids.ravel(), item_ids.ravel()).reshape(user_ids.shape),
            self.config.min_rating,
            self.config.max_rating,
        )

    def score_user(self, user_id: int) -> np.ndarray:
        """Return predicted ratings for every catalog item for one user."""

        self._require_fitted()
        assert self.train_ is not None
        if not 0 <= user_id < self.train_.shape[0]:
            raise IndexError("user ID is outside the fitted matrix")
        items = np.arange(self.train_.shape[1], dtype=np.int64)
        users = np.full(items.shape, user_id, dtype=np.int64)
        return self.predict_pairs(users, items)


def _inverse_frequency(ids: np.ndarray, size: int) -> np.ndarray:
    """Normalize repeated IDs so each entity receives one mean batch update."""

    counts = np.bincount(ids, minlength=size)
    return 1.0 / counts[ids]
