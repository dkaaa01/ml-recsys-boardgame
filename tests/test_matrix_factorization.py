from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from src.models.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig


class MatrixFactorizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = sparse.csr_matrix(
            np.array(
                [
                    [9.0, 8.0, 0.0, 0.0],
                    [8.0, 9.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0, 1.0],
                    [0.0, 3.0, 1.0, 2.0],
                ]
            )
        )
        self.config = MatrixFactorizationConfig(
            factors=4, learning_rate=0.05, regularization=0.01,
            epochs=30, batch_size=4, seed=7,
        )

    def test_training_is_deterministic_and_reduces_error(self) -> None:
        first = MatrixFactorization(self.config).fit(self.ratings)
        second = MatrixFactorization(self.config).fit(self.ratings)
        np.testing.assert_allclose(first.user_factors_, second.user_factors_)
        self.assertLess(first.training_rmse_[-1], first.training_rmse_[0])

    def test_scores_match_pair_predictions_and_stay_in_range(self) -> None:
        model = MatrixFactorization(self.config).fit(self.ratings)
        scores = model.score_user(1)
        pairs = model.predict_pairs(np.full(4, 1), np.arange(4))
        np.testing.assert_allclose(scores, pairs)
        self.assertTrue(np.all((scores >= 1.0) & (scores <= 10.0)))

    def test_prediction_requires_fit(self) -> None:
        with self.assertRaises(RuntimeError):
            MatrixFactorization().score_user(0)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MatrixFactorizationConfig(factors=0)
        with self.assertRaises(ValueError):
            MatrixFactorizationConfig(regularization=-0.1)


if __name__ == "__main__":
    unittest.main()
