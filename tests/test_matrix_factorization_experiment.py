from __future__ import annotations

import unittest

from src.experiments.matrix_factorization_experiment import build_trial_configs


class MatrixFactorizationExperimentTest(unittest.TestCase):
    def test_search_is_deterministic_and_unique(self) -> None:
        first = build_trial_configs(8, seed=42)
        self.assertEqual(first, build_trial_configs(8, seed=42))
        self.assertEqual(len(set(first)), 8)

    def test_search_starts_with_reference(self) -> None:
        config = build_trial_configs(1, seed=7)[0]
        self.assertEqual((config.factors, config.learning_rate), (32, 0.01))
        self.assertEqual((config.regularization, config.epochs), (0.05, 12))
        self.assertEqual(config.seed, 7)

    def test_search_requires_positive_count(self) -> None:
        with self.assertRaises(ValueError):
            build_trial_configs(0, seed=42)


if __name__ == "__main__":
    unittest.main()
