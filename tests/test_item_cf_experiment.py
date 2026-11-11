from __future__ import annotations

import unittest

from src.experiments.item_cf_experiment import build_local_grid, build_trial_configs
from src.models.item_cf import ItemCFConfig


class ItemCFExperimentTest(unittest.TestCase):
    def test_search_is_deterministic_and_contains_all_similarities(self) -> None:
        first = build_trial_configs(8, seed=42, sample_users=100, total_users=1000)
        second = build_trial_configs(8, seed=42, sample_users=100, total_users=1000)
        self.assertEqual(first, second)
        self.assertEqual(
            {config.similarity for config in first},
            {"cosine", "adjusted_cosine", "pearson", "jaccard"},
        )

    def test_requires_at_least_four_trials(self) -> None:
        with self.assertRaises(ValueError):
            build_trial_configs(3, seed=42, sample_users=100, total_users=1000)

    def test_search_gives_similarity_families_balanced_budgets(self) -> None:
        configs = build_trial_configs(12, seed=42, sample_users=100, total_users=1000)
        counts = {
            name: sum(config.similarity == name for config in configs)
            for name in {config.similarity for config in configs}
        }
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_local_grid_keeps_model_family_and_changes_search_values(self) -> None:
        best = ItemCFConfig(
            similarity="pearson", neighbors=40, min_overlap=2, shrinkage=25
        )
        grid = build_local_grid(
            best, sample_users=100, total_users=1000, max_trials=6
        )
        self.assertEqual(len(grid), 6)
        self.assertTrue(all(config.similarity == "pearson" for config in grid))
        self.assertTrue(all(config != best for config in grid))


if __name__ == "__main__":
    unittest.main()
