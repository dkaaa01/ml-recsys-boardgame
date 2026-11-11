from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from src.models.item_cf import ItemCFConfig, ItemItemCF


class ItemItemCFTest(unittest.TestCase):
    def setUp(self) -> None:
        # Users 0 and 1 like items 0/1 together; item 2 has an opposing pattern.
        self.ratings = sparse.csr_matrix(
            np.array(
                [
                    [10.0, 9.0, 2.0, 0.0],
                    [9.0, 8.0, 1.0, 0.0],
                    [0.0, 8.0, 2.0, 9.0],
                    [8.0, 0.0, 0.0, 7.0],
                ]
            )
        )

    def test_all_similarity_families_fit_and_predict(self) -> None:
        for name in ("cosine", "adjusted_cosine", "pearson", "jaccard"):
            with self.subTest(similarity=name):
                model = ItemItemCF(
                    ItemCFConfig(
                        similarity=name,
                        neighbors=3,
                        min_overlap=1,
                        shrinkage=0,
                        positive_threshold=8,
                    )
                ).fit(self.ratings)
                prediction = model.predict(0, 3)
                self.assertGreaterEqual(prediction, 1.0)
                self.assertLessEqual(prediction, 10.0)

    def test_recommendations_exclude_seen_items_and_are_deterministic(self) -> None:
        model = ItemItemCF(
            ItemCFConfig(neighbors=3, min_overlap=1, shrinkage=0)
        ).fit(self.ratings)
        first = model.recommend(0, top_k=2)
        second = model.recommend(0, top_k=2)
        self.assertEqual(first, second)
        self.assertTrue({item for item, _ in first}.isdisjoint({0, 1, 2}))

    def test_shrinkage_reduces_similarity_supported_by_few_users(self) -> None:
        plain = ItemItemCF(
            ItemCFConfig(similarity="cosine", neighbors=3, min_overlap=1, shrinkage=0)
        ).fit(self.ratings)
        shrunk = ItemItemCF(
            ItemCFConfig(similarity="cosine", neighbors=3, min_overlap=1, shrinkage=100)
        ).fit(self.ratings)
        self.assertLess(shrunk.similarity_.max(), plain.similarity_.max())

    def test_pearson_uses_common_rater_correlation(self) -> None:
        ratings = sparse.csr_matrix(
            np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        )
        model = ItemItemCF(
            ItemCFConfig(
                similarity="pearson",
                neighbors=1,
                min_overlap=1,
                shrinkage=0,
                min_similarity=-1.0,
            )
        ).fit(ratings)
        self.assertAlmostEqual(float(model.similarity_[0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
