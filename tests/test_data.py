from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.loader import load_interactions, sample_user_ids


class DataTest(unittest.TestCase):
    def test_loads_sparse_matrix_and_filters_users(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ratings.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(("user_id", "item_id", "rating"))
                writer.writerows(((0, 0, 8), (1, 1, 7), (2, 2, 9)))
            loaded = load_interactions(path, shape=(3, 3), user_filter={0, 2})
            self.assertEqual(loaded.interactions, 2)
            self.assertEqual(loaded.ratings.nnz, 2)
            self.assertEqual(float(loaded.ratings[2, 2]), 9.0)

    def test_user_sample_is_deterministic(self) -> None:
        first = sample_user_ids(100, 10, seed=42)
        second = sample_user_ids(100, 10, seed=42)
        self.assertEqual(first.tolist(), second.tolist())


if __name__ == "__main__":
    unittest.main()
