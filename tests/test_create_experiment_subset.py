from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.create_experiment_subset import create_experiment_subset


class CreateExperimentSubsetTest(unittest.TestCase):
    def test_selects_from_train_and_remaps_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            output = root / "experiment"
            processed.mkdir()

            splits = {
                "train": [
                    (10, 100, 9), (10, 200, 8),
                    (20, 100, 7), (20, 200, 6),
                    (30, 100, 9), (30, 300, 8),
                    (40, 100, 5),
                ],
                "validation": [(10, 200, 9), (20, 300, 10), (30, 100, 8)],
                "test": [(10, 100, 8), (20, 200, 7), (40, 100, 9)],
            }
            for split, rows in splits.items():
                with (processed / f"{split}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as target:
                    writer = csv.writer(target)
                    writer.writerow(("user_id", "item_id", "rating"))
                    writer.writerows(rows)
            with (processed / "games.csv").open(
                "w", encoding="utf-8", newline=""
            ) as target:
                writer = csv.writer(target)
                writer.writerow(("item_id", "bgg_id", "name"))
                writer.writerows(((100, 1, "A"), (200, 2, "B"), (300, 3, "C")))

            report = create_experiment_subset(
                processed,
                output,
                max_items=2,
                max_users=2,
                min_user_train_ratings=2,
                seed=42,
            )

            self.assertEqual(report["users"], 2)
            self.assertEqual(report["items"], 2)
            self.assertEqual(report["selection_source"], "train split only")
            self.assertEqual(json.loads((output / "subset_report.json").read_text())["seed"], 42)

            with (output / "train.csv").open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual({int(row["user_id"]) for row in rows}, {0, 1})
            self.assertTrue({int(row["item_id"]) for row in rows}.issubset({0, 1}))
            with (output / "games.csv").open(encoding="utf-8") as source:
                games = list(csv.DictReader(source))
            self.assertEqual({int(row["item_id"]) for row in games}, {0, 1})
            self.assertEqual({int(row["source_item_id"]) for row in games}, {100, 200})


if __name__ == "__main__":
    unittest.main()
