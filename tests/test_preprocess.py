from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import ingest_ratings, preprocess


class PreprocessTest(unittest.TestCase):
    def _write_metadata(self, path: Path, item_ids: range) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                (
                    "id",
                    "name",
                    "description",
                    "yearpublished",
                    "minplayers",
                    "maxplayers",
                    "minplaytime",
                    "maxplaytime",
                    "minage",
                    "average",
                    "bayesaverage",
                    "averageweight",
                    "usersrated",
                    "boardgamecategory",
                    "boardgamemechanic",
                )
            )
            for item_id in item_ids:
                writer.writerow(
                    (
                        item_id,
                        f"Game {item_id}",
                        f"Description {item_id}",
                        2025,
                        2,
                        4,
                        30,
                        60,
                        10,
                        8.0,
                        7.5,
                        2.5,
                        100,
                        "['Strategy']",
                        "['Drafting']",
                    )
                )

    def test_ingest_removes_exact_duplicates_and_conflicting_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ratings = root / "bgg-26m-reviews.csv"
            database = root / "ratings.sqlite3"
            with ratings.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(("user", "rating", "ID"))
                writer.writerows(
                    (
                        ("alice", 8, 1),
                        ("alice", 8, 1),
                        ("alice", 9, 2),
                        ("alice", 7, 2),
                        ("bob", 7.5, 1),
                        ("", 9, 1),
                        ("bob", 11, 2),
                    )
                )

            report = ingest_ratings(ratings, database, batch_size=2)

            self.assertEqual(report["raw_rows"], 7)
            self.assertEqual(report["invalid_rows"], 2)
            self.assertEqual(report["exact_duplicate_rows_removed"], 1)
            self.assertEqual(report["conflicting_user_item_pairs_removed"], 1)
            self.assertEqual(report["conflicting_input_rows_removed"], 2)
            self.assertEqual(report["clean_user_item_pairs"], 2)
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT user, item, rating FROM clean_ratings ORDER BY user, item"
                ).fetchall()
            self.assertEqual(rows, [("alice", "1", 8.0), ("bob", "1", 7.5)])

    def test_end_to_end_writes_common_splits_and_eval_users(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            ratings = raw / "bgg-26m-reviews.csv"
            with ratings.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("user", "rating", "ID"))
                for user_index in range(8):
                    for item_id in range(1, 7):
                        writer.writerow((f"user-{user_index}", 8 + item_id % 3, item_id))
            self._write_metadata(raw / "games_detailed_info2025.csv", range(1, 7))

            report = preprocess(
                raw,
                output,
                min_user_train_ratings=1,
                min_item_train_ratings=1,
                max_items=6,
                seed=42,
            )

            self.assertEqual(report["raw_rows"], 48)
            self.assertGreater(report["processed_users"], 0)
            self.assertGreater(report["evaluation_users"], 0)
            self.assertTrue((output / "train.csv").exists())
            self.assertTrue((output / "validation.csv").exists())
            self.assertTrue((output / "test.csv").exists())
            self.assertTrue((output / "eval_users.csv").exists())
            saved = json.loads((output / "preprocessing_report.json").read_text())
            self.assertEqual(saved["dataset_version"], 4)


if __name__ == "__main__":
    unittest.main()
