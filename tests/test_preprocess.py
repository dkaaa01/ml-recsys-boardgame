from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import preprocess, stable_split


class PreprocessTest(unittest.TestCase):
    def test_stable_split_is_deterministic(self) -> None:
        self.assertEqual(stable_split("alice", "10", 42), stable_split("alice", "10", 42))

    def test_filters_encodes_splits_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            with (raw / "bgg-19m-reviews.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("user", "rating", "ID", "name"))
                writer.writerows(
                    (
                        ("alice", 8, 10, "A"),
                        ("alice", 7, 11, "B"),
                        ("bob", 9, 10, "A"),
                        ("bob", 6, 12, "C"),
                        ("rare", 10, 13, "D"),
                        ("", 7, 10, "A"),
                        ("alice", 0.5, 14, "Invalid"),
                    )
                )
            with (raw / "games_detailed_info.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", "primary", "yearpublished"))
                writer.writerows(((10, "A", 2020), (11, "B", 2021), (12, "C", 2022)))

            report = preprocess(
                raw,
                output,
                min_user_ratings=2,
                min_item_ratings=1,
                max_items=3,
            )

            self.assertEqual(report["raw_rows"], 7)
            self.assertEqual(report["invalid_rows"], 2)
            self.assertEqual(report["processed_users"], 2)
            self.assertEqual(report["processed_items"], 3)
            self.assertEqual(report["processed_game_metadata_rows"], 3)
            self.assertEqual(sum(report["split_counts"].values()), 4)
            saved = json.loads((output / "preprocessing_report.json").read_text())
            self.assertEqual(saved["processed_users"], 2)
            self.assertTrue((output / "users.csv").exists())
            self.assertTrue((output / "items.csv").exists())


if __name__ == "__main__":
    unittest.main()

