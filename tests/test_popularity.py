from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.models.popularity import rank_popular_games, recommend_unseen_games


class PopularityTest(unittest.TestCase):
    def test_ranking_uses_training_data_and_game_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv"
            games = root / "games.csv"
            with train.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(("user_id", "item_id", "rating"))
                writer.writerows(((0, 0, 10), (1, 0, 9), (0, 1, 6), (1, 1, 6)))
            with games.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(("item_id", "bgg_id", "name"))
                writer.writerows(((0, 100, "High Game"), (1, 200, "Average Game")))

            ranking, metadata = rank_popular_games(train, games, prior_quantile=0.5)

            self.assertEqual(ranking[0]["name"], "High Game")
            self.assertAlmostEqual(metadata["global_mean_rating"], 7.75)
            recommendations = recommend_unseen_games(ranking, seen_item_ids={0}, top_k=1)
            self.assertEqual(recommendations[0]["name"], "Average Game")


if __name__ == "__main__":
    unittest.main()

