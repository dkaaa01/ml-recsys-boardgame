from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from src.models.content_based import (
    ContentBasedRecommender,
    ContentConfig,
    build_item_features,
)
from src.models.popularity import (
    rank_popular_games,
    recommend_unseen_games,
    write_popularity_outputs,
)


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

            report_dir = root / "report"
            handoff = root / "popularity.csv"
            write_popularity_outputs(
                ranking, metadata, report_dir, csv_output=handoff
            )
            self.assertEqual(
                handoff.read_text(encoding="utf-8"),
                (report_dir / "popular_games.csv").read_text(encoding="utf-8"),
            )


class ContentBasedTest(unittest.TestCase):
    def test_description_and_tags_produce_personalized_unseen_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            games = Path(temporary) / "games.csv"
            with games.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(
                    ("item_id", "description", "categories", "mechanics")
                )
                writer.writerows(
                    (
                        (0, "Explore a space galaxy with starships", "['Space']", "['Exploration']"),
                        (1, "Explore distant planets in the galaxy", "['Space']", "['Exploration']"),
                        (2, "Grow crops and animals on a farm", "['Farming']", "['Worker Placement']"),
                    )
                )
            config = ContentConfig(
                max_description_features=50,
                min_document_frequency=1,
                description_weight=0.5,
            )
            features = build_item_features(games, config, expected_items=3)
            ratings = sparse.csr_matrix(
                np.array(((9.0, 0.0, 0.0), (0.0, 0.0, 9.0)), dtype=np.float32)
            )
            model = ContentBasedRecommender(config).fit(ratings, features.matrix)

            recommendations, scores = model.recommend([0], top_k=2, batch_size=1)

            self.assertEqual(recommendations[0][0], 1)
            self.assertNotIn(0, recommendations[0])
            self.assertGreater(scores[0][0], scores[0][1])
            predictions = model.predict_pairs([0, 1], [1, 2])
            self.assertTrue(np.all((predictions >= 1.0) & (predictions <= 10.0)))

    def test_invalid_content_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContentConfig(description_weight=1.5)
        with self.assertRaises(ValueError):
            ContentConfig(max_description_features=0)


if __name__ == "__main__":
    unittest.main()

