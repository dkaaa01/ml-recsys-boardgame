import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.final_item_cf_test import load_frozen_config, write_final_report
from src.models.item_cf import ItemCFConfig


class FinalItemCFTest(unittest.TestCase):
    def test_loads_frozen_config_and_validation_metrics(self) -> None:
        winner = {
            "similarity": "adjusted_cosine",
            "neighbors": "120",
            "min_overlap": "10",
            "shrinkage": "250.0",
            "min_similarity": "0.1",
            "positive_threshold": "8.0",
            "scoring": "weighted_mean",
            "rmse": "1.4",
            "recall_at_10": "0.09",
            "ndcg_at_10": "0.07",
            "coverage": "0.11",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            path.write_text(json.dumps(winner), encoding="utf-8")
            config, metrics = load_frozen_config(path)

        self.assertEqual(config.similarity, "adjusted_cosine")
        self.assertEqual(config.neighbors, 120)
        self.assertAlmostEqual(metrics["ndcg_at_10"], 0.07)

    def test_report_title_uses_frozen_similarity(self) -> None:
        metrics = {
            "rmse": 1.0,
            "recall_at_10": 0.1,
            "ndcg_at_10": 0.1,
            "coverage": 0.5,
            "rating_predictions": 1.0,
            "ranking_users": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "matplotlib.figure.Figure.suptitle"
        ) as suptitle:
            write_final_report(
                Path(directory),
                ItemCFConfig(similarity="pearson"),
                metrics,
                metrics,
                fit_seconds=0.0,
                evaluation_seconds=0.0,
            )

        suptitle.assert_called_once_with(
            "Frozen Pearson Model: Validation vs Test"
        )


if __name__ == "__main__":
    unittest.main()
