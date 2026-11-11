from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.matrix_factorization_outputs import save_final_outputs, save_search_outputs
from src.models.matrix_factorization import MatrixFactorizationConfig


class MatrixFactorizationOutputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "trial": 0,
            "factors": 16,
            "learning_rate": 0.01,
            "regularization": 0.05,
            "epochs": 8,
            "batch_size": 256,
            "seed": 42,
            "min_rating": 1.0,
            "max_rating": 10.0,
            "rmse": 1.2,
            "recall_at_10": 0.1,
            "ndcg_at_10": 0.08,
            "coverage": 0.4,
            "rating_predictions": 100.0,
            "ranking_users": 10.0,
            "train_rmse": 1.0,
            "fit_seconds": 0.1,
            "evaluation_seconds": 0.2,
        }

    def test_search_outputs_write_trials_and_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_search_outputs([self.row], self.row, output_dir)
            self.assertTrue((output_dir / "mf_trials.csv").exists())
            self.assertTrue((output_dir / "best_mf.json").exists())
            self.assertTrue((output_dir / "ranking_metrics.png").exists())

    def test_final_outputs_preserve_evaluation_protocol(self) -> None:
        metrics = {
            "rmse": 1.2,
            "recall_at_10": 0.1,
            "ndcg_at_10": 0.08,
            "coverage": 0.4,
            "rating_predictions": 100.0,
            "ranking_users": 10.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_final_outputs(
                output_dir,
                MatrixFactorizationConfig(factors=16, epochs=8),
                metrics,
                metrics,
                fit_seconds=0.1,
                evaluation_seconds=0.2,
            )
            payload = (output_dir / "final_test_metrics.json").read_text(encoding="utf-8")
            self.assertIn("held-out test", payload)
            self.assertTrue((output_dir / "validation_vs_test.png").exists())


if __name__ == "__main__":
    unittest.main()
