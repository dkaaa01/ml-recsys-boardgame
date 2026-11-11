from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.final_matrix_factorization_test import load_frozen_config


class FinalMatrixFactorizationTest(unittest.TestCase):
    def test_load_frozen_config_restores_model_and_metrics(self) -> None:
        payload = {
            "factors": 16, "learning_rate": 0.02, "regularization": 0.1,
            "epochs": 8, "batch_size": 256, "seed": 3,
            "min_rating": 1.0, "max_rating": 10.0,
            "rmse": 1.2, "recall_at_10": 0.1, "ndcg_at_10": 0.08,
            "coverage": 0.4,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config, metrics = load_frozen_config(path)
        self.assertEqual((config.factors, config.epochs, config.seed), (16, 8, 3))
        self.assertEqual(metrics["ndcg_at_10"], 0.08)


if __name__ == "__main__":
    unittest.main()
