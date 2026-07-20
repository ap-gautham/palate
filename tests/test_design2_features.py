import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from design2_xgboost.features import MAIN_COLS, FEATURE_COLS, main_feature_row


class FeatureContractTests(unittest.TestCase):
    def test_main_feature_row_has_stable_columns_and_decile_summary(self):
        tail = {
            "n_observed": 5,
            "mean_overlap": 3.0,
            "max_overlap": 5.0,
            "tomatometer": 3.5,
            "n_reviewers": 4,
            "dispersion": 0.8,
            "genre_id": 2,
            "user_mean": 3.2,
        }

        row = main_feature_row(
            np.array([0.5, -0.2]), np.array([2.0, 1.0]), tail)

        self.assertEqual(list(row), FEATURE_COLS)
        self.assertAlmostEqual(row[MAIN_COLS[0]], 1.0)
        self.assertEqual(row[MAIN_COLS[10]], 1.0)
        self.assertEqual(row["genre_id"], 2)


if __name__ == "__main__":
    unittest.main()