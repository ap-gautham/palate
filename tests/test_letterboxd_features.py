import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from letterboxd.features import (FEATURE_COLS, MAIN_COLS, GENRE_WIDTH,
                                 FacetContext, _facet_tail, _genre_block,
                                 _theme_block, _actor_block, _director_block,
                                 facet_tail_from_context, main_feature_row)


def _tail(**overrides):
    tail = {
        "n_observed": 5, "mean_overlap": 3.0, "max_overlap": 5.0,
        "n_reviewers": 4, "dispersion": 0.8, "user_mean": 3.2,
    }
    tail.update(_facet_tail(
        frozenset(), frozenset(), frozenset(), frozenset(),
        float("nan"), float("nan"), 0, 0,
        [], [], [], [], np.array([]), np.zeros((1, 1), dtype=np.float32), global_std=1.0))
    tail.update(overrides)
    return tail


class LetterboxdFeatureContractTests(unittest.TestCase):
    """Mirrors tests/test_design2_features.py -- Letterboxd's affinity blocks
    are identical logic to Rotten Tomatoes' (see features.py docstrings), so
    this file exercises the same properties on the Letterboxd module (there
    was no Letterboxd test at all before this contract, an asymmetry with RT
    fixed alongside the new feature contract)."""

    def test_main_feature_row_has_stable_columns_and_decile_summary(self):
        row = main_feature_row(np.array([0.5, -0.2]), np.array([2.0, 1.0]), _tail())
        self.assertEqual(set(row), set(FEATURE_COLS))
        self.assertEqual(len(row), len(FEATURE_COLS))
        self.assertNotIn("tomatometer", FEATURE_COLS)  # LB has no critic-consensus meter
        self.assertEqual(len(FEATURE_COLS), 115)
        self.assertNotIn("genre_id", FEATURE_COLS)
        self.assertAlmostEqual(row[MAIN_COLS[0]], 1.0)
        self.assertEqual(row[MAIN_COLS[10]], 1.0)
        self.assertTrue(np.isnan(row["user_genre_0_z"]))
        self.assertEqual(row["user_genre_0_cnt"], 0.0)
        self.assertEqual(row["runtime_log"], 0.0)

    def test_genre_block_z_score_excludes_target_and_counts_include_it(self):
        target_genre_ids = frozenset({0})
        seen_genre_ids = [frozenset({0}), frozenset({1}), frozenset({0, 2})]
        seen_values = np.array([4.0, 1.0, 2.0])
        mu_u, sigma_u = float(seen_values.mean()), float(seen_values.std(ddof=0))
        out = _genre_block(target_genre_ids, seen_genre_ids, seen_values, mu_u, sigma_u)
        expected_z0 = (np.mean([4.0, 2.0]) - mu_u) / sigma_u
        self.assertAlmostEqual(out["user_genre_0_z"], expected_z0)
        self.assertAlmostEqual(out["user_genre_0_cnt"], np.log1p(2 + 1))
        self.assertEqual(out["user_genre_3_cnt"], 0.0)
        self.assertTrue(np.isnan(out["user_genre_3_z"]))

    def test_theme_block_weights_by_similarity_and_masks_via_mass_log(self):
        theme_matrix = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        out = _theme_block(frozenset({0}), [frozenset({1}), frozenset({2})],
                           np.array([4.0, 1.0]), theme_matrix)
        self.assertAlmostEqual(out["user_theme_avg"], 4.0)
        self.assertAlmostEqual(out["user_theme_mass_log"], np.log1p(1.0))
        self.assertAlmostEqual(out["user_theme_simcnt_hi"], np.log1p(1))

    def test_theme_block_no_target_themes_is_nan_and_zero_mass(self):
        out = _theme_block(frozenset(), [frozenset({0})], np.array([4.0]),
                           np.zeros((1, 1), dtype=np.float32))
        self.assertTrue(np.isnan(out["user_theme_avg"]))
        self.assertEqual(out["user_theme_mass_log"], 0.0)

    def test_actor_block_rankings_and_padding(self):
        target_cast = frozenset({"A", "B"})
        seen_cast_list = [frozenset({"A", "B"}), frozenset({"A"}), frozenset({"C"})]
        seen_values = np.array([5.0, 5.0, 1.0])
        mu_u, sigma_u = float(seen_values.mean()), float(seen_values.std(ddof=0))
        out = _actor_block(target_cast, seen_cast_list, seen_values, mu_u, sigma_u)
        self.assertAlmostEqual(out["user_actor_byrating1_z"], (5.0 - mu_u) / sigma_u)
        self.assertAlmostEqual(out["user_actor_bycount1_cnt"], np.log1p(2))
        self.assertTrue(np.isnan(out["user_actor_byrating3_z"]))
        self.assertEqual(out["user_actor_byrating3_cnt"], 0.0)
        self.assertAlmostEqual(out["user_cast_overlap1_n"], np.log1p(2))
        self.assertAlmostEqual(out["user_cast_overlap1_rating"], 5.0)
        self.assertEqual(out["user_cast_overlap3_n"], 0.0)
        self.assertTrue(np.isnan(out["user_cast_overlap3_rating"]))

    def test_director_block_any_shared_director_counts_as_a_hit(self):
        target_directors = frozenset({"Villeneuve"})
        seen_director_list = [frozenset({"Villeneuve", "Deakins"}), frozenset({"Nolan"})]
        seen_values = np.array([5.0, 3.0])
        mu_u, sigma_u = float(seen_values.mean()), float(seen_values.std(ddof=0))
        out = _director_block(target_directors, seen_director_list, seen_values, mu_u, sigma_u)
        self.assertAlmostEqual(out["user_director_z"], (5.0 - mu_u) / sigma_u)
        self.assertAlmostEqual(out["user_director_cnt"], np.log1p(1))

    def test_no_match_gives_nan_average_and_zero_count_everywhere(self):
        out = _facet_tail(
            frozenset(), frozenset(), frozenset(), frozenset(),
            float("nan"), float("nan"), 0, 0,
            [], [], [], [], np.array([]), np.zeros((1, 1), dtype=np.float32), global_std=1.0)
        for g in range(GENRE_WIDTH):
            self.assertEqual(out[f"user_genre_{g}_cnt"], 0.0)
            self.assertTrue(np.isnan(out[f"user_genre_{g}_z"]))
        self.assertTrue(np.isnan(out["user_theme_avg"]))
        for i in range(1, 6):
            self.assertEqual(out[f"user_actor_byrating{i}_cnt"], 0.0)
            self.assertTrue(np.isnan(out[f"user_actor_byrating{i}_z"]))
        self.assertEqual(out["user_director_cnt"], 0.0)
        self.assertTrue(np.isnan(out["user_director_z"]))

    def test_facet_context_positions_align_with_movie_index(self):
        theme_matrix = np.zeros((1, 1), dtype=np.float32)
        fc = FacetContext(
            genre_ids=[frozenset({0}), frozenset({1})],
            theme_ids=[frozenset(), frozenset()],
            actor_sets=[frozenset(), frozenset()],
            director_sets=[frozenset(), frozenset()],
            runtime_log=np.array([4.5, 4.6]), gs_rating=np.array([3.5, 3.0]),
            n_themes=np.array([2.0, 1.0]), n_languages=np.array([1.0, 1.0]),
            theme_matrix=theme_matrix, global_std=1.0)
        out = facet_tail_from_context(fc, np.array([1]), np.array([2.0]), target_col=0)
        self.assertTrue(np.isnan(out["user_genre_0_z"]))
        self.assertAlmostEqual(out["user_genre_0_cnt"], np.log1p(1))
        self.assertAlmostEqual(out["user_genre_1_z"], 0.0)
        self.assertAlmostEqual(out["user_genre_1_cnt"], np.log1p(1))


if __name__ == "__main__":
    unittest.main()
