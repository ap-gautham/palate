import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotten_tomatoes.features import (FEATURE_COLS, MAIN_COLS, GENRE_WIDTH,
                                      FacetContext, _facet_tail, _genre_block,
                                      _theme_block, _actor_block, _director_block,
                                      facet_tail_from_context, main_feature_row)


def _tail(**overrides):
    tail = {
        "n_observed": 5, "mean_overlap": 3.0, "max_overlap": 5.0,
        "tomatometer": 3.5, "n_reviewers": 4, "dispersion": 0.8,
        "user_mean": 3.2,
    }
    tail.update(_facet_tail(
        frozenset(), frozenset(), frozenset(), frozenset(),
        float("nan"), float("nan"), 0, 0,
        [], [], [], [], np.array([]), np.zeros((1, 1), dtype=np.float32), global_std=1.0))
    tail.update(overrides)
    return tail


class FeatureContractTests(unittest.TestCase):
    def test_main_feature_row_has_stable_columns_and_decile_summary(self):
        # Every block absent (target has no genre/theme/actor/director data)
        # -- every z-score/average must default to NaN, every count to 0.
        row = main_feature_row(np.array([0.5, -0.2]), np.array([2.0, 1.0]), _tail())

        # main_feature_row's contract is the column *set* (any downstream
        # pd.DataFrame(rows, columns=FEATURE_COLS) reindexes explicitly), not
        # dict insertion order.
        self.assertEqual(set(row), set(FEATURE_COLS))
        self.assertEqual(len(row), len(FEATURE_COLS))
        self.assertEqual(len(FEATURE_COLS), 116)
        self.assertNotIn("genre_id", FEATURE_COLS)
        self.assertAlmostEqual(row[MAIN_COLS[0]], 1.0)
        self.assertEqual(row[MAIN_COLS[10]], 1.0)
        self.assertTrue(np.isnan(row["user_genre_0_z"]))
        self.assertEqual(row["user_genre_0_cnt"], 0.0)
        self.assertEqual(row["runtime_log"], 0.0)  # NaN target numeric -> neutral 0

    def test_genre_block_z_score_excludes_target_and_counts_include_it(self):
        """Two seen films are genre 0 (target's genre), one is genre 1 only;
        the z-score must average only the two hits (never the target's own
        unknown rating), and the count must be len(hits) + 1 for the target's
        own genre (0), plain len(hits) for a genre the target doesn't have."""
        target_genre_ids = frozenset({0})
        seen_genre_ids = [frozenset({0}), frozenset({1}), frozenset({0, 2})]
        seen_values = np.array([4.0, 1.0, 2.0])  # mu_u=(4+1+2)/3, sigma_u=std(...)
        mu_u, sigma_u = float(seen_values.mean()), float(seen_values.std(ddof=0))
        out = _genre_block(target_genre_ids, seen_genre_ids, seen_values, mu_u, sigma_u)
        expected_z0 = (np.mean([4.0, 2.0]) - mu_u) / sigma_u
        self.assertAlmostEqual(out["user_genre_0_z"], expected_z0)
        self.assertAlmostEqual(out["user_genre_0_cnt"], np.log1p(2 + 1))  # 2 seen + target itself
        expected_z1 = (1.0 - mu_u) / sigma_u
        self.assertAlmostEqual(out["user_genre_1_z"], expected_z1)
        self.assertAlmostEqual(out["user_genre_1_cnt"], np.log1p(1))  # target isn't genre 1
        # genre 2: seen film 2 ({0, 2}) is a hit; target doesn't have genre 2.
        expected_z2 = (2.0 - mu_u) / sigma_u
        self.assertAlmostEqual(out["user_genre_2_z"], expected_z2)
        self.assertAlmostEqual(out["user_genre_2_cnt"], np.log1p(1))
        self.assertEqual(out["user_genre_3_cnt"], 0.0)  # no seen film, target doesn't have it either
        self.assertTrue(np.isnan(out["user_genre_3_z"]))

    def test_genre_block_falls_back_to_global_std_when_seen_set_is_constant(self):
        seen_genre_ids = [frozenset({0}), frozenset({0})]
        seen_values = np.array([3.0, 3.0])  # zero variance
        mu_u, sigma_u = 3.0, 0.0
        sigma_u = 2.0 if sigma_u < 1e-9 else sigma_u  # mirrors _facet_tail's fallback
        out = _genre_block(frozenset({0}), seen_genre_ids, seen_values, mu_u, sigma_u)
        self.assertAlmostEqual(out["user_genre_0_z"], 0.0)  # (3.0 - 3.0) / 2.0

    def test_theme_block_weights_by_similarity_and_masks_via_mass_log(self):
        """3 themes: 0 and 1 are identical (cos=1), 2 is unrelated (cos=0).
        Target has theme 0; one seen film has theme 1 (should count fully,
        since w=1), another has theme 2 (should contribute nothing)."""
        theme_matrix = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        target_theme_ids = frozenset({0})
        seen_theme_ids = [frozenset({1}), frozenset({2})]
        seen_values = np.array([4.0, 1.0])
        out = _theme_block(target_theme_ids, seen_theme_ids, seen_values, theme_matrix)
        self.assertAlmostEqual(out["user_theme_avg"], 4.0)  # only the w=1 film counts
        self.assertAlmostEqual(out["user_theme_mass_log"], np.log1p(1.0))
        self.assertAlmostEqual(out["user_theme_simcnt_hi"], np.log1p(1))  # w=1 > 0.8

    def test_theme_block_no_target_themes_is_nan_and_zero_mass(self):
        out = _theme_block(frozenset(), [frozenset({0})], np.array([4.0]),
                           np.zeros((1, 1), dtype=np.float32))
        self.assertTrue(np.isnan(out["user_theme_avg"]))
        self.assertEqual(out["user_theme_mass_log"], 0.0)  # exact mask for the NaN average
        self.assertEqual(out["user_theme_simcnt_hi"], 0.0)

    def test_actor_block_rankings_and_padding(self):
        """Actor A appears in 2 seen films (mean 5.0), actor B in 1 (mean
        3.0); both are in the target's cast. by-rating puts A first (higher
        mean), by-count also puts A first (2 vs 1 viewing). Cast overlap:
        the 2-shared-actor film ranks above the 1-shared-actor film."""
        target_cast = frozenset({"A", "B"})
        seen_cast_list = [frozenset({"A", "B"}), frozenset({"A"}), frozenset({"C"})]
        seen_values = np.array([5.0, 5.0, 1.0])
        mu_u, sigma_u = float(seen_values.mean()), float(seen_values.std(ddof=0))
        out = _actor_block(target_cast, seen_cast_list, seen_values, mu_u, sigma_u)
        self.assertAlmostEqual(out["user_actor_byrating1_z"], (5.0 - mu_u) / sigma_u)  # A
        self.assertAlmostEqual(out["user_actor_byrating1_cnt"], np.log1p(2))
        self.assertAlmostEqual(out["user_actor_byrating2_z"], (5.0 - mu_u) / sigma_u)  # B (tie on avg, fewer cnt)
        self.assertAlmostEqual(out["user_actor_byrating2_cnt"], np.log1p(1))
        self.assertAlmostEqual(out["user_actor_bycount1_cnt"], np.log1p(2))  # A ranks first by count too
        # Only 2 actors matched -> ranks 3-5 are padded.
        self.assertTrue(np.isnan(out["user_actor_byrating3_z"]))
        self.assertEqual(out["user_actor_byrating3_cnt"], 0.0)
        # Cast overlap: film 0 shares 2 actors, film 1 shares 1, film 2 shares 0 (excluded).
        self.assertAlmostEqual(out["user_cast_overlap1_n"], np.log1p(2))
        self.assertAlmostEqual(out["user_cast_overlap1_rating"], 5.0)
        self.assertAlmostEqual(out["user_cast_overlap2_n"], np.log1p(1))
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
        """The count-is-the-mask invariant: for every NaN-able column,
        isnan(value) iff its companion count is 0 (or, for genre, cnt<=log1p(1)
        when the target itself supplies the +1)."""
        out = _facet_tail(
            frozenset(), frozenset(), frozenset(), frozenset(),
            float("nan"), float("nan"), 0, 0,
            [], [], [], [], np.array([]), np.zeros((1, 1), dtype=np.float32), global_std=1.0)
        for g in range(GENRE_WIDTH):
            self.assertEqual(out[f"user_genre_{g}_cnt"], 0.0)
            self.assertTrue(np.isnan(out[f"user_genre_{g}_z"]))
        self.assertTrue(np.isnan(out["user_theme_avg"]))
        self.assertEqual(out["user_theme_mass_log"], 0.0)
        for i in range(1, 6):
            self.assertEqual(out[f"user_actor_byrating{i}_cnt"], 0.0)
            self.assertTrue(np.isnan(out[f"user_actor_byrating{i}_z"]))
            self.assertEqual(out[f"user_cast_overlap{i}_n"], 0.0)
            self.assertTrue(np.isnan(out[f"user_cast_overlap{i}_rating"]))
        self.assertEqual(out["user_director_cnt"], 0.0)
        self.assertTrue(np.isnan(out["user_director_z"]))

    def test_facet_context_positions_align_with_movie_index(self):
        """A tiny synthetic FacetContext, to confirm facet_tail_from_context
        correctly re-indexes by Split position (not movie_id)."""
        theme_matrix = np.zeros((1, 1), dtype=np.float32)
        fc = FacetContext(
            genre_ids=[frozenset({0}), frozenset({1})],
            theme_ids=[frozenset(), frozenset()],
            actor_sets=[frozenset(), frozenset()],
            director_sets=[frozenset(), frozenset()],
            runtime_log=np.array([4.5, 4.6]), gs_rating=np.array([3.5, 3.0]),
            n_themes=np.array([2.0, 1.0]), n_languages=np.array([1.0, 1.0]),
            theme_matrix=theme_matrix, global_std=1.0)
        # seen_cols=[1] (genre 1, no overlap with target's genre 0) -> NaN z,
        # but the count still reflects the target's OWN genre membership.
        out = facet_tail_from_context(fc, np.array([1]), np.array([2.0]), target_col=0)
        self.assertTrue(np.isnan(out["user_genre_0_z"]))
        self.assertAlmostEqual(out["user_genre_0_cnt"], np.log1p(1))  # target is genre 0, 0 seen
        # genre 1: one seen film (position 1) is a hit, but the target isn't
        # genre 1, so cnt is the seen-hit count alone (no +1).
        self.assertAlmostEqual(out["user_genre_1_z"], 0.0)  # (2.0 - mu_u=2.0) / sigma
        self.assertAlmostEqual(out["user_genre_1_cnt"], np.log1p(1))


if __name__ == "__main__":
    unittest.main()
