import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotten_tomatoes.features import (FACET_TAIL_COLS, MAIN_COLS, FEATURE_COLS,
                                      FacetContext, _facet_tail, facet_tail_from_context,
                                      main_feature_row)
from rotten_tomatoes import movie_features as MF


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
        # Facet tail with every facet absent (target has no facets at all) --
        # every dev/cnt must default to 0, every multi-hot to all-zero.
        tail.update(_facet_tail({}, [], [], float("nan"), float("nan"), 0, 0, 0, [], np.array([])))

        row = main_feature_row(
            np.array([0.5, -0.2]), np.array([2.0, 1.0]), tail)

        # main_feature_row's contract is the column *set* (any downstream
        # pd.DataFrame(rows, columns=FEATURE_COLS) reindexes explicitly), not
        # dict insertion order -- the facet tail is built as dev-then-cnt
        # groups, not interleaved per-facet.
        self.assertEqual(set(row), set(FEATURE_COLS))
        self.assertEqual(len(row), len(FEATURE_COLS))
        self.assertAlmostEqual(row[MAIN_COLS[0]], 1.0)
        self.assertEqual(row[MAIN_COLS[10]], 1.0)
        self.assertEqual(row["genre_id"], 2)
        self.assertEqual(row["user_genre_dev"], 0.0)
        self.assertEqual(row["runtime_log"], 0.0)  # NaN target numeric -> neutral 0

    def test_facet_affinity_is_leave_target_out_and_correctly_averaged(self):
        """Two seen films share the target's genre (Drama), one doesn't
        (Comedy-only); the affinity dev must average only the two hits, and
        the target movie itself is never consulted for the user's own facets
        (only seen films are), so there is no leakage."""
        target_facets = {"genre": frozenset({"Drama", "History"})}
        seen_facets_list = [
            {"genre": frozenset({"Drama"})},        # hit
            {"genre": frozenset({"Comedy"})},        # miss
            {"genre": frozenset({"History", "War"})},  # hit
        ]
        seen_devs = np.array([1.0, -5.0, 3.0])  # the miss's deviation must be ignored
        out = _facet_tail(target_facets, [], [], float("nan"), float("nan"), 0, 0, 0,
                          seen_facets_list, seen_devs)
        self.assertAlmostEqual(out["user_genre_dev"], (1.0 + 3.0) / 2)
        self.assertAlmostEqual(out["user_genre_cnt"], np.log1p(2))

    def test_facet_context_positions_align_with_movie_index(self):
        """build via a tiny synthetic MovieFacets stand-in to confirm
        facet_tail_from_context correctly re-indexes by Split position."""
        fc = FacetContext(
            facet_sets=[{"genre": frozenset({"Drama"})}, {"genre": frozenset({"Comedy"})}],
            genre_mh=[[0], [1]], decade_mh=[[0], [0]],
            runtime_log=np.array([4.5, 4.6]), gs_rating=np.array([3.5, 3.0]),
            n_themes=np.array([2.0, 1.0]), n_languages=np.array([1.0, 1.0]),
            n_countries=np.array([1.0, 1.0]))
        # seen_cols=[1] (Comedy, no overlap with target's Drama) -> zero dev
        out = facet_tail_from_context(fc, np.array([1]), np.array([2.0]), target_col=0)
        self.assertEqual(out["user_genre_dev"], 0.0)
        self.assertEqual(out["mh_genre_0"], 1.0)
        self.assertEqual(out["mh_genre_1"], 0.0)


if __name__ == "__main__":
    unittest.main()