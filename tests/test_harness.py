import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Design 1 owns the analytic formula; the pseudo-user substrate is identical in
# every design folder, so testing one copy tests them all.
from rotten_tomatoes.design1_analytic.pseudo_users import (Split, iter_paired_episodes,
                                           partition_pseudo_users,
                                           sample_random_holdout, similarity)
from rotten_tomatoes.design1_analytic.analytic import predict_movies


def make_split() -> Split:
    history = sparse.csc_matrix([
        [2.5, 5.0],
        [2.0, 4.0],
        [4.0, 2.0],
    ])
    targets = sparse.csr_matrix([
        [5.0],
        [4.0],
        [2.0],
    ])
    history_mask = history.copy()
    history_mask.data[:] = 1.0
    target_mask = targets.copy()
    target_mask.data[:] = 1.0
    return Split(
        critic_index=pd.Index(["user", "aligned", "inverse"]),
        tgt_movie_index=pd.Index(["target"]),
        H=history,
        Hmask=history_mask,
        T=targets,
        Tmask=target_mask,
        TT=targets.T.tocsr(),
        TTmask=target_mask.T.tocsr(),
    )


class MovieMeanCenteredPredictionTests(unittest.TestCase):
    def test_similarity_reports_magnitude_scale(self):
        split = make_split()

        sim, overlap, mag_sim = similarity(
            split, 0, np.array([0, 1]), np.array([2.5, 5.0]))

        self.assertEqual(overlap[1], 2)
        self.assertAlmostEqual(sim[1], 1.0)
        self.assertAlmostEqual(mag_sim[1], 1.25)

    def test_prediction_uses_movie_mean_and_magnitude_factor(self):
        split = make_split()
        sim = np.array([0.0, 1.0, -0.5])
        mag_sim = np.array([1.0, 1.25, 1.0])

        pred, den, movie_mean, reviewer_count = predict_movies(
            split, 0, sim, mag_sim, np.array([0]))

        self.assertEqual(reviewer_count[0], 2)
        self.assertAlmostEqual(movie_mean[0], 3.0)
        self.assertAlmostEqual(den[0], 1.5)
        expected = ((1.0 * 3.0 + 1.0 * (4.0 - 3.0)) * 1.25
                    + (0.5 * 3.0 - 0.5 * (2.0 - 3.0))) / 1.5
        self.assertAlmostEqual(pred[0], expected)

    def test_random_holdout_never_uses_target_as_seen_history(self):
        split = make_split()
        split.user_hist[0] = (np.array([0, 1, 2, 3]),
                              np.array([1.0, 2.0, 3.0, 4.0]))
        split.pop_weight = {0: 1, 1: 2, 2: 3, 3: 4}

        episode = sample_random_holdout(
            np.random.default_rng(42), split, 0, n=2,
            target_ok=np.array([True, True, True, True]))

        self.assertIsNotNone(episode)
        seen_cols, seen_values, target_col, target_value = episode
        self.assertEqual(len(seen_cols), 2)
        self.assertNotIn(target_col, seen_cols)
        self.assertEqual(target_value, split.user_hist[0][1][target_col])
        self.assertEqual(len(seen_values), len(seen_cols))

    def test_paired_episodes_are_nested_and_deterministic(self):
        import numpy as np
        from scipy import sparse
        # one pseudo-user with 60 movies (> N_MAX_FINITE) reviewed by 3 critics
        n_movies = 60
        rng = np.random.default_rng(0)
        data = rng.integers(0, 6, size=(3, n_movies)).astype(float)
        H = sparse.csc_matrix(data)
        Hmask = H.copy(); Hmask.data[:] = 1.0
        T, Tmask = H.tocsr(), Hmask.tocsr()
        split = Split(
            critic_index=pd.Index([f"c{i}" for i in range(3)]),
            tgt_movie_index=pd.Index(range(n_movies)),
            H=H, Hmask=Hmask, T=T, Tmask=Tmask,
            TT=T.T.tocsr(), TTmask=Tmask.T.tocsr())
        split.users = [0]
        split.user_hist = {0: (np.arange(n_movies), data[0])}
        split.n_reviewers = np.full(n_movies, 3)
        split.pop_weight = {c: 1 for c in range(n_movies)}

        episodes = list(iter_paired_episodes(split, [0], seed=1))
        again = list(iter_paired_episodes(split, [0], seed=1))
        self.assertEqual([e[:5] for e in episodes], [e[:5] for e in again])

        by_key = {}
        for upos, tcol, tval, n, draw, seen_cols, _ in episodes:
            by_key.setdefault((tcol, draw), {})[n] = list(seen_cols)
            self.assertNotIn(tcol, seen_cols)   # target never in the seen set
        for key, seen_by_n in by_key.items():
            self.assertEqual(sorted(seen_by_n), [-1, 3, 5, 10, 20, 50])
            # nested: each finite seen set is a prefix of the next
            for small, big in [(3, 5), (5, 10), (10, 20), (20, 50)]:
                self.assertEqual(seen_by_n[small], seen_by_n[big][:small])

    def test_pseudo_user_partitions_are_disjoint_and_complete(self):
        split = make_split()
        split.users = [0, 1, 2]

        partitions = partition_pseudo_users(
            split, seed=42, validation_fraction=1 / 3, test_fraction=1 / 3)

        assigned = np.concatenate(list(partitions.values()))
        self.assertEqual(set(assigned), {0, 1, 2})
        self.assertEqual(len(assigned), len(set(assigned)))


if __name__ == "__main__":
    unittest.main()