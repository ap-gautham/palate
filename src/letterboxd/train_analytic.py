"""Train/evaluate the Letterboxd analytic neighbourhood model on 1--10 data.

The target is a real member's held-out rating.  This is intentionally a
separate implementation and result directory from the Rotten Tomatoes model.
"""
from __future__ import annotations

import json
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

from .config import MODELS, RATING_MAX, RATING_MIN, RATINGS_PARQUET, RESULTS, SEED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-test-users", type=int, default=1_000,
                        help="Deterministic held-out evaluation sample; the full rating matrix is always used.")
    parser.add_argument("--max-seen", type=int, default=50,
                        help="Cap each held-out member profile to control sparse working-set size.")
    args = parser.parse_args()
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")
    ratings = pd.read_parquet(RATINGS_PARQUET)
    users = pd.Index(ratings.user_id.drop_duplicates())
    movies = pd.Index(ratings.movie_id.drop_duplicates())
    ui = pd.Categorical(ratings.user_id, categories=users).codes
    mi = pd.Categorical(ratings.movie_id, categories=movies).codes
    matrix = sparse.csr_matrix((ratings.rating.to_numpy(float), (ui, mi)), shape=(len(users), len(movies)))
    mask = matrix.copy(); mask.data[:] = 1
    rng = np.random.default_rng(SEED)
    eligible_users = np.flatnonzero(np.diff(matrix.indptr) >= 5)
    test_users = rng.choice(eligible_users, size=min(args.max_test_users, len(eligible_users)), replace=False)
    squared_error: list[float] = []
    # A compact evaluation loop; it does not materialize an all-user dense matrix.
    for u in test_users:
        watched = matrix.indices[matrix.indptr[u]:matrix.indptr[u + 1]]
        target = int(rng.choice(watched))
        seen = watched[watched != target]
        if len(seen) > args.max_seen:
            seen = rng.choice(seen, size=args.max_seen, replace=False)
        x = matrix[u, seen].toarray().ravel()
        peers = matrix[:, seen]
        peer_mask = mask[:, seen]
        overlap = np.asarray(peer_mask.sum(axis=1)).ravel()
        peer_values = np.asarray(peers.sum(axis=1)).ravel()
        peers_sq = peers.copy(); peers_sq.data **= 2
        peer_sq_values = np.asarray(peers_sq.sum(axis=1)).ravel()
        sx = np.asarray(peer_mask @ x).ravel()
        sxx = np.asarray(peer_mask @ (x ** 2)).ravel()
        sxy = np.asarray(peers @ x).ravel()
        # Correlation on each member's actual overlap, shrunk for short
        # histories. This follows the same raw-score principle as the RT model.
        numer = sxy - sx * peer_values / np.maximum(overlap, 1)
        var_x = sxx - sx ** 2 / np.maximum(overlap, 1)
        var_peer = peer_sq_values - peer_values ** 2 / np.maximum(overlap, 1)
        denom = np.sqrt(var_x * var_peer)
        sim = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
        sim *= np.minimum(overlap, 8) / 8
        sim[u] = 0
        target_rows = matrix[:, target].toarray().ravel()
        target_mask = mask[:, target].toarray().ravel().astype(bool)
        if not target_mask.any():
            continue
        mean = target_rows[target_mask].mean()
        weights = np.abs(sim) * target_mask
        # Identical movie-mean-centred, signed-similarity and magnitude
        # formula to RT Design 1; only the clipping range differs (1--10).
        mag = np.divide(sxy, peer_sq_values, out=np.ones_like(sxy), where=peer_sq_values > 1e-12)
        numerator = ((np.abs(sim) * mean + sim * (target_rows - mean)) * mag * target_mask).sum()
        pred = mean if weights.sum() == 0 else float(numerator / weights.sum())
        truth = float(matrix[u, target])
        squared_error.append((np.clip(pred, RATING_MIN, RATING_MAX) - truth) ** 2)
    RESULTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    report = {"model": "analytic_neighbourhood", "rating_scale": [RATING_MIN, RATING_MAX],
              "matrix_users": int(len(users)), "matrix_movies": int(len(movies)), "matrix_ratings": int(matrix.nnz),
              "test_episodes": len(squared_error), "rmse": float(np.sqrt(np.mean(squared_error)),),
              "evaluation_sample": int(len(test_users)), "max_seen": args.max_seen, "seed": SEED}
    (RESULTS / "analytic_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
