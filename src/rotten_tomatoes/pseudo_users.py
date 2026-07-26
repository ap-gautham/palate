"""Pseudo-user substrate: the critic-by-movie matrix, Pearson+magnitude
similarity, held-out-episode sampling, and the deterministic paired/nested
test protocol.

This file lives once at the package root and is imported by all three design
trainers (``train_analytic.py``, ``train_xgboost.py``, ``train_neural.py``), so
the seeded episode generator produces byte-identical (user, target, n, draw)
test episodes across designs -- which is what makes the cross-design comparison
in ``analyze.py`` valid. Only the shared
constants in ``config.py`` (paths, seeds, the evaluation grid) are imported.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from rotten_tomatoes.config import (EVAL_DRAWS, EVAL_TARGETS_PER_USER, MIN_HISTORY,
                    MIN_OTHER_REVIEWERS, N_GRID, N_MAX_FINITE, SEED, VALUE_COL)


@dataclass
class Split:
    """The all-time matrix plus per-critic statistics for one experiment."""
    critic_index: pd.Index          # pool critics (rows of every matrix)
    tgt_movie_index: pd.Index       # movie columns (targets share the all-time matrix)
    H: sparse.csc_matrix            # critic x movie raw scores (column slicing)
    Hmask: sparse.csc_matrix        # 1.0 where a rating exists
    T: sparse.csr_matrix            # same matrix, row-oriented for target lookups
    Tmask: sparse.csr_matrix
    TT: sparse.csr_matrix           # T transposed (movie x critic), for matvecs
    TTmask: sparse.csr_matrix
    users: list = field(default_factory=list)   # eligible pseudo-user critic positions
    user_hist: dict = field(default_factory=dict)   # pos -> (movie col idx, raw scores)
    pop_weight: dict = field(default_factory=dict)  # movie col idx -> popularity weight
    tm_z: np.ndarray | None = None  # per target movie: Tomatometer mapped to value scale
    dispersion: np.ndarray | None = None  # per target movie: std of pool values
    n_reviewers: np.ndarray | None = None
    global_mean: float = 0.0        # all-time grand mean (B1)
    global_std: float = 0.0         # all-time grand std -- sigma_u fallback when a
                                     # user's own seen-set std is ~0 (see features.py)
    critic_mean: np.ndarray | None = None  # per pool critic all-time mean
    critic_sum: np.ndarray | None = None
    critic_count: np.ndarray | None = None


def _pivot(df: pd.DataFrame, critic_index: pd.Index, movie_index: pd.Index,
          value_col: str = VALUE_COL):
    ci = pd.Categorical(df["critic_id"], categories=critic_index).codes
    mi = pd.Categorical(df["movie_id"], categories=movie_index).codes
    ok = (ci >= 0) & (mi >= 0)
    vals = sparse.csr_matrix(
        (df[value_col].to_numpy()[ok], (ci[ok], mi[ok])),
        shape=(len(critic_index), len(movie_index)),
    )
    vals.sum_duplicates()
    mask = vals.copy()
    mask.data[:] = 1.0
    # a critic reviewing the same movie twice would sum; divide back by count
    counts = sparse.csr_matrix(
        (np.ones(ok.sum()), (ci[ok], mi[ok])),
        shape=vals.shape,
    )
    counts.sum_duplicates()
    vals.data = vals.data / counts.data
    return vals, mask


def build_split(scored: pd.DataFrame, movies: pd.DataFrame,
                value_col: str = VALUE_COL) -> Split:
    """Build the all-time matrix for random seen-movie / unseen-target episodes.

    Every critic with at least ``MIN_HISTORY`` scored movies is a pseudo-user.
    H and T are the same all-time ratings matrix: episode sampling keeps the
    target movie out of the user's seen set before similarity is computed, so
    any unrated movie is a valid prediction target regardless of release year.

    ``value_col`` selects which numeric column populates the matrix -- the raw
    standardized score (default) or the critic's all-time z-score. Building two
    Splits with the same ``scored``/``movies`` input but different ``value_col``
    yields identical critic_index/movie_index (and therefore identical row/
    column positions), which is what lets the z-score track re-use a Split
    built in critic-z units for peer similarity while keeping the user's own
    seen ratings and ground truth on the raw scale (see design*/train.py
    ``--target z``).
    """
    counts = scored.groupby("critic_id").size()
    pool = counts[counts >= MIN_HISTORY].index
    ratings = scored[scored["critic_id"].isin(pool)]

    critic_index = pd.Index(sorted(pool))
    movie_index = pd.Index(sorted(ratings["movie_id"].unique()))
    H, Hmask = _pivot(ratings, critic_index, movie_index, value_col)
    T, Tmask = H.tocsr(), Hmask.tocsr()
    sp = Split(critic_index=critic_index,
               tgt_movie_index=movie_index,
               H=H.tocsc(), Hmask=Hmask.tocsc(),
               T=T, Tmask=Tmask,
               TT=T.T.tocsr(), TTmask=Tmask.T.tocsr())

    H_rows = sp.H.tocsr()
    for upos in range(len(critic_index)):
        lo, hi = H_rows.indptr[upos], H_rows.indptr[upos + 1]
        cols = H_rows.indices[lo:hi].copy()
        values = H_rows.data[lo:hi].copy()
        if len(cols) < MIN_HISTORY:
            continue
        sp.users.append(upos)
        sp.user_hist[upos] = (cols, values)

    popularity = np.asarray(sp.Hmask.sum(axis=0)).ravel()
    sp.pop_weight = {col: int(count) for col, count in enumerate(popularity)}
    sp.global_mean = float(ratings[value_col].mean())
    sp.global_std = float(ratings[value_col].std(ddof=0))

    critic_sums = np.asarray(sp.H.sum(axis=1)).ravel()
    critic_counts = np.asarray(sp.Hmask.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        sp.critic_mean = np.where(critic_counts > 0,
                                  critic_sums / np.maximum(critic_counts, 1),
                                  sp.global_mean)
    sp.critic_sum = critic_sums
    sp.critic_count = critic_counts

    movie_meta = movies.drop_duplicates("movie_id").set_index("movie_id")
    tomatometer = movie_meta["tomatoMeter"]
    movie_mean = ratings.groupby("movie_id")[value_col].agg(["mean", "size"])
    fit_df = movie_mean[movie_mean["size"] >= 5].join(tomatometer, how="inner")
    fit_df = fit_df.dropna(subset=["tomatoMeter"])
    slope, intercept = np.polyfit(fit_df["tomatoMeter"].astype(float),
                                  fit_df["mean"], 1)
    sp.tm_z = (slope * tomatometer.reindex(movie_index).astype(float)
               + intercept).to_numpy()

    sums = np.asarray(sp.TT.sum(axis=1)).ravel()
    reviewer_counts = np.asarray(sp.TTmask.sum(axis=1)).ravel()
    squared = sp.TT.copy()
    squared.data = squared.data ** 2
    sum_squares = np.asarray(squared.sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = ((sum_squares - sums ** 2 / np.maximum(reviewer_counts, 1))
                    / np.maximum(reviewer_counts - 1, 1))
    sp.dispersion = np.sqrt(np.maximum(variance, 0))
    sp.n_reviewers = reviewer_counts
    return sp


def similarity(sp: Split, upos: int, cols: np.ndarray, x: np.ndarray):
    """Pearson correlation (on the overlapping raw 0-5 scores) of the user
    history (cols, x) against every pool critic. Pearson is invariant to each
    critic's location and scale, so it measures taste co-movement regardless of
    harsh/generous level.

    Also returns mag_sim, the least-squares through-origin multiplier from a
    critic's ratings to the user's on that overlap (user = 1.25 * critic gives
    mag_sim = 1.25). Returns (r, overlap_count, mag_sim); self excluded.
    """
    Hs = sp.H[:, cols]
    Ms = sp.Hmask[:, cols]
    cnt = np.asarray(Ms.sum(axis=1)).ravel()
    sy = np.asarray(Hs.sum(axis=1)).ravel()
    Hs2 = Hs.copy(); Hs2.data = Hs2.data ** 2
    syy = np.asarray(Hs2.sum(axis=1)).ravel()
    sx = Ms @ x
    sxx = Ms @ (x ** 2)
    sxy = np.asarray(Hs @ x).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / np.maximum(cnt, 1)
        varx = sxx - sx ** 2 / np.maximum(cnt, 1)
        vary = syy - sy ** 2 / np.maximum(cnt, 1)
        r = cov / np.sqrt(varx * vary)
    mag_sim = np.ones_like(r)
    np.divide(sxy, syy, out=mag_sim, where=(cnt >= 2) & (syy > 1e-12))
    r[(cnt < 2) | ~np.isfinite(r)] = 0.0
    mag_sim[~np.isfinite(mag_sim)] = 1.0
    r[upos] = 0.0
    cnt[upos] = 0
    mag_sim[upos] = 1.0
    return r, cnt, mag_sim


def sample_random_holdout(rng: np.random.Generator, sp: Split, upos: int,
                          n: int | None, target_ok: np.ndarray | None = None,
                          weighted: bool = True):
    """Sample seen ratings and a distinct held-out target for one fake user.

    For finite ``n``, seen movies are sampled first and the target is chosen
    uniformly from the remaining eligible ratings. ``n=None`` means every rating
    except one target is seen. Returns
    ``(seen_cols, seen_values, target_col, target_value)`` or ``None``.
    """
    cols, values = sp.user_hist[upos]
    if len(cols) < 2:
        return None

    eligible = np.ones(len(cols), dtype=bool)
    if target_ok is not None:
        eligible &= target_ok[cols]
    if not eligible.any():
        return None

    if n is None:
        target_idx = int(rng.choice(np.flatnonzero(eligible)))
        keep = np.ones(len(cols), dtype=bool)
        keep[target_idx] = False
        return cols[keep], values[keep], int(cols[target_idx]), float(values[target_idx])

    if n <= 0 or n >= len(cols):
        return None

    if weighted:
        weights = np.array([sp.pop_weight.get(col, 1) for col in cols], dtype=float)
        probabilities = weights / weights.sum()
    else:
        probabilities = None

    # Sample seen movies first (matches the product workflow); retry when that
    # sample happens to consume every eligible target for this pseudo-user.
    for _ in range(20):
        seen_idx = rng.choice(len(cols), size=n, replace=False, p=probabilities)
        available = eligible.copy()
        available[seen_idx] = False
        if not available.any():
            continue
        target_idx = int(rng.choice(np.flatnonzero(available)))
        return (cols[seen_idx], values[seen_idx], int(cols[target_idx]),
                float(values[target_idx]))
    return None


def target_ok_mask(sp: Split, upos: int) -> np.ndarray:
    """Per movie: True where at least MIN_OTHER_REVIEWERS critics besides the
    pseudo-user reviewed it (so a held-out target has real evidence)."""
    own = np.asarray(sp.Tmask[upos].toarray()).ravel()
    return (sp.n_reviewers - own) >= MIN_OTHER_REVIEWERS


def iter_paired_episodes_for_user(sp: Split, upos: int, seed: int,
                                  n_targets: int = EVAL_TARGETS_PER_USER,
                                  n_draws: int = EVAL_DRAWS):
    """`iter_paired_episodes`'s body for a single pseudo-user. Factored out so
    the per-user work can be split across processes (see `generate_paired_rows`
    in features.py): each user gets its own independent RNG stream, seeded from
    ``(seed, upos)`` rather than by consuming a single stream in iteration
    order, so the result is identical regardless of how users are chunked or
    in what order they're processed."""
    rng = np.random.default_rng([seed, upos])
    cols, vals = sp.user_hist[upos]
    if len(cols) <= N_MAX_FINITE:            # cannot support the whole grid
        return
    ok = target_ok_mask(sp, upos)
    eligible_local = np.flatnonzero(ok[cols])
    if len(eligible_local) == 0:
        return
    chosen = rng.choice(eligible_local,
                        size=min(n_targets, len(eligible_local)),
                        replace=False)
    for t_local in chosen:
        target_col = int(cols[t_local])
        target_value = float(vals[t_local])
        keep = np.ones(len(cols), dtype=bool)
        keep[t_local] = False
        rem_cols, rem_vals = cols[keep], vals[keep]
        weights = np.array([sp.pop_weight.get(int(c), 1) for c in rem_cols],
                           dtype=float)
        probs = weights / weights.sum()
        for draw in range(n_draws):
            order = rng.choice(len(rem_cols), size=len(rem_cols),
                               replace=False, p=probs)
            for n in N_GRID:
                seen = order if n is None else order[:n]
                yield (upos, target_col, target_value,
                       -1 if n is None else n, draw,
                       rem_cols[seen], rem_vals[seen])


def iter_paired_episodes(sp: Split, users, seed: int = SEED,
                         n_targets: int = EVAL_TARGETS_PER_USER,
                         n_draws: int = EVAL_DRAWS):
    """Deterministically yield PAIRED, NESTED held-out episodes.

    For each pseudo-user with more than ``N_MAX_FINITE`` movies, a fixed set of
    target movies is chosen once. For each target and each redraw, the remaining
    movies are placed in one popularity-weighted random order, and the seen set
    at seen-count ``n`` is the first ``n`` of that order (all of them for
    ``n = all``). Because the target and seen ORDER are fixed across the n-grid,
    every seen-count -- and every baseline -- is scored on an identical
    (user, target, draw) set, removing the target-sampling wobble.

    Each user draws from its own independent RNG stream (see
    `iter_paired_episodes_for_user`), so users can be processed in any order --
    or split across worker processes -- with an identical result.

    Yields ``(upos, target_col, target_value, n, draw, seen_cols, seen_values)``.
    """
    for upos in sorted(int(u) for u in users):
        yield from iter_paired_episodes_for_user(sp, upos, seed, n_targets, n_draws)


def partition_pseudo_users(sp: Split, seed: int = SEED,
                           validation_fraction: float = 0.15,
                           test_fraction: float = 0.15) -> dict[str, np.ndarray]:
    """Return deterministic, critic-disjoint train/validation/test positions."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave training users")

    users = np.asarray(sp.users, dtype=int).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_validation = max(1, int(round(len(users) * validation_fraction)))
    n_test = max(1, int(round(len(users) * test_fraction)))
    if n_validation + n_test >= len(users):
        raise ValueError("not enough pseudo-users for three partitions")
    return {
        "train": users[n_validation + n_test:],
        "validation": users[:n_validation],
        "test": users[n_validation:n_validation + n_test],
    }


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))
