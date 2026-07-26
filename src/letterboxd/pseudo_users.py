"""Member pseudo-user substrate: the member-by-film sparse matrix, the shrunk
Pearson+magnitude similarity, the deterministic train/validation/test member
partition, and the nested paired-episode protocol.

This is the Letterboxd counterpart of ``rotten_tomatoes/pseudo_users.py`` --
same role, same boundary against ``features.py`` (which owns the feature
contract and the row builders on top of this substrate). It lives once at the
package root so every design trainer draws byte-identical episodes, which is
what makes the cross-design comparison in ``analyze.py`` valid.

Self-contained: never imports Rotten Tomatoes code, and imports nothing from
this package except ``config.py`` (paths, seeds, scales) -- so ``features.py``
can build on it without an import cycle. ``make_genre_maps`` lives here rather
than in ``features.py`` (where Rotten Tomatoes keeps it) because it feeds
``LBData.genre_id``, i.e. the matrix itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .config import SEED

MIN_APP_OVERLAP = 2            # min shared films before a member similarity counts
K_SHRINK = 8                   # overlap shrinkage, same constant as RT


def rmse(err) -> float:
    """RMSE of an error vector (prediction - truth)."""
    err = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


# ---- genre map -------------------------------------------------------------
def _first_genre(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except (json.JSONDecodeError, ValueError):
        return raw.split(",")[0].strip().strip("[]\"' ")
    return ""


def make_genre_maps(movies: pd.DataFrame):
    """movie_id -> genre_id, plus the genre->id table and the unknown id."""
    genres = (movies.drop_duplicates("movie_id").set_index("movie_id")["genres"]
              .map(_first_genre))
    genre_to_id = {g: i for i, g in enumerate(sorted(genres.unique()))}
    unknown_genre_id = len(genre_to_id)
    genre_to_id["__unknown__"] = unknown_genre_id
    movie_to_genre = {mid: genre_to_id[g] for mid, g in genres.items()}
    return movie_to_genre, genre_to_id, unknown_genre_id


# ---- sparse-matrix substrate (members x films) -----------------------------
@dataclass
class LBData:
    mat_csc: sparse.csc_matrix       # ratings, members x films
    mask_csc: sparse.csc_matrix      # 1 where rated
    members: pd.Index                # position -> member_id
    movies: pd.Index                 # position -> movie_id
    member_sum: np.ndarray           # all-time rating sum per member
    member_count: np.ndarray         # all-time rating count per member
    movie_mean: np.ndarray
    movie_std: np.ndarray            # ddof=0 dispersion per film
    movie_count: np.ndarray
    genre_id: np.ndarray             # per movie position
    global_mean: float
    global_std: float                # all-time grand std -- sigma_u fallback when a
                                      # user's own seen-set std is ~0 (see features.py)
    unknown_genre_id: int
    # member position -> (film col idx, ratings), extracted once at build time
    # so per-member episode sampling never re-converts the whole CSC matrix
    # (mirrors rotten_tomatoes Split.user_hist).
    member_hist: dict = field(default_factory=dict)

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def n_movies(self) -> int:
        return len(self.movies)


def build_data(ratings: pd.DataFrame, movies: pd.DataFrame, value: str = "raw") -> LBData:
    """``value="z"`` builds the PEER matrix from each member's all-time z-score
    instead of their raw rating -- members are indexed identically either way,
    so a raw-built and a z-built LBData share row/column positions (see
    rotten_tomatoes.pseudo_users.build_split's docstring for why this matters:
    it's what lets the z-score track reuse a peer matrix built in z-units
    while the user's own seen ratings are standardized per-episode instead,
    see episode_feature_row_z). Members with ~zero all-time rating variance
    get z=0 (a neutral, information-free peer contribution) rather than being
    dropped, since LBData's row/column set must stay identical across both
    builds.
    """
    members = pd.Index(ratings.user_id.drop_duplicates())
    movie_ids = pd.Index(ratings.movie_id.drop_duplicates())
    ui = pd.Categorical(ratings.user_id, categories=members).codes
    mi = pd.Categorical(ratings.movie_id, categories=movie_ids).codes
    raw_values = ratings.rating.to_numpy(float)
    if value == "z":
        member_sum_raw = np.bincount(ui, weights=raw_values, minlength=len(members))
        member_count_raw = np.bincount(ui, minlength=len(members)).astype(float)
        member_mu_raw = member_sum_raw / np.maximum(member_count_raw, 1)
        member_sqsum_raw = np.bincount(ui, weights=raw_values ** 2, minlength=len(members))
        member_var_raw = np.maximum(
            member_sqsum_raw / np.maximum(member_count_raw, 1) - member_mu_raw ** 2, 0)
        member_sigma_raw = np.sqrt(member_var_raw)
        safe_sigma = np.where(member_sigma_raw > 1e-9, member_sigma_raw, 1.0)
        values = np.where(member_sigma_raw[ui] > 1e-9,
                          (raw_values - member_mu_raw[ui]) / safe_sigma[ui], 0.0)
    else:
        values = raw_values
    mat = sparse.csr_matrix((values, (ui, mi)), shape=(len(members), len(movie_ids)))
    mask = mat.copy()
    mask.data[:] = 1.0

    member_sum = np.asarray(mat.sum(axis=1)).ravel()
    member_count = np.asarray(mask.sum(axis=1)).ravel()
    movie_sum = np.asarray(mat.sum(axis=0)).ravel()
    movie_count = np.asarray(mask.sum(axis=0)).ravel()
    movie_mean = np.divide(movie_sum, movie_count, out=np.zeros_like(movie_sum),
                           where=movie_count > 0)
    sq = mat.copy(); sq.data **= 2
    movie_sq = np.asarray(sq.sum(axis=0)).ravel()
    movie_var = np.maximum(movie_sq / np.maximum(movie_count, 1) - movie_mean ** 2, 0)
    movie_std = np.sqrt(movie_var)

    movie_to_genre, _, unknown_genre_id = make_genre_maps(movies)
    genre_id = np.array([movie_to_genre.get(mid, unknown_genre_id) for mid in movie_ids],
                        dtype=np.int64)

    # mat is still CSR here: extract every member's (films, ratings) row once,
    # so per-member sampling downstream is a dict lookup instead of a full
    # CSC->CSR conversion per member (mirrors rotten_tomatoes Split.user_hist).
    member_hist = {m: (mat.indices[mat.indptr[m]:mat.indptr[m + 1]].copy(),
                       mat.data[mat.indptr[m]:mat.indptr[m + 1]].copy())
                   for m in range(len(members))}
    return LBData(mat.tocsc(), mask.tocsc(), members, movie_ids, member_sum,
                  member_count, movie_mean, movie_std, movie_count, genre_id,
                  float(values.mean()), float(values.std(ddof=0)), unknown_genre_id,
                  member_hist)


def similarity(data: LBData, seen_cols: np.ndarray, seen_vals: np.ndarray,
               exclude_member: int | None):
    """Shrunk Pearson alignment + magnitude of every member to this profile.

    Returns (sim[M], mag[M], overlap[M]). Vectorised over all members via the
    sparse film submatrix, the same formula as the RT model."""
    x = np.asarray(seen_vals, dtype=float)
    peers = data.mat_csc[:, seen_cols]
    peer_mask = data.mask_csc[:, seen_cols]
    overlap = np.asarray(peer_mask.sum(axis=1)).ravel()
    peer_values = np.asarray(peers.sum(axis=1)).ravel()
    peers_sq = peers.copy(); peers_sq.data **= 2
    peer_sq_values = np.asarray(peers_sq.sum(axis=1)).ravel()
    sx = np.asarray(peer_mask @ x).ravel()
    sxx = np.asarray(peer_mask @ (x ** 2)).ravel()
    sxy = np.asarray(peers @ x).ravel()
    denom_overlap = np.maximum(overlap, 1)
    numer = sxy - sx * peer_values / denom_overlap
    var_x = sxx - sx ** 2 / denom_overlap
    var_peer = peer_sq_values - peer_values ** 2 / denom_overlap
    denom = np.sqrt(np.maximum(var_x * var_peer, 0))
    sim = np.divide(numer, denom, out=np.zeros_like(numer),
                    where=(overlap >= MIN_APP_OVERLAP) & (denom > 1e-12))
    sim *= np.minimum(overlap, K_SHRINK) / K_SHRINK
    mag = np.divide(sxy, peer_sq_values, out=np.ones_like(sxy), where=peer_sq_values > 1e-12)
    if exclude_member is not None:
        sim[exclude_member] = 0.0
    return sim, mag, overlap


# ---- member partition ------------------------------------------------------
def eligible_members(data: LBData, min_ratings: int) -> np.ndarray:
    return np.flatnonzero(data.member_count >= min_ratings)


def partition_members(data: LBData, min_ratings: int = 6, seed: int = SEED):
    """Deterministic disjoint train/validation/test member split (70/15/15),
    mirroring the RT pseudo-user partition."""
    perm = np.random.default_rng(seed).permutation(eligible_members(data, min_ratings))
    n_train = int(0.70 * len(perm))
    n_val = int(0.15 * len(perm))
    return {"train": perm[:n_train],
            "validation": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:]}


# ---- paired/nested episode protocol ----------------------------------------
def iter_paired_episodes_for_member(data: LBData, member: int, n_grid,
                                    targets_per_user: int, draws: int,
                                    n_max_finite: int, seed: int = SEED):
    """`iter_paired_episodes`'s body for a single member. Factored out so the
    per-member work can be split across processes (see `generate_paired_rows`
    in features.py): each member gets its own independent RNG stream, seeded
    from ``(seed, member)`` rather than by consuming a single stream in
    iteration order, so the result is identical regardless of how members are
    chunked or in what order they're processed."""
    rng = np.random.default_rng([seed, member])
    popularity = data.movie_count
    films, film_vals = data.member_hist[member]
    if len(films) <= n_max_finite:
        return
    target_local = rng.choice(len(films), size=min(targets_per_user, len(films)),
                              replace=False)
    for t in target_local:
        target_col = int(films[t])
        target_value = float(film_vals[t])
        rest = np.array([j for j in range(len(films)) if j != t])
        weights = popularity[films[rest]].astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else None
        for draw in range(draws):
            order = rng.choice(rest, size=len(rest), replace=False, p=weights)
            for n in n_grid:
                size = len(order) if n is None else n
                if n is not None and size > len(order):
                    continue
                seen_local = order[:size]
                yield (member, target_col, target_value,
                       -1 if n is None else n, draw,
                       films[seen_local], film_vals[seen_local])


def iter_paired_episodes(data: LBData, members: np.ndarray, n_grid,
                         targets_per_user: int, draws: int, n_max_finite: int):
    """Nested paired episodes: fixed targets + fixed popularity-ordered seen
    prefix per (member, draw), so every n is scored on an identical set.

    Each member draws from its own independent RNG stream (see
    `iter_paired_episodes_for_member`), so members can be processed in any
    order -- or split across worker processes -- with an identical result.
    """
    for member in members:
        yield from iter_paired_episodes_for_member(
            data, int(member), n_grid, targets_per_user, draws, n_max_finite)
