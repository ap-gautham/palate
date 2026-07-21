"""Letterboxd feature contract, mirroring the Rotten Tomatoes design-2/3 schema.

Per (member profile, target film) episode the learned models see a fixed-width
row: the target's raters sorted by member similarity and summarised in ten
deciles of ``similarity x (rater score - that rater's leave-one-out all-time
mean)``, plus a tail of seen count, overlap statistics, rater count, dispersion,
genre, and the member's mean rating -- and, beyond that, a rich movie-facet
tail (see "rich movie-facet contract" below) built from the gsimonx37/
letterboxd metadata join (`movie_features.py`): per-facet user affinity (does
this member's seen history over- or under-rate films sharing this genre/
theme/studio/director/actor/decade/language/country with the target?) plus a
genre/decade multi-hot and a small numeric tail (runtime, external rating,
facet counts). This is the RT contract **without the Tomatometer feature**
(Letterboxd has no critic-consensus meter), on the 1-10 member scale.

Self-contained: this module never imports Rotten Tomatoes code (the gsimonx37
join is duplicated, not shared, in `rotten_tomatoes/movie_features.py`). It
offers a sparse-matrix path (``build_data`` + ``similarity`` +
``episode_feature_row`` + row generators) used for training and analysis, and
a DataFrame app-path (``app_similarity`` + ``app_features``) used by and
mirrored in the browser TypeScript port.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from . import movie_features as MF
from .config import RATING_MIN, RATING_MAX, SEED

MIN_OTHER_REVIEWERS = 3        # a target film needs this many OTHER raters
MIN_APP_OVERLAP = 2            # min shared films before a member similarity counts
K_SHRINK = 8                   # overlap shrinkage, same constant as RT
DECILES = 10

MAIN_COLS = ([f"d{i}_mean" for i in range(DECILES)]
             + [f"d{i}_cnt" for i in range(DECILES)]
             + [f"d{i}_std" for i in range(DECILES)])

# ---- rich movie-facet contract (see module docstring) ----------------------
FACET_DEV_COLS = [f"user_{f}_dev" for f in MF.FACETS]
FACET_CNT_COLS = [f"user_{f}_cnt" for f in MF.FACETS]
GENRE_MH_COLS = [f"mh_genre_{i}" for i in range(MF.GENRE_VOCAB_K + 1)]
DECADE_MH_COLS = [f"mh_decade_{i}" for i in range(MF.DECADE_VOCAB_K + 1)]
NUMERIC_TAIL_COLS = ["runtime_log", "gs_rating", "n_themes_log", "n_languages_log",
                     "n_countries_log"]
FACET_TAIL_COLS = FACET_DEV_COLS + FACET_CNT_COLS + GENRE_MH_COLS + DECADE_MH_COLS + NUMERIC_TAIL_COLS

TAIL_COLS = (["n_observed", "mean_overlap", "max_overlap", "n_reviewers",
              "dispersion", "genre_id", "user_mean"] + FACET_TAIL_COLS)
FEATURE_COLS = MAIN_COLS + TAIL_COLS
_EMPTY_FACET_SETS = {f: frozenset() for f in MF.FACETS}


def _facet_tail(target_facets: dict, target_genre_ids: list, target_decade_ids: list,
                target_runtime_log: float, target_gs_rating: float,
                target_n_themes: float, target_n_languages: float,
                target_n_countries: float, seen_facets_list: list,
                seen_devs: np.ndarray) -> dict:
    """Core facet-tail computation, agnostic to whether the caller resolved
    facets by sparse-matrix position (offline) or by movie_id (app-time). For
    each facet, the user-affinity pair is the mean deviation (and log1p count)
    over the *seen* films whose facet set intersects the target's -- built
    leave-the-target-out by construction, since only seen films are examined.
    Identical logic to `rotten_tomatoes/features.py`'s `_facet_tail`."""
    out = {}
    for f in MF.FACETS:
        tgt_set = target_facets.get(f) or frozenset()
        if not tgt_set:
            out[f"user_{f}_dev"] = 0.0
            out[f"user_{f}_cnt"] = 0.0
            continue
        hits = [seen_devs[i] for i, sf in enumerate(seen_facets_list)
               if sf.get(f) and (sf[f] & tgt_set)]
        out[f"user_{f}_dev"] = float(np.mean(hits)) if hits else 0.0
        out[f"user_{f}_cnt"] = float(np.log1p(len(hits)))

    genre_vec = [0.0] * len(GENRE_MH_COLS)
    for gid in target_genre_ids:
        genre_vec[gid] = 1.0
    out.update(zip(GENRE_MH_COLS, genre_vec))

    decade_vec = [0.0] * len(DECADE_MH_COLS)
    for did in target_decade_ids:
        decade_vec[did] = 1.0
    out.update(zip(DECADE_MH_COLS, decade_vec))

    out["runtime_log"] = float(target_runtime_log) if np.isfinite(target_runtime_log) else 0.0
    out["gs_rating"] = float(target_gs_rating) if np.isfinite(target_gs_rating) else 0.0
    out["n_themes_log"] = float(np.log1p(target_n_themes))
    out["n_languages_log"] = float(np.log1p(target_n_languages))
    out["n_countries_log"] = float(np.log1p(target_n_countries))
    return out


@dataclass
class FacetContext:
    """Movie facets, position-aligned to `LBData.movies` (so the sparse-matrix
    episode builders can index by integer position)."""
    facet_sets: list
    genre_mh: list
    decade_mh: list
    runtime_log: np.ndarray
    gs_rating: np.ndarray
    n_themes: np.ndarray
    n_languages: np.ndarray
    n_countries: np.ndarray


def load_project_movie_facets(movies: pd.DataFrame):
    """The movie_id-keyed `movie_features.MovieFacets` for this project (cached
    after the first run). Used directly by app-time code (`app_features`,
    validators); `build_facet_context` re-indexes the same object to sparse
    matrix positions for the offline path. ``movies`` needs `movie_id`,
    `title`, `year` columns; its own `genres` (JSON list) column, if present,
    feeds the gsimonx37-miss fallback."""
    cat = movies.drop_duplicates("movie_id").copy()
    own_genre = None
    if "genres" in cat.columns:
        def _parse(raw):
            if not isinstance(raw, str) or not raw:
                return []
            try:
                parsed = json.loads(raw)
                return [str(g).strip() for g in parsed] if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        own_genre = cat.set_index("movie_id")["genres"].map(_parse).to_dict()
    return MF.load_or_build_movie_facets(cat, own_genre=own_genre)


def build_facet_context(movies: pd.DataFrame, movie_index: pd.Index) -> FacetContext:
    """Join to gsimonx37 (cached after the first run) and re-index the result
    to the sparse matrix's movie positions."""
    mf = load_project_movie_facets(movies)
    ids = list(movie_index)
    facet_sets = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS) for mid in ids]
    genre_mh = [mf.genre_multihot.get(mid, []) for mid in ids]
    decade_mh = [mf.decade_multihot.get(mid, []) for mid in ids]

    def arr(d: dict, default: float) -> np.ndarray:
        a = np.array([d.get(mid, default) for mid in ids], dtype=np.float64)
        if default != default:
            finite = a[np.isfinite(a)]
            fill = float(finite.mean()) if len(finite) else 0.0
            a[~np.isfinite(a)] = fill
        return a

    return FacetContext(
        facet_sets=facet_sets, genre_mh=genre_mh, decade_mh=decade_mh,
        runtime_log=arr(mf.runtime_log, np.nan),
        gs_rating=arr(mf.gs_rating, np.nan),
        n_themes=arr(mf.n_themes, 0.0),
        n_languages=arr(mf.n_languages, 0.0),
        n_countries=arr(mf.n_countries, 0.0),
    )


def facet_tail_from_context(fc: FacetContext, seen_cols: np.ndarray,
                            seen_devs: np.ndarray, target_col: int) -> dict:
    seen_facets_list = [fc.facet_sets[c] for c in seen_cols]
    return _facet_tail(fc.facet_sets[target_col], fc.genre_mh[target_col],
                       fc.decade_mh[target_col], fc.runtime_log[target_col],
                       fc.gs_rating[target_col], fc.n_themes[target_col],
                       fc.n_languages[target_col], fc.n_countries[target_col],
                       seen_facets_list, seen_devs)


def facet_tail_from_ids(mf, seen_movie_ids, seen_devs: np.ndarray, target_movie_id) -> dict:
    """App-time counterpart, keyed by movie_id (a `movie_features.MovieFacets`)."""
    seen_facets_list = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS) for mid in seen_movie_ids]
    return _facet_tail(
        mf.facet_sets.get(target_movie_id, _EMPTY_FACET_SETS),
        mf.genre_multihot.get(target_movie_id, []),
        mf.decade_multihot.get(target_movie_id, []),
        mf.runtime_log.get(target_movie_id, np.nan),
        mf.gs_rating.get(target_movie_id, np.nan),
        mf.n_themes.get(target_movie_id, 0), mf.n_languages.get(target_movie_id, 0),
        mf.n_countries.get(target_movie_id, 0), seen_facets_list, seen_devs)


# ---- pure feature contract (identical logic to RT) ------------------------
def decile_features(similarities: np.ndarray, deviations: np.ndarray) -> np.ndarray:
    """Summarize similarity-times-deviation in ten similarity-ranked deciles."""
    order = np.argsort(-similarities)
    products = similarities[order] * deviations[order]
    features = np.zeros(3 * DECILES, dtype=np.float32)
    for index, chunk in enumerate(np.array_split(products, DECILES)):
        if len(chunk):
            features[index] = chunk.mean()
            features[DECILES + index] = len(chunk)
            features[2 * DECILES + index] = chunk.std()
    return features


def main_feature_row(similarities: np.ndarray, deviations: np.ndarray,
                     tail: dict) -> dict:
    values = decile_features(similarities, deviations)
    row = {column: float(value) for column, value in zip(MAIN_COLS, values)}
    row.update(tail)
    missing = [c for c in TAIL_COLS if c not in row]
    if missing:
        raise ValueError(f"missing tail features: {missing}")
    return row


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
    unknown_genre_id: int

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

    return LBData(mat.tocsc(), mask.tocsc(), members, movie_ids, member_sum,
                  member_count, movie_mean, movie_std, movie_count, genre_id,
                  float(values.mean()), unknown_genre_id)


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


def target_raters(data: LBData, target_col: int, exclude_member: int | None):
    lo, hi = data.mat_csc.indptr[target_col], data.mat_csc.indptr[target_col + 1]
    raters = data.mat_csc.indices[lo:hi]
    values = data.mat_csc.data[lo:hi]
    if exclude_member is not None:
        keep = raters != exclude_member
        raters, values = raters[keep], values[keep]
    return raters, values


def target_deviations(data: LBData, raters: np.ndarray, values: np.ndarray) -> np.ndarray:
    remaining_sum = data.member_sum[raters] - values
    remaining_count = data.member_count[raters] - 1
    peer_mean = np.full(len(raters), data.global_mean, dtype=float)
    np.divide(remaining_sum, remaining_count, out=peer_mean, where=remaining_count > 0)
    return values - peer_mean


def episode_feature_row(data: LBData, seen_cols: np.ndarray, seen_vals: np.ndarray,
                        target_col: int, exclude_member: int | None, fc: FacetContext):
    """One model-ready row for a (seen set, target) episode, or None if the
    target has too few other raters."""
    sim, _, overlap = similarity(data, seen_cols, seen_vals, exclude_member)
    raters, values = target_raters(data, target_col, exclude_member)
    if len(raters) < MIN_OTHER_REVIEWERS:
        return None
    positive_overlap = overlap[overlap > 0]
    tail = {
        "n_observed": int(len(seen_cols)),
        "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
        "max_overlap": float(overlap.max()) if len(overlap) else 0.0,
        "n_reviewers": int(len(raters)),
        "dispersion": float(data.movie_std[target_col]),
        "genre_id": int(data.genre_id[target_col]),
        "user_mean": float(np.mean(seen_vals)),
    }
    tail.update(facet_tail_from_context(fc, seen_cols, seen_vals - seen_vals.mean(), target_col))
    return main_feature_row(sim[raters], target_deviations(data, raters, values), tail)


# ---- z-score track: isolate variation from each member's own rating level --
# The user (fake profile offline, or the real visitor in the app) is
# standardized by the mean/std of THIS EPISODE's own sampled seen ratings --
# never by the member's all-time mu/sigma, since a real visitor only ever has
# their own seen films to standardize against. Peers (the target film's other
# raters) are standardized by their all-time mu/sigma (``data_z``, built with
# value="z"). Mirrors rotten_tomatoes/features.py's episode_feature_row_z.
def episode_feature_row_z(data_raw: LBData, data_z: LBData, seen_cols: np.ndarray,
                          seen_vals: np.ndarray, target_col: int,
                          exclude_member: int | None, fc: FacetContext):
    """Like `episode_feature_row`, but in z-space. Returns (row, mu, sigma) or
    None if the target lacks enough other raters, or this episode's seen
    ratings have ~zero variance (can't standardize)."""
    mu = float(seen_vals.mean())
    sigma = float(seen_vals.std(ddof=0))
    if sigma <= 1e-9:
        return None
    seen_z = (seen_vals - mu) / sigma
    sim, _, overlap = similarity(data_z, seen_cols, seen_z, exclude_member)
    raters, values_z = target_raters(data_z, target_col, exclude_member)
    if len(raters) < MIN_OTHER_REVIEWERS:
        return None
    positive_overlap = overlap[overlap > 0]
    tail = {
        "n_observed": int(len(seen_cols)),
        "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
        "max_overlap": float(overlap.max()) if len(overlap) else 0.0,
        "n_reviewers": int(len(raters)),
        "dispersion": float(data_z.movie_std[target_col]),
        "genre_id": int(data_z.genre_id[target_col]),
        "user_mean": float(seen_z.mean()),  # ~0 by construction
    }
    tail.update(facet_tail_from_context(fc, seen_cols, seen_z, target_col))
    row = main_feature_row(sim[raters], target_deviations(data_z, raters, values_z), tail)
    return row, mu, sigma


# ---- episode generation ----------------------------------------------------
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


def generate_rows(data: LBData, members: np.ndarray, rng: np.random.Generator,
                  n_grid, profiles_per_n: int, fc: FacetContext, data_z: LBData | None = None):
    """Unpaired random-holdout rows for train/val (label = held-out rating).

    If ``data_z`` is given, also builds the z-track row for the identical
    sampled episode, so raw and z rows are drawn from the same distribution.
    Returns ``(raw_frame, z_frame, mu, sigma)`` in that case, else just
    ``raw_frame``. An episode is dropped from BOTH tracks if either row is
    unavailable, keeping the two tracks exactly aligned.
    """
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    mat_csr = data.mat_csc.tocsr()
    for i, member in enumerate(members):
        member = int(member)
        lo, hi = mat_csr.indptr[member], mat_csr.indptr[member + 1]
        films = mat_csr.indices[lo:hi]
        film_vals = mat_csr.data[lo:hi]
        if len(films) < 6:
            continue
        for n in n_grid:
            size = len(films) - 1 if n is None else n
            if size < 1 or size >= len(films):
                continue
            for _ in range(profiles_per_n):
                order = rng.permutation(len(films))
                target_pos = int(order[0])
                seen_pos = order[1:size + 1]
                seen_cols, seen_vals = films[seen_pos], film_vals[seen_pos]
                target_col, target_value = int(films[target_pos]), float(film_vals[target_pos])
                row = episode_feature_row(data, seen_cols, seen_vals, target_col, member, fc)
                if row is None:
                    continue
                if data_z is not None:
                    z_result = episode_feature_row_z(data, data_z, seen_cols, seen_vals,
                                                      target_col, member, fc)
                    if z_result is None:
                        continue
                    z_row, mu, sigma = z_result
                    z_rows.append(z_row)
                    z_targets.append((target_value - mu) / sigma)
                    z_meta.append((member, target_col, -1 if n is None else n, len(seen_pos)))
                    mus.append(mu)
                    sigmas.append(sigma)
                rows.append(row)
                targets.append(target_value)
                meta.append((member, target_col, -1 if n is None else n,
                             len(seen_pos)))
        if (i + 1) % 500 == 0:
            print(f"  rows: {i + 1}/{len(members)} members")
    raw_frame = _frame(rows, targets, meta)
    if data_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def iter_paired_episodes(data: LBData, members: np.ndarray, n_grid,
                         targets_per_user: int, draws: int, n_max_finite: int):
    """Nested paired episodes: fixed targets + fixed popularity-ordered seen
    prefix per (member, draw), so every n is scored on an identical set."""
    rng = np.random.default_rng(SEED)
    mat_csr = data.mat_csc.tocsr()
    popularity = data.movie_count
    for member in members:
        member = int(member)
        lo, hi = mat_csr.indptr[member], mat_csr.indptr[member + 1]
        films = mat_csr.indices[lo:hi]
        film_vals = mat_csr.data[lo:hi]
        if len(films) <= n_max_finite:
            continue
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


def generate_paired_rows(data: LBData, members: np.ndarray, n_grid,
                         targets_per_user: int, draws: int, n_max_finite: int,
                         fc: FacetContext, data_z: LBData | None = None):
    """See `generate_rows` for the ``data_z`` contract."""
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    for (member, target_col, target_value, n, draw, seen_cols, seen_vals) in \
            iter_paired_episodes(data, members, n_grid, targets_per_user, draws, n_max_finite):
        row = episode_feature_row(data, seen_cols, seen_vals, target_col, member, fc)
        if row is None:
            continue
        if data_z is not None:
            z_result = episode_feature_row_z(data, data_z, seen_cols, seen_vals, target_col, member, fc)
            if z_result is None:
                continue
            z_row, mu, sigma = z_result
            z_rows.append(z_row)
            z_targets.append((target_value - mu) / sigma)
            z_meta.append((member, target_col, n, draw, len(seen_cols)))
            mus.append(mu)
            sigmas.append(sigma)
        rows.append(row)
        targets.append(target_value)
        meta.append((member, target_col, n, draw, len(seen_cols)))
    raw_frame = _frame(rows, targets, meta)
    if data_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def _frame(rows, targets, meta):
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    if len(features):
        features["genre_id"] = features["genre_id"].astype(int)
    paired = bool(meta) and len(meta[0]) == 5
    columns = ["member", "tcol", "n", "draw", "n_seen"] if paired else ["member", "tcol", "n", "n_seen"]
    meta_df = pd.DataFrame(meta, columns=columns)
    return features, np.asarray(targets, dtype=np.float32), meta_df


# ---- app-time construction (DataFrame path; mirrored by the browser TS) ----
def app_similarity(scores: pd.DataFrame, user: pd.Series, k_shrink: int = K_SHRINK) -> pd.DataFrame:
    """Per-member shrunk Pearson alignment + magnitude from the catalog scores."""
    overlap = scores[scores["movie_id"].isin(user.index)].copy()
    overlap["user_rating"] = overlap["movie_id"].map(user)
    overlap["xy"] = overlap["user_rating"] * overlap["rating"]
    grouped = overlap.groupby("user_id")
    out = grouped.agg(overlap=("rating", "size"), sx=("user_rating", "sum"),
                      sy=("rating", "sum"),
                      sxx=("user_rating", lambda s: float(np.dot(s, s))),
                      syy=("rating", lambda s: float(np.dot(s, s))),
                      sxy=("xy", "sum"))
    n = out["overlap"].to_numpy(float)
    cov = out["sxy"].to_numpy() - out["sx"].to_numpy() * out["sy"].to_numpy() / n
    vx = out["sxx"].to_numpy() - out["sx"].to_numpy() ** 2 / n
    vy = out["syy"].to_numpy() - out["sy"].to_numpy() ** 2 / n
    corr = np.divide(cov, np.sqrt(vx * vy), out=np.zeros_like(cov),
                     where=(n >= MIN_APP_OVERLAP) & (vx > 1e-9) & (vy > 1e-9))
    out["sim"] = corr * np.minimum(n, k_shrink) / k_shrink
    out["mag_sim"] = np.divide(out["sxy"], out["syy"], out=np.ones(len(out)),
                               where=out["syy"] > 1e-12)
    return out[["overlap", "sim", "mag_sim"]]


def app_features(target_scores: pd.DataFrame, matches: pd.DataFrame,
                 members: pd.DataFrame, user: pd.Series,
                 movie_genre: dict, unknown_genre_id: int, mf):
    """Model features for the selected catalog films. `members` is indexed by
    member_id with all-time `rating_sum`,`rating_count`. ``mf`` is a
    `movie_features.MovieFacets` (movie_id-keyed). Returns (df, movie_ids)."""
    overlap_counts = matches["overlap"].to_numpy()
    positive = overlap_counts[overlap_counts > 0]
    mean_overlap = float(positive.mean()) if len(positive) else 0.0
    max_overlap = float(overlap_counts.max()) if len(overlap_counts) else 0.0
    seen_movie_ids = list(user.index)
    seen_devs = user.to_numpy(dtype=float) - float(user.mean())
    rows, movie_ids = [], []
    for movie_id, group in target_scores.groupby("movie_id"):
        peer = members.reindex(group["user_id"])
        values = group["rating"].to_numpy(dtype=float)
        peer_count = peer["rating_count"].to_numpy(dtype=float) - 1
        peer_sum = peer["rating_sum"].to_numpy(dtype=float) - values
        peer_mean = np.full(len(group), float(values.mean()), dtype=float)
        np.divide(peer_sum, peer_count, out=peer_mean, where=peer_count > 0)
        sim = group["user_id"].map(matches["sim"]).fillna(0.0).to_numpy()
        tail = {
            "n_observed": int(len(user)),
            "mean_overlap": mean_overlap, "max_overlap": max_overlap,
            "n_reviewers": int(len(group)),
            "dispersion": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "genre_id": int(movie_genre.get(movie_id, unknown_genre_id)),
            "user_mean": float(user.mean())}
        tail.update(facet_tail_from_ids(mf, seen_movie_ids, seen_devs, movie_id))
        rows.append(main_feature_row(sim, values - peer_mean, tail))
        movie_ids.append(movie_id)
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    features["genre_id"] = features["genre_id"].astype(int)
    return features, movie_ids
