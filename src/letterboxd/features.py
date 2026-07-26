"""Letterboxd feature contract, mirroring the Rotten Tomatoes design-2/3 schema.

Per (member profile, target film) episode the learned models see a fixed-width
row: the target's raters sorted by member similarity and summarised in ten
deciles of ``similarity x (rater score - that rater's leave-one-out all-time
mean)``, plus a tail of seen count, overlap statistics, rater count,
dispersion, genre, and the member's mean rating -- and, beyond that, real
content-based affinity features built from the gsimonx37 movie-facet join
(`movie_features.py`): per-genre z-scored affinity, a theme-embedding
similarity-weighted rating estimate, per-actor affinity (by rating and by
count) plus cast-overlap statistics, and per-director affinity. This is the RT
contract **without the Tomatometer feature** (Letterboxd has no
critic-consensus meter), on the 1-10 member scale. See report.pdf's
feature-engineering section for the full column-by-column writeup.

Self-contained: this module never imports Rotten Tomatoes code (the gsimonx37
join is duplicated, not shared, in `rotten_tomatoes/movie_features.py`). The
member-by-film matrix, similarity and episode protocol it builds on live in
``pseudo_users.py``, mirroring the Rotten Tomatoes split. On top of them it
offers an offline path (``episode_feature_row`` + row generators) used for
training and analysis, and a DataFrame app-path (``app_similarity`` +
``app_features``) used by and mirrored in the browser TypeScript port.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import movie_features as MF
from .config import RATING_MIN, RATING_MAX, SEED
# The substrate this contract is built on (same split as RT's features.py /
# pseudo_users.py). Re-exported here so callers can keep reaching them through
# ``features`` as well as importing ``pseudo_users`` directly.
from .pseudo_users import (K_SHRINK, LBData, MIN_APP_OVERLAP, build_data,
                           eligible_members, iter_paired_episodes,
                           iter_paired_episodes_for_member, make_genre_maps,
                           partition_members, rmse, similarity)

MIN_OTHER_REVIEWERS = 3        # a target film needs this many OTHER raters
DECILES = 10

MAIN_COLS = ([f"d{i}_mean" for i in range(DECILES)]
             + [f"d{i}_cnt" for i in range(DECILES)]
             + [f"d{i}_std" for i in range(DECILES)])

# ---- content-based affinity contract (see module docstring) ----------------
# Genre: one z-score + one log1p(count) per canonical genre slot (20 =
# GENRE_VOCAB_K + "__other__"). The count INCLUDES the target film itself, so
# it doubles as the old multi-hot: cnt==0 -> target isn't this genre, cnt==1
# -> target is this genre and the user has seen none, cnt==n+1 -> n seen. The
# z-score only ever averages over seen films (never the target), so this adds
# no leakage.
GENRE_WIDTH = MF.GENRE_VOCAB_K + 1
GENRE_Z_COLS = [f"user_genre_{g}_z" for g in range(GENRE_WIDTH)]
GENRE_CNT_COLS = [f"user_genre_{g}_cnt" for g in range(GENRE_WIDTH)]

# Theme: a similarity-weighted average RAW rating (not a deviation -- see
# report.pdf) over every (target theme, seen film) pair, weighted by the
# sentence-embedding cosine similarity between the target theme and the seen
# film's most-similar theme; plus the total similarity mass (the confidence
# companion, and the exact mask for the average -- 0 iff the average is NaN)
# and a count of strongly on-theme (>0.8 cosine) seen films.
THEME_COLS = ["user_theme_avg", "user_theme_mass_log", "user_theme_simcnt_hi"]

# Actor: three top-5 rankings of the target's cast against the user's seen
# history -- by the user's mean rating for that actor, by the user's viewing
# count for that actor, and seen films ranked by shared-cast count with the
# target. Unused ranks are padded (z/rating -> NaN, count -> 0).
ACTOR_BYRATING_Z_COLS = [f"user_actor_byrating{i}_z" for i in range(1, 6)]
ACTOR_BYRATING_CNT_COLS = [f"user_actor_byrating{i}_cnt" for i in range(1, 6)]
ACTOR_BYCOUNT_Z_COLS = [f"user_actor_bycount{i}_z" for i in range(1, 6)]
ACTOR_BYCOUNT_CNT_COLS = [f"user_actor_bycount{i}_cnt" for i in range(1, 6)]
CAST_OVERLAP_N_COLS = [f"user_cast_overlap{i}_n" for i in range(1, 6)]
CAST_OVERLAP_RATING_COLS = [f"user_cast_overlap{i}_rating" for i in range(1, 6)]

# Director: one z-score + one log1p(count) over seen films sharing any
# director with the target.
DIRECTOR_COLS = ["user_director_z", "user_director_cnt"]

NUMERIC_TAIL_COLS = ["runtime_log", "gs_rating", "n_themes_log", "n_languages_log"]

FACET_TAIL_COLS = (GENRE_Z_COLS + GENRE_CNT_COLS + THEME_COLS
                   + ACTOR_BYRATING_Z_COLS + ACTOR_BYRATING_CNT_COLS
                   + ACTOR_BYCOUNT_Z_COLS + ACTOR_BYCOUNT_CNT_COLS
                   + CAST_OVERLAP_N_COLS + CAST_OVERLAP_RATING_COLS
                   + DIRECTOR_COLS + NUMERIC_TAIL_COLS)

TAIL_COLS = (["n_observed", "mean_overlap", "max_overlap", "n_reviewers",
              "dispersion", "user_mean"] + FACET_TAIL_COLS)
FEATURE_COLS = MAIN_COLS + TAIL_COLS
_EMPTY_FACET_SETS = {f: frozenset() for f in MF.FACETS}


# ---- per-block affinity computation (identical logic to RT) ----------------
def _genre_block(target_genre_ids: frozenset, seen_genre_ids_list: list,
                 seen_values: np.ndarray, mu_u: float, sigma_u: float) -> dict:
    """One pass over the (typically 1-3-genre) seen sets, bucketing each seen
    film's rating into every genre slot it belongs to, rather than rescanning
    the whole seen list once per of the 20 genre slots (the original approach
    -- ~20x more scanning for the same result)."""
    sums = np.zeros(GENRE_WIDTH)
    counts = np.zeros(GENRE_WIDTH, dtype=np.int64)
    for gset, value in zip(seen_genre_ids_list, seen_values):
        for g in gset:
            sums[g] += value
            counts[g] += 1
    out = {}
    for g in range(GENRE_WIDTH):
        cnt = int(counts[g]) + (1 if g in target_genre_ids else 0)
        out[f"user_genre_{g}_z"] = (sums[g] / counts[g] - mu_u) / sigma_u if counts[g] else np.nan
        out[f"user_genre_{g}_cnt"] = float(np.log1p(cnt))
    return out


def _theme_block(target_theme_ids: frozenset, seen_theme_ids_list: list,
                 seen_values: np.ndarray, theme_matrix: np.ndarray) -> dict:
    """Similarity-weighted average rating over (target theme, seen film)
    pairs. Per target theme t and seen film f with theme ids T_f:
    w(t,f) = max_{u in T_f} cos(e_t, e_u); the combined average collapses to
    Σ_{t,f} w(t,f)·rating(f) / Σ_{t,f} w(t,f) (see report.pdf for the algebra
    showing this is exactly the mass-weighted combination of each theme's own
    weighted average). Identical logic to rotten_tomatoes/features.py."""
    if not target_theme_ids:
        return {"user_theme_avg": np.nan, "user_theme_mass_log": 0.0,
                "user_theme_simcnt_hi": 0.0}
    S = np.fromiter(target_theme_ids, dtype=np.int64)
    target_rows = theme_matrix[S]  # gather the target-theme rows once, not per film
    total_num, total_den, hi_count = 0.0, 0.0, 0
    for tset, rating in zip(seen_theme_ids_list, seen_values):
        if not tset:
            continue
        Tf = np.fromiter(tset, dtype=np.int64)
        w = np.clip(target_rows[:, Tf].max(axis=1), 0.0, 1.0)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        total_num += w_sum * float(rating)
        total_den += w_sum
        if float(w.max()) > 0.8:
            hi_count += 1
    avg = total_num / total_den if total_den > 0 else np.nan
    return {"user_theme_avg": avg, "user_theme_mass_log": float(np.log1p(total_den)),
            "user_theme_simcnt_hi": float(np.log1p(hi_count))}


def _actor_block(target_cast: frozenset, seen_cast_list: list,
                 seen_values: np.ndarray, mu_u: float, sigma_u: float) -> dict:
    actor_ratings: dict = {}
    for cast, rating in zip(seen_cast_list, seen_values):
        if not cast:
            continue
        for a in (cast & target_cast):
            actor_ratings.setdefault(a, []).append(float(rating))
    entries = [(a, float(np.mean(rs)), len(rs)) for a, rs in actor_ratings.items()]
    by_rating = sorted(entries, key=lambda e: (-e[1], -e[2]))[:5]
    by_count = sorted(entries, key=lambda e: (-e[2], -e[1]))[:5]

    out = {}
    for i in range(5):
        z_col, cnt_col = ACTOR_BYRATING_Z_COLS[i], ACTOR_BYRATING_CNT_COLS[i]
        if i < len(by_rating):
            _, avg, cnt = by_rating[i]
            out[z_col] = (avg - mu_u) / sigma_u
            out[cnt_col] = float(np.log1p(cnt))
        else:
            out[z_col] = np.nan
            out[cnt_col] = 0.0
    for i in range(5):
        z_col, cnt_col = ACTOR_BYCOUNT_Z_COLS[i], ACTOR_BYCOUNT_CNT_COLS[i]
        if i < len(by_count):
            _, avg, cnt = by_count[i]
            out[z_col] = (avg - mu_u) / sigma_u
            out[cnt_col] = float(np.log1p(cnt))
        else:
            out[z_col] = np.nan
            out[cnt_col] = 0.0

    overlaps = []
    for cast, rating in zip(seen_cast_list, seen_values):
        if not cast:
            continue
        n = len(cast & target_cast)
        if n > 0:
            overlaps.append((n, float(rating)))
    overlaps.sort(key=lambda e: -e[0])
    for i in range(5):
        n_col, rating_col = CAST_OVERLAP_N_COLS[i], CAST_OVERLAP_RATING_COLS[i]
        if i < len(overlaps):
            n, rating = overlaps[i]
            out[n_col] = float(np.log1p(n))
            out[rating_col] = rating
        else:
            out[n_col] = 0.0
            out[rating_col] = np.nan
    return out


def _director_block(target_directors: frozenset, seen_director_list: list,
                    seen_values: np.ndarray, mu_u: float, sigma_u: float) -> dict:
    hits = [float(r) for d, r in zip(seen_director_list, seen_values) if d and (d & target_directors)]
    if hits:
        return {"user_director_z": (float(np.mean(hits)) - mu_u) / sigma_u,
                "user_director_cnt": float(np.log1p(len(hits)))}
    return {"user_director_z": np.nan, "user_director_cnt": 0.0}


def _facet_tail(target_genre_ids: frozenset, target_theme_ids: frozenset,
                target_actor_set: frozenset, target_director_set: frozenset,
                target_runtime_log: float, target_gs_rating: float,
                target_n_themes: float, target_n_languages: float,
                seen_genre_ids_list: list, seen_theme_ids_list: list,
                seen_actor_list: list, seen_director_list: list,
                seen_values: np.ndarray, theme_matrix: np.ndarray,
                global_std: float) -> dict:
    """Core affinity-tail computation, agnostic to whether the caller resolved
    facets by sparse-matrix position (offline) or by movie_id (app-time).
    ``seen_values`` are RAW ratings on whichever scale the caller is working
    in (the raw track's actual ratings, or the z-track's already-standardized
    per-episode z-values) -- every block computes its own local mu_u/sigma_u
    from this array and expresses its affinity relative to it, falling back
    to ``global_std`` when the seen set has ~zero variance. Identical logic
    to rotten_tomatoes/features.py's `_facet_tail`."""
    mu_u = float(seen_values.mean()) if len(seen_values) else 0.0
    sigma_u = float(seen_values.std(ddof=0)) if len(seen_values) else 0.0
    if sigma_u < 1e-9:
        sigma_u = global_std if global_std > 1e-9 else 1.0

    out = {}
    out.update(_genre_block(target_genre_ids, seen_genre_ids_list, seen_values, mu_u, sigma_u))
    out.update(_theme_block(target_theme_ids, seen_theme_ids_list, seen_values, theme_matrix))
    out.update(_actor_block(target_actor_set, seen_actor_list, seen_values, mu_u, sigma_u))
    out.update(_director_block(target_director_set, seen_director_list, seen_values, mu_u, sigma_u))
    out["runtime_log"] = float(target_runtime_log) if np.isfinite(target_runtime_log) else 0.0
    out["gs_rating"] = float(target_gs_rating) if np.isfinite(target_gs_rating) else 0.0
    out["n_themes_log"] = float(np.log1p(target_n_themes))
    out["n_languages_log"] = float(np.log1p(target_n_languages))
    return out


@dataclass
class FacetContext:
    """Movie facets, position-aligned to `LBData.movies` (so the sparse-matrix
    episode builders can index by integer position). Genre/theme are resolved
    to fixed-vocab ids once here; actor/director stay as raw-string sets."""
    genre_ids: list           # position -> frozenset[int] canonical genre ids
    theme_ids: list           # position -> frozenset[int] theme-vocab ids
    actor_sets: list          # position -> frozenset[str]
    director_sets: list       # position -> frozenset[str]
    runtime_log: np.ndarray
    gs_rating: np.ndarray
    n_themes: np.ndarray
    n_languages: np.ndarray
    theme_matrix: np.ndarray
    global_std: float


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


def load_project_theme_similarity():
    """The theme embedding vocab + cosine similarity matrix (cached after the
    first run) -- the full known gsimonx37 theme vocabulary, independent of
    which films are in this project's catalog."""
    return MF.load_or_build_theme_similarity(MF.load_all_themes())


def _theme_ids_of(mf, theme_sim, mid) -> frozenset:
    strs = mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["theme"]
    return frozenset(theme_sim.vocab[t] for t in strs if t in theme_sim.vocab)


def build_facet_context(movies: pd.DataFrame, movie_index: pd.Index,
                        global_std: float) -> FacetContext:
    """Join to gsimonx37 (cached after the first run) and re-index the result
    to the sparse matrix's movie positions."""
    mf = load_project_movie_facets(movies)
    theme_sim = load_project_theme_similarity()
    ids = list(movie_index)
    genre_ids = [frozenset(mf.genre_multihot.get(mid, [])) for mid in ids]
    theme_ids = [_theme_ids_of(mf, theme_sim, mid) for mid in ids]
    actor_sets = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["actor"] for mid in ids]
    director_sets = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["director"] for mid in ids]

    def arr(d: dict, default: float) -> np.ndarray:
        a = np.array([d.get(mid, default) for mid in ids], dtype=np.float64)
        if default != default:
            finite = a[np.isfinite(a)]
            fill = float(finite.mean()) if len(finite) else 0.0
            a[~np.isfinite(a)] = fill
        return a

    return FacetContext(
        genre_ids=genre_ids, theme_ids=theme_ids, actor_sets=actor_sets,
        director_sets=director_sets,
        runtime_log=arr(mf.runtime_log, np.nan),
        gs_rating=arr(mf.gs_rating, np.nan),
        n_themes=arr(mf.n_themes, 0.0),
        n_languages=arr(mf.n_languages, 0.0),
        theme_matrix=theme_sim.matrix, global_std=global_std,
    )


def facet_tail_from_context(fc: FacetContext, seen_cols: np.ndarray,
                            seen_values: np.ndarray, target_col: int) -> dict:
    """``seen_values`` must be RAW values on the caller's scale (not
    deviations) -- see `_facet_tail`'s docstring."""
    seen_genre_ids_list = [fc.genre_ids[c] for c in seen_cols]
    seen_theme_ids_list = [fc.theme_ids[c] for c in seen_cols]
    seen_actor_list = [fc.actor_sets[c] for c in seen_cols]
    seen_director_list = [fc.director_sets[c] for c in seen_cols]
    return _facet_tail(fc.genre_ids[target_col], fc.theme_ids[target_col],
                       fc.actor_sets[target_col], fc.director_sets[target_col],
                       fc.runtime_log[target_col], fc.gs_rating[target_col],
                       fc.n_themes[target_col], fc.n_languages[target_col],
                       seen_genre_ids_list, seen_theme_ids_list,
                       seen_actor_list, seen_director_list,
                       seen_values, fc.theme_matrix, fc.global_std)


def facet_tail_from_ids(mf, theme_sim, seen_movie_ids, seen_values: np.ndarray,
                        target_movie_id, global_std: float) -> dict:
    """App-time counterpart, keyed by movie_id (a `movie_features.MovieFacets`).
    ``seen_values`` must be RAW ratings (not deviations)."""
    def genre_ids_of(mid):
        return frozenset(mf.genre_multihot.get(mid, []))

    def actor_of(mid):
        return mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["actor"]

    def director_of(mid):
        return mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["director"]

    seen_genre_ids_list = [genre_ids_of(mid) for mid in seen_movie_ids]
    seen_theme_ids_list = [_theme_ids_of(mf, theme_sim, mid) for mid in seen_movie_ids]
    seen_actor_list = [actor_of(mid) for mid in seen_movie_ids]
    seen_director_list = [director_of(mid) for mid in seen_movie_ids]

    return _facet_tail(
        genre_ids_of(target_movie_id), _theme_ids_of(mf, theme_sim, target_movie_id),
        actor_of(target_movie_id), director_of(target_movie_id),
        mf.runtime_log.get(target_movie_id, np.nan), mf.gs_rating.get(target_movie_id, np.nan),
        mf.n_themes.get(target_movie_id, 0), mf.n_languages.get(target_movie_id, 0),
        seen_genre_ids_list, seen_theme_ids_list, seen_actor_list, seen_director_list,
        seen_values, theme_sim.matrix, global_std)


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
        "user_mean": float(np.mean(seen_vals)),
    }
    tail.update(facet_tail_from_context(fc, seen_cols, seen_vals, target_col))
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
        "user_mean": float(seen_z.mean()),  # ~0 by construction
    }
    # seen_z is already the per-episode standardized rating (mean 0, std 1),
    # so it IS the "raw value on this track's scale" the affinity tail wants.
    tail.update(facet_tail_from_context(fc, seen_cols, seen_z, target_col))
    row = main_feature_row(sim[raters], target_deviations(data_z, raters, values_z), tail)
    return row, mu, sigma


# ---- episode generation, parallelized over members --------------------------
# generate_rows/generate_paired_rows fan out over independent members, so the
# per-member work (below) can run in a ProcessPoolExecutor. data/data_z/fc are
# read-only for the whole call, so each worker receives them exactly once (via
# the pool initializer) rather than once per task. Each member draws from its
# own RNG stream seeded by (base_seed, member) -- never a stream threaded
# sequentially through the member loop -- so the resulting rows are identical
# regardless of --jobs or how members are chunked across workers.
_WORKER: dict = {}


def _init_worker(data, data_z, fc):
    _WORKER["data"] = data
    _WORKER["data_z"] = data_z
    _WORKER["fc"] = fc


def _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas):
    r_rows, r_targets, r_meta, r_z_rows, r_z_targets, r_z_meta, r_mus, r_sigmas = result
    rows.extend(r_rows); targets.extend(r_targets); meta.extend(r_meta)
    z_rows.extend(r_z_rows); z_targets.extend(r_z_targets); z_meta.extend(r_z_meta)
    mus.extend(r_mus); sigmas.extend(r_sigmas)


def _rows_one_member(data, data_z, fc, member, n_grid, profiles_per_n, base_seed):
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    rng = np.random.default_rng([base_seed, member])
    films, film_vals = data.member_hist[member]
    if len(films) < 6:
        return rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas
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
            meta.append((member, target_col, -1 if n is None else n, len(seen_pos)))
    return rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas


def _rows_task(args):
    member, n_grid, profiles_per_n, base_seed = args
    return _rows_one_member(_WORKER["data"], _WORKER["data_z"], _WORKER["fc"],
                            member, n_grid, profiles_per_n, base_seed)


def _paired_rows_one_member(data, data_z, fc, member, n_grid, targets_per_user,
                            draws, n_max_finite, seed):
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    for (member_, target_col, target_value, n, draw, seen_cols, seen_vals) in \
            iter_paired_episodes_for_member(data, member, n_grid, targets_per_user,
                                            draws, n_max_finite, seed):
        row = episode_feature_row(data, seen_cols, seen_vals, target_col, member_, fc)
        if row is None:
            continue
        if data_z is not None:
            z_result = episode_feature_row_z(data, data_z, seen_cols, seen_vals,
                                              target_col, member_, fc)
            if z_result is None:
                continue
            z_row, mu, sigma = z_result
            z_rows.append(z_row)
            z_targets.append((target_value - mu) / sigma)
            z_meta.append((member_, target_col, n, draw, len(seen_cols)))
            mus.append(mu)
            sigmas.append(sigma)
        rows.append(row)
        targets.append(target_value)
        meta.append((member_, target_col, n, draw, len(seen_cols)))
    return rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas


def _paired_rows_task(args):
    member, n_grid, targets_per_user, draws, n_max_finite, seed = args
    return _paired_rows_one_member(_WORKER["data"], _WORKER["data_z"], _WORKER["fc"],
                                   member, n_grid, targets_per_user, draws,
                                   n_max_finite, seed)


def generate_rows(data: LBData, members: np.ndarray, rng: np.random.Generator,
                  n_grid, profiles_per_n: int, fc: FacetContext,
                  data_z: LBData | None = None, jobs: int = 1):
    """Unpaired random-holdout rows for train/val (label = held-out rating).

    If ``data_z`` is given, also builds the z-track row for the identical
    sampled episode, so raw and z rows are drawn from the same distribution.
    Returns ``(raw_frame, z_frame, mu, sigma)`` in that case, else just
    ``raw_frame``. An episode is dropped from BOTH tracks if either row is
    unavailable, keeping the two tracks exactly aligned.

    ``jobs`` > 1 splits the members across a `ProcessPoolExecutor` (see the
    module-level worker helpers above); ``jobs=1`` runs in-process exactly as
    before. One integer is drawn from ``rng`` up front to seed every member's
    independent stream, so the resulting rows are identical either way.
    """
    members = [int(m) for m in members]
    base_seed = int(rng.integers(0, 2**63 - 1))
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    if jobs <= 1:
        for i, member in enumerate(members):
            result = _rows_one_member(data, data_z, fc, member, n_grid, profiles_per_n, base_seed)
            _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
            if (i + 1) % 500 == 0:
                print(f"  rows: {i + 1}/{len(members)} members")
    else:
        tasks = [(m, n_grid, profiles_per_n, base_seed) for m in members]
        chunksize = max(1, len(members) // (jobs * 4))
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(data, data_z, fc)) as ex:
            for i, result in enumerate(ex.map(_rows_task, tasks, chunksize=chunksize)):
                _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
                if (i + 1) % 500 == 0:
                    print(f"  rows: {i + 1}/{len(members)} members")
    raw_frame = _frame(rows, targets, meta)
    if data_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def generate_paired_rows(data: LBData, members: np.ndarray, n_grid,
                         targets_per_user: int, draws: int, n_max_finite: int,
                         fc: FacetContext, data_z: LBData | None = None,
                         jobs: int = 1, seed: int = SEED):
    """See `generate_rows` for the ``data_z`` and ``jobs`` contract."""
    members = [int(m) for m in members]
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    if jobs <= 1:
        for member in members:
            result = _paired_rows_one_member(data, data_z, fc, member, n_grid,
                                             targets_per_user, draws, n_max_finite, seed)
            _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
    else:
        tasks = [(m, n_grid, targets_per_user, draws, n_max_finite, seed) for m in members]
        chunksize = max(1, len(members) // (jobs * 4))
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(data, data_z, fc)) as ex:
            for result in ex.map(_paired_rows_task, tasks, chunksize=chunksize):
                _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
    raw_frame = _frame(rows, targets, meta)
    if data_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def _frame(rows, targets, meta):
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
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
                 members: pd.DataFrame, user: pd.Series, mf, theme_sim,
                 global_std: float):
    """Model features for the selected catalog films. `members` is indexed by
    member_id with all-time `rating_sum`,`rating_count`. ``mf`` is a
    `movie_features.MovieFacets` (movie_id-keyed), ``theme_sim`` a
    `movie_features.ThemeSimilarity`. Returns (df, movie_ids)."""
    overlap_counts = matches["overlap"].to_numpy()
    positive = overlap_counts[overlap_counts > 0]
    mean_overlap = float(positive.mean()) if len(positive) else 0.0
    max_overlap = float(overlap_counts.max()) if len(overlap_counts) else 0.0
    seen_movie_ids = list(user.index)
    seen_values = user.to_numpy(dtype=float)
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
            "user_mean": float(user.mean())}
        tail.update(facet_tail_from_ids(mf, theme_sim, seen_movie_ids, seen_values,
                                        movie_id, global_std))
        rows.append(main_feature_row(sim, values - peer_mean, tail))
        movie_ids.append(movie_id)
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    return features, movie_ids
