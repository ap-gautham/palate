"""Design 2/3 feature contract and construction.

The learned models never see the critic matrix directly; they see, per
(pseudo-user, target movie) episode, a fixed-width feature row: reviewers of
the target sorted by similarity and summarized in ten deciles of
`similarity x (critic score - that critic's leave-one-out mean)`, plus a tail
of seen count, overlap statistics, mapped Tomatometer, reviewer count,
dispersion, genre, and the user's mean rating -- and, beyond that, real
content-based affinity features built from the gsimonx37 movie-facet join
(`movie_features.py`): per-genre z-scored affinity, a theme-embedding
similarity-weighted rating estimate, per-actor affinity (by rating and by
count) plus cast-overlap statistics, and per-director affinity. See
report.pdf's feature-engineering section for the full column-by-column
writeup and the reasoning behind each design choice.

This module is self-contained within the package and shared verbatim by
``train_xgboost.py`` and ``train_neural.py`` (imported, not copied) so the two
learned models are trained and scored on identical features (a fair
comparison). It contains three layers: the pure contract (`main_feature_row`),
offline generation from a Split (`generate_rows`, `generate_paired_rows`), and
app-time construction (`app_similarity`, `app_features`).
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rotten_tomatoes import movie_features as MF
from rotten_tomatoes.config import MIN_OTHER_REVIEWERS, SEED
from .pseudo_users import (iter_paired_episodes_for_user, sample_random_holdout,
                           similarity, target_ok_mask)

MIN_APP_OVERLAP = 2
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

TAIL_COLS = (["n_observed", "mean_overlap", "max_overlap", "tomatometer",
              "n_reviewers", "dispersion", "user_mean"]
             + FACET_TAIL_COLS)
FEATURE_COLS = MAIN_COLS + TAIL_COLS
_EMPTY_FACET_SETS = {f: frozenset() for f in MF.FACETS}


# ---- per-block affinity computation -----------------------------------------
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
    weighted average)."""
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
    to ``global_std`` when the seen set has ~zero variance (e.g. n=1, or a
    tied small n) so a z-score is never a divide-by-zero or an artifact of a
    tiny sample's own spread."""
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
    """Movie facets, position-aligned to a Split's `tgt_movie_index` (so the
    sparse-matrix episode builders can index by integer position, the same
    way `sp.tm_z`/`sp.dispersion` already do). Genre/theme are resolved to
    fixed-vocab ids once here (offline, O(catalog) not O(episodes)); actor/
    director stay as raw-string sets (small, per-movie, no vocab needed)."""
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
    predict_*.py, validators); `build_facet_context` re-indexes the same
    object to Split positions for the offline sparse-matrix path."""
    cat, own_genre = MF.prepare_rt_catalog(movies)
    return MF.load_or_build_movie_facets(cat, own_genre=own_genre)


def load_project_theme_similarity():
    """The theme embedding vocab + cosine similarity matrix (cached after the
    first run) -- the full known gsimonx37 theme vocabulary, independent of
    which films are in this project's catalog."""
    return MF.load_or_build_theme_similarity(MF.load_all_themes())


def _theme_ids_of(mf, theme_sim, mid) -> frozenset:
    strs = mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["theme"]
    return frozenset(theme_sim.vocab[t] for t in strs if t in theme_sim.vocab)


def build_facet_context(movies: pd.DataFrame, tgt_movie_index: pd.Index,
                        global_std: float) -> FacetContext:
    """Join to gsimonx37 (cached after the first run) and re-index the result
    to the Split's movie positions."""
    mf = load_project_movie_facets(movies)
    theme_sim = load_project_theme_similarity()
    ids = list(tgt_movie_index)
    genre_ids = [frozenset(mf.genre_multihot.get(mid, [])) for mid in ids]
    theme_ids = [_theme_ids_of(mf, theme_sim, mid) for mid in ids]
    actor_sets = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["actor"] for mid in ids]
    director_sets = [mf.facet_sets.get(mid, _EMPTY_FACET_SETS)["director"] for mid in ids]

    def arr(d: dict, default: float) -> np.ndarray:
        a = np.array([d.get(mid, default) for mid in ids], dtype=np.float64)
        if default != default:  # default is NaN -> impute to the column mean
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
    """App-time counterpart of `facet_tail_from_context`, keyed by movie_id
    (the `mf: movie_features.MovieFacets` object) instead of Split position.
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


# ---- pure feature contract ------------------------------------------------
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
    """One model-ready feature row in stable column order."""
    values = decile_features(similarities, deviations)
    row = {column: float(value) for column, value in zip(MAIN_COLS, values)}
    row.update(tail)
    missing = [c for c in TAIL_COLS if c not in row]
    if missing:
        raise ValueError(f"missing tail features: {missing}")
    return row


# ---- offline generation from a Split --------------------------------------
def make_genre_maps(movies: pd.DataFrame):
    genres = (movies.drop_duplicates("movie_id").set_index("movie_id")["genre"]
              .fillna("").str.split(",").str[0].str.strip())
    genre_to_id = {g: i for i, g in enumerate(sorted(genres.unique()))}
    unknown_genre_id = len(genre_to_id)
    genre_to_id["__unknown__"] = unknown_genre_id
    movie_to_id = {mid: genre_to_id[g] for mid, g in genres.items()}
    return movie_to_id, genre_to_id, unknown_genre_id


def reviewers_of(sp, upos: int, target_col: int):
    lo, hi = sp.TT.indptr[target_col], sp.TT.indptr[target_col + 1]
    critics = sp.TT.indices[lo:hi]
    values = sp.TT.data[lo:hi]
    keep = critics != upos
    return critics[keep], values[keep]


def target_deviations(sp, critics: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each peer's target score minus that peer's leave-one-out all-time mean."""
    remaining_sum = sp.critic_sum[critics] - values
    remaining_count = sp.critic_count[critics] - 1
    peer_mean = np.full(len(critics), sp.global_mean, dtype=float)
    np.divide(remaining_sum, remaining_count, out=peer_mean, where=remaining_count > 0)
    return values - peer_mean


def episode_feature_row(sp, upos, seen_cols, seen_values, target_col,
                        fc: FacetContext):
    """Build one feature row for a (seen set, target) episode, or None if the
    target lacks enough other reviewers."""
    sim, overlap, _ = similarity(sp, upos, seen_cols, seen_values)
    critics, values = reviewers_of(sp, upos, target_col)
    if len(critics) < MIN_OTHER_REVIEWERS:
        return None
    positive_overlap = overlap[overlap > 0]
    tail = {
        "n_observed": int(len(seen_cols)),
        "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
        "max_overlap": float(overlap.max()),
        "tomatometer": float(sp.tm_z[target_col]),
        "n_reviewers": int(len(critics)),
        "dispersion": float(sp.dispersion[target_col]),
        "user_mean": float(seen_values.mean()),
    }
    tail.update(facet_tail_from_context(fc, seen_cols, seen_values, target_col))
    return main_feature_row(sim[critics], target_deviations(sp, critics, values), tail)


# ---- z-score track: isolate variation from each user's own rating level ----
# The user (fake pseudo-user offline, or the real visitor in the app) is
# standardized by the mean/std of THIS EPISODE's own sampled seen ratings --
# never by the critic's all-time mu/sigma, since a real visitor only ever has
# their own seen films to standardize against. Peers (the target movie's other
# reviewers) are standardized by their all-time mu/sigma (``sp_z``, built with
# value_col="z"). Because a z-episode is a transform of the SAME sampled raw
# episode (not an independent draw), the raw seen_cols/seen_values here must
# come from the raw Split's sampler -- this function only re-expresses them.
def episode_feature_row_z(sp_raw, sp_z, upos, seen_cols, seen_values, target_col,
                          fc: FacetContext):
    """Like `episode_feature_row`, but in z-space. Returns (row, mu, sigma) or
    None if the target lacks enough other reviewers, or this episode's seen
    ratings have ~zero variance (can't standardize)."""
    mu = float(seen_values.mean())
    sigma = float(seen_values.std(ddof=0))
    if sigma <= 1e-9:
        return None
    seen_z = (seen_values - mu) / sigma
    sim, overlap, _ = similarity(sp_z, upos, seen_cols, seen_z)
    critics, values_z = reviewers_of(sp_z, upos, target_col)
    if len(critics) < MIN_OTHER_REVIEWERS:
        return None
    positive_overlap = overlap[overlap > 0]
    tail = {
        "n_observed": int(len(seen_cols)),
        "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
        "max_overlap": float(overlap.max()),
        "tomatometer": float(sp_raw.tm_z[target_col]),  # external feature, stays raw-scale
        "n_reviewers": int(len(critics)),
        "dispersion": float(sp_z.dispersion[target_col]),
        "user_mean": float(seen_z.mean()),  # ~0 by construction -- the level we removed
    }
    # seen_z is already the per-episode standardized rating (mean 0, std 1),
    # so it IS the "raw value on this track's scale" the affinity tail wants.
    tail.update(facet_tail_from_context(fc, seen_cols, seen_z, target_col))
    row = main_feature_row(sim[critics], target_deviations(sp_z, critics, values_z), tail)
    return row, mu, sigma


# ---- row generation, parallelized over users --------------------------------
# generate_rows/generate_paired_rows fan out over independent pseudo-users, so
# the per-user work (below) can run in a ProcessPoolExecutor. sp/sp_z/fc are
# read-only for the whole call, so each worker receives them exactly once (via
# the pool initializer) rather than once per task. Each user draws from its
# own RNG stream seeded by (base_seed, upos) -- never a stream threaded
# sequentially through the user loop -- so the resulting rows are identical
# regardless of --jobs or how users are chunked across workers.
_WORKER: dict = {}


def _init_worker(sp, sp_z, fc):
    _WORKER["sp"] = sp
    _WORKER["sp_z"] = sp_z
    _WORKER["fc"] = fc


def _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas):
    r_rows, r_targets, r_meta, r_z_rows, r_z_targets, r_z_meta, r_mus, r_sigmas = result
    rows.extend(r_rows); targets.extend(r_targets); meta.extend(r_meta)
    z_rows.extend(r_z_rows); z_targets.extend(r_z_targets); z_meta.extend(r_z_meta)
    mus.extend(r_mus); sigmas.extend(r_sigmas)


def _rows_one_user(sp, sp_z, fc, upos, n_grid, profiles_per_n, base_seed):
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    rng = np.random.default_rng([base_seed, upos])
    target_ok = target_ok_mask(sp, upos)
    for n in n_grid:
        for profile in range(profiles_per_n):
            episode = sample_random_holdout(rng, sp, upos, n, target_ok)
            if episode is None:
                continue
            seen_cols, seen_values, target_col, target_value = episode
            row = episode_feature_row(sp, upos, seen_cols, seen_values, target_col, fc)
            if row is None:
                continue
            if sp_z is not None:
                z_result = episode_feature_row_z(sp, sp_z, upos, seen_cols, seen_values,
                                                  target_col, fc)
                if z_result is None:
                    continue
                z_row, mu, sigma = z_result
                z_rows.append(z_row)
                z_targets.append((target_value - mu) / sigma)
                z_meta.append((upos, target_col, -1 if n is None else n, profile, len(seen_cols)))
                mus.append(mu)
                sigmas.append(sigma)
            rows.append(row)
            targets.append(target_value)
            meta.append((upos, target_col, -1 if n is None else n, profile, len(seen_cols)))
    return rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas


def _rows_task(args):
    upos, n_grid, profiles_per_n, base_seed = args
    return _rows_one_user(_WORKER["sp"], _WORKER["sp_z"], _WORKER["fc"],
                          upos, n_grid, profiles_per_n, base_seed)


def _paired_rows_one_user(sp, sp_z, fc, upos, seed):
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    for (upos_, target_col, target_value, n, draw, seen_cols, seen_values) in \
            iter_paired_episodes_for_user(sp, upos, seed):
        row = episode_feature_row(sp, upos_, seen_cols, seen_values, target_col, fc)
        if row is None:
            continue
        if sp_z is not None:
            z_result = episode_feature_row_z(sp, sp_z, upos_, seen_cols, seen_values,
                                              target_col, fc)
            if z_result is None:
                continue
            z_row, mu, sigma = z_result
            z_rows.append(z_row)
            z_targets.append((target_value - mu) / sigma)
            z_meta.append((upos_, target_col, n, draw, len(seen_cols)))
            mus.append(mu)
            sigmas.append(sigma)
        rows.append(row)
        targets.append(target_value)
        meta.append((upos_, target_col, n, draw, len(seen_cols)))
    return rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas


def _paired_rows_task(args):
    upos, seed = args
    return _paired_rows_one_user(_WORKER["sp"], _WORKER["sp_z"], _WORKER["fc"], upos, seed)


def generate_rows(sp, users, rng, n_grid, profiles_per_n, fc: FacetContext,
                  sp_z=None, jobs: int = 1):
    """Training/validation rows: many unpaired random profiles per critic.

    If ``sp_z`` is given, also builds the z-track row for the identical
    sampled episode (same seen movies/values), so raw and z rows are drawn
    from the same distribution of episodes. Returns
    ``(raw_frame, z_frame, mu, sigma)`` in that case, else just ``raw_frame``.
    An episode is dropped from BOTH tracks if either row is unavailable (e.g.
    a target with too few other reviewers, or -- z only -- ~zero seen-rating
    variance), keeping the two tracks exactly aligned for comparison.

    ``jobs`` > 1 splits the users across a `ProcessPoolExecutor` (see the
    module-level worker helpers above); ``jobs=1`` runs in-process exactly as
    before. One integer is drawn from ``rng`` up front to seed every user's
    independent stream, so the resulting rows are identical either way.
    """
    users = [int(u) for u in users]
    base_seed = int(rng.integers(0, 2**63 - 1))
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    if jobs <= 1:
        for i, upos in enumerate(users):
            result = _rows_one_user(sp, sp_z, fc, upos, n_grid, profiles_per_n, base_seed)
            _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(users)} critics")
    else:
        tasks = [(u, n_grid, profiles_per_n, base_seed) for u in users]
        chunksize = max(1, len(users) // (jobs * 4))
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(sp, sp_z, fc)) as ex:
            for i, result in enumerate(ex.map(_rows_task, tasks, chunksize=chunksize)):
                _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
                if (i + 1) % 250 == 0:
                    print(f"  {i + 1}/{len(users)} critics")
    raw_frame = _frame(rows, targets, meta)
    if sp_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def generate_paired_rows(sp, users, fc: FacetContext, sp_z=None, jobs: int = 1,
                         seed: int = SEED):
    """Test rows on the shared paired, nested episodes (identical keys across
    all designs). See `generate_rows` for the ``sp_z`` and ``jobs`` contract."""
    users = sorted(int(u) for u in users)
    rows, targets, meta = [], [], []
    z_rows, z_targets, z_meta, mus, sigmas = [], [], [], [], []
    if jobs <= 1:
        for upos in users:
            result = _paired_rows_one_user(sp, sp_z, fc, upos, seed)
            _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
    else:
        tasks = [(u, seed) for u in users]
        chunksize = max(1, len(users) // (jobs * 4))
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(sp, sp_z, fc)) as ex:
            for result in ex.map(_paired_rows_task, tasks, chunksize=chunksize):
                _accumulate(result, rows, targets, meta, z_rows, z_targets, z_meta, mus, sigmas)
    raw_frame = _frame(rows, targets, meta)
    if sp_z is not None:
        z_frame = _frame(z_rows, z_targets, z_meta)
        return raw_frame, z_frame, np.asarray(mus, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)
    return raw_frame


def _frame(rows, targets, meta):
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    meta = pd.DataFrame(meta, columns=["user", "tcol", "n", "draw", "n_seen"])
    return features, np.asarray(targets, dtype=np.float32), meta


# ---- app-time construction (from the catalog scores, no Split) -------------
def app_similarity(scores: pd.DataFrame, user: pd.Series, k_shrink: int) -> pd.DataFrame:
    """Per-critic shrunk Pearson alignment from the catalog scores table."""
    overlap = scores[scores["movie_id"].isin(user.index)].copy()
    overlap["user_rating"] = overlap["movie_id"].map(user)
    rows = []
    for critic_id, group in overlap.groupby("critic_id"):
        n = len(group)
        pearson = 0.0
        if (n >= MIN_APP_OVERLAP and group["score_std"].std() > 1e-9
                and group["user_rating"].std() > 1e-9):
            pearson = float(np.corrcoef(group["user_rating"], group["score_std"])[0, 1])
        rows.append((critic_id, n, pearson * min(n, k_shrink) / k_shrink))
    return pd.DataFrame(rows, columns=["critic_id", "overlap", "sim"]).set_index("critic_id")


def app_features(target_scores: pd.DataFrame, matches: pd.DataFrame,
                 critics: pd.DataFrame, user: pd.Series, mf, theme_sim,
                 global_std: float):
    """Build model features for the selected catalog movies. ``mf`` is a
    `movie_features.MovieFacets` (movie_id-keyed), ``theme_sim`` a
    `movie_features.ThemeSimilarity` (see `load_project_theme_similarity`).
    Returns (features_df, movie_ids)."""
    overlap_counts = matches["overlap"].to_numpy()
    positive = overlap_counts[overlap_counts > 0]
    mean_overlap = float(positive.mean()) if len(positive) else 0.0
    max_overlap = float(overlap_counts.max()) if len(overlap_counts) else 0.0
    seen_movie_ids = list(user.index)
    seen_values = user.to_numpy(dtype=float)
    rows, movie_ids = [], []
    for movie_id, group in target_scores.groupby("movie_id"):
        peer = critics.reindex(group["critic_id"])
        values = group["score_std"].to_numpy(dtype=float)
        peer_count = peer["score_count"].to_numpy(dtype=float) - 1
        peer_sum = peer["score_sum"].to_numpy(dtype=float) - values
        peer_mean = np.full(len(group), float(values.mean()), dtype=float)
        np.divide(peer_sum, peer_count, out=peer_mean, where=peer_count > 0)
        sim = group["critic_id"].map(matches["sim"]).fillna(0.0).to_numpy()
        tail = {
            "n_observed": int(len(user)),
            "mean_overlap": mean_overlap, "max_overlap": max_overlap,
            "tomatometer": float(group["tomatometer_score"].iloc[0]),
            "n_reviewers": int(len(group)),
            "dispersion": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "user_mean": float(user.mean())}
        tail.update(facet_tail_from_ids(mf, theme_sim, seen_movie_ids, seen_values,
                                        movie_id, global_std))
        rows.append(main_feature_row(sim, values - peer_mean, tail))
        movie_ids.append(movie_id)
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    return features, movie_ids
