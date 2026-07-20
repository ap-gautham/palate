"""Design 2/3 feature contract and construction.

The learned models never see the critic matrix directly; they see, per
(pseudo-user, target movie) episode, a fixed-width feature row: reviewers of the
target sorted by similarity and summarized in ten deciles of
`similarity x (critic score - that critic's leave-one-out mean)`, plus a tail of
seen count, overlap statistics, mapped Tomatometer, reviewer count, dispersion,
genre, and the user's mean rating.

This module is self-contained within the design package and is duplicated
verbatim in design3_neural so the two learned models are trained and scored on
identical features (a fair comparison). It contains three layers: the pure
contract (`main_feature_row`), offline generation from a Split
(`generate_rows`, `generate_paired_rows`), and app-time construction
(`app_similarity`, `app_features`).
"""
import numpy as np
import pandas as pd

from config import MIN_OTHER_REVIEWERS
from .pseudo_users import (iter_paired_episodes, sample_random_holdout,
                           similarity, target_ok_mask)

MIN_APP_OVERLAP = 2
DECILES = 10
MAIN_COLS = ([f"d{i}_mean" for i in range(DECILES)]
             + [f"d{i}_cnt" for i in range(DECILES)]
             + [f"d{i}_std" for i in range(DECILES)])
TAIL_COLS = ["n_observed", "mean_overlap", "max_overlap", "tomatometer",
             "n_reviewers", "dispersion", "genre_id", "user_mean"]
FEATURE_COLS = MAIN_COLS + TAIL_COLS


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
                        genre_of_movie, unknown_genre_id):
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
        "genre_id": int(genre_of_movie.get(sp.tgt_movie_index[target_col], unknown_genre_id)),
        "user_mean": float(seen_values.mean()),
    }
    return main_feature_row(sim[critics], target_deviations(sp, critics, values), tail)


def generate_rows(sp, users, rng, genre_of_movie, n_grid, profiles_per_n,
                  unknown_genre_id):
    """Training/validation rows: many unpaired random profiles per critic."""
    rows, targets, meta = [], [], []
    for i, upos in enumerate(users):
        upos = int(upos)
        target_ok = target_ok_mask(sp, upos)
        for n in n_grid:
            for profile in range(profiles_per_n):
                episode = sample_random_holdout(rng, sp, upos, n, target_ok)
                if episode is None:
                    continue
                seen_cols, seen_values, target_col, target_value = episode
                row = episode_feature_row(sp, upos, seen_cols, seen_values,
                                          target_col, genre_of_movie, unknown_genre_id)
                if row is None:
                    continue
                rows.append(row)
                targets.append(target_value)
                meta.append((upos, target_col, -1 if n is None else n, profile,
                             len(seen_cols)))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(users)} critics")
    return _frame(rows, targets, meta)


def generate_paired_rows(sp, users, genre_of_movie, unknown_genre_id):
    """Test rows on the shared paired, nested episodes (identical keys across
    all designs)."""
    rows, targets, meta = [], [], []
    for (upos, target_col, target_value, n, draw,
         seen_cols, seen_values) in iter_paired_episodes(sp, users):
        row = episode_feature_row(sp, upos, seen_cols, seen_values, target_col,
                                  genre_of_movie, unknown_genre_id)
        if row is None:
            continue
        rows.append(row)
        targets.append(target_value)
        meta.append((upos, target_col, n, draw, len(seen_cols)))
    return _frame(rows, targets, meta)


def _frame(rows, targets, meta):
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    if len(features):
        features["genre_id"] = features["genre_id"].astype(int)
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
                 critics: pd.DataFrame, user: pd.Series):
    """Build model features for the selected catalog movies. Returns
    (features_df, movie_ids)."""
    overlap_counts = matches["overlap"].to_numpy()
    positive = overlap_counts[overlap_counts > 0]
    mean_overlap = float(positive.mean()) if len(positive) else 0.0
    max_overlap = float(overlap_counts.max()) if len(overlap_counts) else 0.0
    rows, movie_ids = [], []
    for movie_id, group in target_scores.groupby("movie_id"):
        peer = critics.reindex(group["critic_id"])
        values = group["score_std"].to_numpy(dtype=float)
        peer_count = peer["score_count"].to_numpy(dtype=float) - 1
        peer_sum = peer["score_sum"].to_numpy(dtype=float) - values
        peer_mean = np.full(len(group), float(values.mean()), dtype=float)
        np.divide(peer_sum, peer_count, out=peer_mean, where=peer_count > 0)
        sim = group["critic_id"].map(matches["sim"]).fillna(0.0).to_numpy()
        rows.append(main_feature_row(sim, values - peer_mean, {
            "n_observed": int(len(user)),
            "mean_overlap": mean_overlap, "max_overlap": max_overlap,
            "tomatometer": float(group["tomatometer_score"].iloc[0]),
            "n_reviewers": int(len(group)),
            "dispersion": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "genre_id": int(group["genre_id"].iloc[0]),
            "user_mean": float(user.mean())}))
        movie_ids.append(movie_id)
    features = pd.DataFrame(rows, columns=FEATURE_COLS)
    features["genre_id"] = features["genre_id"].astype(int)
    return features, movie_ids
