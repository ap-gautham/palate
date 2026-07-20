"""Design 1 app inference: turn a user's star ratings into analytic-formula
predictions for chosen target films. Self-contained -- computes its own
critic-similarity table from the catalog scores (no offline Split needed).
"""
import numpy as np
import pandas as pd

MIN_OVERLAP = 2


def critic_matches(scores: pd.DataFrame, user: pd.Series,
                   k_shrink: int) -> pd.DataFrame:
    """Shrunk Pearson alignment and magnitude scale for every overlapping critic.

    `scores` is the long catalog table (critic_id, movie_id, score_std, ...);
    `user` maps the user's seen movie_ids to their 1-5 ratings.
    """
    overlap = scores[scores["movie_id"].isin(user.index)].copy()
    overlap["user_rating"] = overlap["movie_id"].map(user)
    rows = []
    for critic_id, group in overlap.groupby("critic_id"):
        n = len(group)
        pearson, magnitude = 0.0, 1.0
        if (n >= MIN_OVERLAP and group["score_std"].std() > 1e-9
                and group["user_rating"].std() > 1e-9):
            pearson = float(np.corrcoef(group["user_rating"], group["score_std"])[0, 1])
            critic_ratings = group["score_std"].to_numpy(dtype=float)
            denom = float(critic_ratings @ critic_ratings)
            if denom > 1e-12:
                magnitude = float(group["user_rating"].to_numpy() @ critic_ratings / denom)
        rows.append((critic_id, pearson, n, pearson * min(n, k_shrink) / k_shrink,
                     magnitude))
    return pd.DataFrame(rows, columns=[
        "critic_id", "pearson", "overlap", "sim", "mag_sim"]).set_index("critic_id")


def predict(target_scores: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Movie-mean-centered, magnitude-scaled analytic prediction per target movie.
    Returns a frame indexed by movie_id with columns [prediction, movie_mean]."""
    work = target_scores.copy()
    work["sim"] = work["critic_id"].map(matches["sim"]).fillna(0.0)
    work["mag_sim"] = work["critic_id"].map(matches["mag_sim"]).fillna(1.0)
    work["movie_mean"] = work.groupby("movie_id")["score_std"].transform("mean")
    work["num"] = ((work["sim"].abs() * work["movie_mean"]
                    + work["sim"] * (work["score_std"] - work["movie_mean"]))
                   * work["mag_sim"])
    work["den"] = work["sim"].abs()
    agg = work.groupby("movie_id").agg(numerator=("num", "sum"),
                                       denominator=("den", "sum"),
                                       movie_mean=("movie_mean", "first"))
    agg["prediction"] = np.where(agg["denominator"] > 0,
                                 agg["numerator"] / agg["denominator"],
                                 agg["movie_mean"]).clip(0, 5)
    return agg[["prediction", "movie_mean"]]
