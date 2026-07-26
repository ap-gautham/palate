"""Design 1 app inference: turn a user's star ratings into analytic-formula
predictions for chosen target films. Self-contained -- computes its own
member-similarity table from the catalog scores (no offline LBData needed).
Mirrors rotten_tomatoes/predict_analytic.py.
"""
import numpy as np
import pandas as pd

MIN_OVERLAP = 2


def member_matches(scores: pd.DataFrame, user: pd.Series,
                   k_shrink: int) -> pd.DataFrame:
    """Shrunk Pearson alignment and magnitude scale for every overlapping member.

    `scores` is the long catalog table (user_id, movie_id, rating, ...);
    `user` maps the user's seen movie_ids to their 1-10 ratings.
    """
    overlap = scores[scores["movie_id"].isin(user.index)].copy()
    overlap["user_rating"] = overlap["movie_id"].map(user)
    rows = []
    for member_id, group in overlap.groupby("user_id"):
        n = len(group)
        pearson, magnitude = 0.0, 1.0
        if (n >= MIN_OVERLAP and group["rating"].std() > 1e-9
                and group["user_rating"].std() > 1e-9):
            pearson = float(np.corrcoef(group["user_rating"], group["rating"])[0, 1])
            member_ratings = group["rating"].to_numpy(dtype=float)
            denom = float(member_ratings @ member_ratings)
            if denom > 1e-12:
                magnitude = float(group["user_rating"].to_numpy() @ member_ratings / denom)
        rows.append((member_id, pearson, n, pearson * min(n, k_shrink) / k_shrink,
                     magnitude))
    return pd.DataFrame(rows, columns=[
        "user_id", "pearson", "overlap", "sim", "mag_sim"]).set_index("user_id")


def predict(target_scores: pd.DataFrame, matches: pd.DataFrame,
           rating_min: float, rating_max: float) -> pd.DataFrame:
    """Movie-mean-centered, magnitude-scaled analytic prediction per target movie.
    Returns a frame indexed by movie_id with columns [prediction, movie_mean]."""
    work = target_scores.copy()
    work["sim"] = work["user_id"].map(matches["sim"]).fillna(0.0)
    work["mag_sim"] = work["user_id"].map(matches["mag_sim"]).fillna(1.0)
    work["movie_mean"] = work.groupby("movie_id")["rating"].transform("mean")
    work["num"] = ((work["sim"].abs() * work["movie_mean"]
                    + work["sim"] * (work["rating"] - work["movie_mean"]))
                   * work["mag_sim"])
    work["den"] = work["sim"].abs()
    agg = work.groupby("movie_id").agg(numerator=("num", "sum"),
                                       denominator=("den", "sum"),
                                       movie_mean=("movie_mean", "first"))
    agg["prediction"] = np.where(agg["denominator"] > 0,
                                 agg["numerator"] / agg["denominator"],
                                 agg["movie_mean"]).clip(rating_min, rating_max)
    return agg[["prediction", "movie_mean"]]


def predict_topk_abs(target_scores: pd.DataFrame, matches: pd.DataFrame,
                     rating_min: float, rating_max: float, k: int = 10) -> pd.DataFrame:
    """A variation of the formula above: restrict each target movie's
    neighbourhood to the ``k`` members with the largest |sim| (aligned or
    anti-aligned), then the identical formula. Mirrors
    web/src/lib/letterboxd/design1.ts:predictAnalyticTop10."""
    work = target_scores.copy()
    work["sim"] = work["user_id"].map(matches["sim"]).fillna(0.0)
    work["mag_sim"] = work["user_id"].map(matches["mag_sim"]).fillna(1.0)

    def per_movie(group: pd.DataFrame) -> pd.Series:
        top = group.reindex(group["sim"].abs().sort_values(ascending=False).index[:k])
        movie_mean = float(top["rating"].mean())
        weight = top["sim"].abs()
        den = float(weight.sum())
        num = float(((weight * movie_mean + top["sim"] * (top["rating"] - movie_mean))
                     * top["mag_sim"]).sum())
        prediction = num / den if den > 0 else movie_mean
        return pd.Series({"prediction": float(np.clip(prediction, rating_min, rating_max)),
                          "movie_mean": movie_mean})

    return work.groupby("movie_id", group_keys=False).apply(per_movie, include_groups=False)
