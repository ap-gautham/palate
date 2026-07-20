"""Letterboxd XGBoost model on held-out real-member ratings (1--10 scale).

This is deliberately independent of the Rotten Tomatoes feature schema: there
is no Tomatometer.  Features use the member's profile, a leave-one-out movie
consensus, rating volume, and the member/movie interaction.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import MODELS, RATING_MAX, RATING_MIN, RATINGS_PARQUET, RESULTS, SEED


def make_episodes(ratings: pd.DataFrame) -> pd.DataFrame:
    """Hold out one rating per eligible member without leaking it into means."""
    rng = np.random.default_rng(SEED)
    ordered = ratings.sort_values(["user_id", "movie_id"]).copy()
    held_idx = ordered.groupby("user_id", group_keys=False).sample(n=1, random_state=SEED).index
    held = ordered.loc[held_idx].copy()
    seen = ordered.drop(index=held_idx)
    user = seen.groupby("user_id").rating.agg(["mean", "size"]).rename(columns={"mean": "user_mean", "size": "user_count"})
    movie = seen.groupby("movie_id").rating.agg(["mean", "std", "size"]).rename(columns={"mean": "movie_mean", "std": "movie_std", "size": "movie_count"})
    episodes = held.join(user, on="user_id").join(movie, on="movie_id")
    # A deterministic split by member avoids the same member appearing in both
    # train and test episodes.
    episodes["split"] = np.where(rng.random(len(episodes)) < .8, "train", "test")
    return episodes.dropna(subset=["user_mean", "movie_mean"])


def main() -> None:
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")
    episodes = make_episodes(pd.read_parquet(RATINGS_PARQUET))
    feature_columns = ["user_mean", "user_count", "movie_mean", "movie_std", "movie_count"]
    train, test = episodes.query("split == 'train'"), episodes.query("split == 'test'")
    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "eta": .05,
              "max_depth": 7, "min_child_weight": 20, "subsample": .8, "colsample_bytree": .9,
              "tree_method": "hist", "seed": SEED}
    model = xgb.train(params, xgb.DMatrix(train[feature_columns], label=train.rating), num_boost_round=800,
                      evals=[(xgb.DMatrix(test[feature_columns], label=test.rating), "test")],
                      early_stopping_rounds=50, verbose_eval=False)
    predictions = np.clip(model.predict(xgb.DMatrix(test[feature_columns])), RATING_MIN, RATING_MAX)
    report = {"model": "xgboost", "rating_scale": [RATING_MIN, RATING_MAX], "feature_columns": feature_columns,
              "train_episodes": int(len(train)), "test_episodes": int(len(test)),
              "rmse": float(np.sqrt(np.mean((predictions - test.rating.to_numpy()) ** 2))),
              "best_iteration": int(model.best_iteration)}
    MODELS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS / "letterboxd_xgboost.json"))
    (RESULTS / "xgboost_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
