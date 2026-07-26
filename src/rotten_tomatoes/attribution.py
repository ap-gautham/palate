"""Design 1 attribution: decompose the analytic formula into its pieces to see
what actually helps -- the movie consensus, the magnitude scaling, or the
signed similarity deviation.

Run from src/:  python -m rotten_tomatoes.attribution
"""
import json

import numpy as np
import pandas as pd

from rotten_tomatoes.config import MOVIES_PARQUET, REVIEWS_PARQUET, SEED, TABLES, VALUE_COL
from .analytic import predict_movies, shrink
from rotten_tomatoes.pseudo_users import (build_split, partition_pseudo_users, rmse,
                           sample_random_holdout, similarity, target_ok_mask)

PROFILES_PER_TEST_CRITIC = 8


def main() -> None:
    rng = np.random.default_rng(SEED + 99)
    scored = pd.read_parquet(REVIEWS_PARQUET)
    scored = scored[scored[VALUE_COL].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET)
    split = build_split(scored, movies)
    partitions = partition_pseudo_users(split)
    k_star = int(json.loads((TABLES / "k_star.json").read_text())["k_star"])

    errors = {name: [] for name in
              ["reviewer", "magnitude_anchor", "similarity_only", "design1"]}
    for upos in partitions["test"]:
        upos = int(upos)
        target_ok = target_ok_mask(split, upos)
        for _ in range(PROFILES_PER_TEST_CRITIC):
            episode = sample_random_holdout(rng, split, upos, None, target_ok)
            if episode is None:
                continue
            seen_cols, seen_values, target_col, target_value = episode
            sim, overlap, mag_sim = similarity(split, upos, seen_cols, seen_values)
            shrunk = shrink(sim, overlap, k_star)
            target = np.array([target_col])
            pred, denominator, reviewer_mean, _ = predict_movies(
                split, upos, shrunk, mag_sim, target)
            similarity_only, _, _, _ = predict_movies(
                split, upos, shrunk, np.ones_like(mag_sim), target)

            magnitude_weights = np.abs(shrunk) * mag_sim
            magnitude_sum = np.asarray(split.TTmask[target] @ magnitude_weights).ravel()
            with np.errstate(invalid="ignore", divide="ignore"):
                magnitude_anchor = np.clip(
                    reviewer_mean * magnitude_sum / denominator, 0, 5)

            full = np.where(denominator > 0, pred, reviewer_mean)
            magnitude_anchor = np.where(denominator > 0, magnitude_anchor, reviewer_mean)
            similarity_only = np.where(denominator > 0, similarity_only, reviewer_mean)
            errors["reviewer"].append(float(reviewer_mean[0] - target_value))
            errors["magnitude_anchor"].append(float(magnitude_anchor[0] - target_value))
            errors["similarity_only"].append(float(similarity_only[0] - target_value))
            errors["design1"].append(float(full[0] - target_value))

    labels = {
        "reviewer": "1. reviewer mean / movie consensus (B3)",
        "magnitude_anchor": "2. magnitude-scaled movie consensus",
        "similarity_only": "3. movie-centered signed similarity (mag_sim = 1)",
        "design1": "4. Design 1 (movie mean + signed deviation + mag_sim)",
    }
    output = pd.DataFrame([
        {"predictor": labels[name], "rmse": rmse(np.asarray(values)),
         "n_episodes": len(values), "protocol": "all_time_random_holdout"}
        for name, values in errors.items()])
    output.to_csv(TABLES / "attribution.csv", index=False)
    print(output.round(4).to_string(index=False))
    print(f"\nmagnitude-anchor gain (B3 -> magnitude anchor): "
          f"{output.iloc[0].rmse - output.iloc[1].rmse:+.4f}")
    print(f"signed-deviation gain (magnitude anchor -> D1): "
          f"{output.iloc[1].rmse - output.iloc[3].rmse:+.4f}")


if __name__ == "__main__":
    main()
