"""Export model-compatible assets for the Streamlit catalog and predictions."""
import json

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MODELS, MOVIES_PARQUET, REVIEWS_PARQUET

N_MOVIES = 1_000
MODEL_META_FILE = MODELS / "design2_xgboost_meta.json"


def main() -> None:
    if not MODEL_META_FILE.exists():
        raise FileNotFoundError(
            f"Run design2_xgboost.train before exporting the catalog: {MODEL_META_FILE}")
    model_meta = json.loads(MODEL_META_FILE.read_text())
    genre_to_id = model_meta["genre_to_id"]
    unknown_genre_id = int(model_meta["unknown_genre_id"])

    reviews = pd.read_parquet(REVIEWS_PARQUET)
    reviews = reviews[reviews["score_std"].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET).drop_duplicates("movie_id")

    # Match the model matrix convention: one averaged score per critic/movie.
    scores = (reviews.groupby(["critic_id", "movie_id"], as_index=False)["score_std"]
              .mean())
    top_movies = scores.groupby("movie_id").size().nlargest(N_MOVIES).index
    catalog_scores = scores[scores["movie_id"].isin(top_movies)]

    movie_meta = movies.set_index("movie_id")
    genres = movie_meta["genre"].fillna("").str.split(",").str[0].str.strip()
    movie_genre_id = genres.map(genre_to_id).fillna(unknown_genre_id).astype(int)

    # Use the same all-time Tomatometer-to-score calibration as Design 2.
    movie_mean = scores.groupby("movie_id")["score_std"].agg(["mean", "size"])
    fit_df = movie_mean[movie_mean["size"] >= 5].join(
        movie_meta["tomatoMeter"], how="inner").dropna(subset=["tomatoMeter"])
    slope, intercept = np.polyfit(fit_df["tomatoMeter"].astype(float),
                                  fit_df["mean"], 1)

    out = catalog_scores.merge(
        movies[["movie_id", "title", "releaseDateTheaters", "tomatoMeter"]],
        on="movie_id", how="left")
    out["year"] = pd.to_datetime(out["releaseDateTheaters"], errors="coerce").dt.year
    out["genre_id"] = out["movie_id"].map(movie_genre_id).fillna(unknown_genre_id).astype(int)
    out["tomatometer_score"] = slope * out["tomatoMeter"].astype(float) + intercept
    out = out.drop(columns=["releaseDateTheaters"])
    out.to_parquet(DATA_PROCESSED / "demo_scores.parquet", index=False)

    publication = (reviews.groupby("critic_id")["publicationName"]
                   .agg(lambda values: values.mode().iat[0] if len(values.mode()) else ""))
    critic_count = scores.groupby("critic_id")["score_std"].size().rename("score_count")
    critic_sum = scores.groupby("critic_id")["score_std"].sum().rename("score_sum")
    critics = pd.concat([publication, critic_count, critic_sum], axis=1).reset_index()
    critics["n_reviews"] = critics["score_count"]
    critics.to_parquet(DATA_PROCESSED / "demo_critics.parquet", index=False)

    print(f"demo assets: {out['movie_id'].nunique():,} movies, "
          f"{out['critic_id'].nunique():,} critics, {len(out):,} scores")


if __name__ == "__main__":
    main()