"""Build a clean 1--10 Letterboxd rating matrix from the Kaggle export.

The Kaggle export has changed column spellings over time, so this loader
detects the user, movie, and rating fields rather than assuming a fixed schema.
It streams the ratings CSV, keeps only numeric ratings, converts Letterboxd's
half-star representation to 1--10, and removes members with fewer than five
ratings.  No Rotten Tomatoes file is read here.

Run from ``src`` after downloading the three Kaggle files::

    python -m letterboxd.preprocess
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BENCHMARK_SECONDS, MIN_USER_RATINGS, MOVIES_CSV,
                     MOVIES_PARQUET, PROCESSED, RATING_MAX, RATING_MIN,
                     RATINGS_CSV, RATINGS_PARQUET, SEED, USER_SCALE_STEPS)


def column(columns: list[str], *names: str) -> str:
    def normalized(value: str) -> str:
        return "".join(c for c in value.casefold() if c.isalnum())

    lookup = {normalized(str(c)): c for c in columns}
    for name in names:
        found = lookup.get(normalized(name))
        if found is not None:
            return found
    raise ValueError(f"Expected one of {names}; found columns: {columns}")


def rating_to_ten(values: pd.Series, source_max: float) -> pd.Series:
    """Accept either 0.5--5 stars or the export's 1--10 numeric score."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric * 2 if source_max <= 5 else numeric


def load_ratings(max_users: int | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    if not RATINGS_CSV.exists():
        raise FileNotFoundError(f"Download ratings_export.csv to {RATINGS_CSV}")
    header = pd.read_csv(RATINGS_CSV, nrows=0)
    user_col = column(list(header), "user_id", "userid", "user", "username")
    movie_col = column(list(header), "movie_id", "movieid", "film_id", "filmid", "movie")
    rating_col = column(list(header), "rating", "rating_val", "ratingvalue", "score")
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(RATINGS_CSV, usecols=[user_col, movie_col, rating_col], chunksize=chunksize):
        chunk = chunk.rename(columns={user_col: "user_id", movie_col: "movie_id", rating_col: "rating_raw"})
        chunk["rating"] = pd.to_numeric(chunk.pop("rating_raw"), errors="coerce")
        chunk = chunk.dropna(subset=["user_id", "movie_id", "rating"])
        chunk = chunk[chunk["rating"].between(RATING_MIN, RATING_MAX)]
        rows.append(chunk)
    ratings = pd.concat(rows, ignore_index=True)
    # One value per member/film pair; retaining the last row accommodates a
    # re-rating in a repeated export.
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    ratings["rating"] = rating_to_ten(ratings["rating"], float(ratings["rating"].max()))
    ratings = ratings[ratings["rating"].between(RATING_MIN, RATING_MAX)]
    eligible = ratings.groupby("user_id").size()
    eligible = eligible[eligible >= MIN_USER_RATINGS].sort_index()
    if max_users is not None:
        eligible = eligible.iloc[:max_users]
    return ratings[ratings["user_id"].isin(eligible.index)].copy()


def load_movies() -> pd.DataFrame:
    if not MOVIES_CSV.exists():
        return pd.DataFrame(columns=["movie_id", "title", "year", "genres"])
    header = pd.read_csv(MOVIES_CSV, nrows=0)
    movie_col = column(list(header), "movie_id", "movieid", "id", "film_id")
    title_col = column(list(header), "title", "name", "film_name", "movie_name", "movie_title")
    year_col = next((c for c in header if str(c).casefold()
                     in {"year", "release_year", "releaseyear", "year_released", "yearreleased"}), None)
    genre_col = next((c for c in header if "genre" in str(c).casefold()), None)
    # The Kaggle movie export contains unusually long overview text; Python's
    # CSV parser avoids the C parser's buffer-overflow failure on this file.
    selected = [movie_col, title_col] + ([year_col] if year_col else []) + ([genre_col] if genre_col else [])
    raw = pd.read_csv(MOVIES_CSV, usecols=selected, engine="python")
    out = pd.DataFrame({"movie_id": raw[movie_col], "title": raw[title_col]})
    out["year"] = pd.to_numeric(raw[year_col], errors="coerce") if year_col else np.nan
    out["genres"] = raw[genre_col].fillna("") if genre_col else ""
    return out.drop_duplicates("movie_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=USER_SCALE_STEPS[0])
    args = parser.parse_args()
    started = time.monotonic()
    ratings = load_ratings(args.max_users)
    movies = load_movies()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(RATINGS_PARQUET, index=False)
    movies.to_parquet(MOVIES_PARQUET, index=False)
    elapsed = time.monotonic() - started
    metadata = {"rating_scale": [RATING_MIN, RATING_MAX], "min_user_ratings": MIN_USER_RATINGS,
                "max_users_requested": args.max_users, "users_written": int(ratings.user_id.nunique()),
                "ratings_written": int(len(ratings)), "elapsed_seconds": elapsed, "seed": SEED,
                "next_scale": next((n for n in USER_SCALE_STEPS if n > args.max_users), None)
                if elapsed <= BENCHMARK_SECONDS else None}
    (PROCESSED / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    if elapsed <= BENCHMARK_SECONDS and metadata["next_scale"]:
        print(f"Completed within {BENCHMARK_SECONDS}s; next safe benchmark: --max-users {metadata['next_scale']}")
    elif elapsed > BENCHMARK_SECONDS:
        print("Stopped scale-up: this level exceeded the 30-second budget.")


if __name__ == "__main__":
    main()
