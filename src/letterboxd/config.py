"""Paths and safe scale-up settings for the Letterboxd experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "letterboxd" / "raw"
PROCESSED = ROOT / "data" / "letterboxd" / "processed"
RESULTS = ROOT / "results" / "letterboxd"
MODELS = RESULTS / "models"

RATINGS_CSV = RAW / "ratings_export.csv"
MOVIES_CSV = RAW / "movie_data.csv"
USERS_CSV = RAW / "users_export.csv"
RATINGS_PARQUET = PROCESSED / "ratings_1_to_10.parquet"
MOVIES_PARQUET = PROCESSED / "movies.parquet"

RATING_MIN, RATING_MAX = 1.0, 10.0
MIN_USER_RATINGS = 5
# Start with this many eligible members.  The runner only increases the cap
# after the preceding level finishes inside the 30-second budget.
USER_SCALE_STEPS = (100_000, 1_000_000, 10_000_000)
BENCHMARK_SECONDS = 30
SEED = 42
