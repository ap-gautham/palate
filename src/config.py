"""Central paths and constants for the critic-matched prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = RESULTS / "models"

MOVIES_CSV = DATA_RAW / "rotten_tomatoes_movies.csv"
REVIEWS_CSV = DATA_RAW / "rotten_tomatoes_movie_reviews.csv"

REVIEWS_PARQUET = DATA_PROCESSED / "reviews_scored.parquet"
MOVIES_PARQUET = DATA_PROCESSED / "movies.parquet"

# Data-quality floor: drop critics with <= this many scored reviews entirely
# (n > 5 movies retained). Applied globally before any modeling.
MIN_CRITIC_MOVIES = 6

# Eligibility
MIN_HISTORY = 10            # distinct scored movies for an all-time pseudo-user
MIN_OTHER_REVIEWERS = 3     # a target movie needs this many OTHER critics

# Paired-evaluation grid. A held-out target is fixed per pseudo-user, then the
# seen set is a nested popularity-weighted prefix, so every n (and every
# baseline) is scored on an identical (user, target) set -> smooth curves.
N_GRID = [3, 5, 10, 20, 50, None]      # None = all history except the target
N_MAX_FINITE = 50                      # a paired user must own > this many movies
EVAL_TARGETS_PER_USER = 8
EVAL_DRAWS = 3                         # nested seen-order redraws, averaged

# Standardized score scale: all parsed fractions are quantized onto {0..5}
SCORE_LEVELS = 5

# The value every model predicts, aggregates, and is scored on.
#   "score_std" -> raw standardized 0-5 score (keeps each critic's absolute
#                  level: a 3.5-centered critic differs from a 4.5-centered one)
#   "z"         -> per-critic z-score (level-invariant; the earlier approach)
VALUE_COL = "score_std"

SEED = 42
