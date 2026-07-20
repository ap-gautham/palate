"""Ingest CSVs -> scored parquet.

Steps: parse originalScore, dedup critic names, drop very-low-volume critics
(data-quality floor), standardize every score onto a common 0-5 ordinal
scale, attach per-critic z-scores.

Per-critic mu/sigma are computed on all available standardized scores. The
current raw-score models do not use z; it remains a diagnostic representation.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from rotten_tomatoes.config import (DATA_PROCESSED, MIN_CRITIC_MOVIES, MIN_HISTORY, MOVIES_CSV,
                    MOVIES_PARQUET, REVIEWS_CSV, REVIEWS_PARQUET, SCORE_LEVELS,
                    SEED)
from .parse_scores import parse_series, standardize_to_levels


def normalize_name(name: str) -> str:
    """Canonical form for critic-name dedup: casefold, strip accents,
    punctuation, and extra whitespace."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = re.sub(r"[.\-'’]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedup_critics(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge critic-name variants. Returns (df with critic_id, n_merges)."""
    names = df["criticName"].dropna().unique()
    canon = {n: normalize_name(n) for n in names}
    # Representative spelling per canonical form = most frequent variant
    freq = df["criticName"].value_counts()
    rep: dict[str, str] = {}
    for n in sorted(names, key=lambda x: -freq[x]):
        rep.setdefault(canon[n], n)
    n_merges = len(names) - len(rep)
    df = df.copy()
    df["critic_id"] = df["criticName"].map(lambda n: rep[canon[n]] if isinstance(n, str) else np.nan)
    return df, n_merges


def main() -> None:
    print("Loading reviews CSV ...")
    reviews = pd.read_csv(
        REVIEWS_CSV,
        usecols=["id", "reviewId", "creationDate", "criticName", "isTopCritic",
                 "originalScore", "reviewState", "publicatioName"],
        dtype={"originalScore": "string"},
    ).rename(columns={"id": "movie_id", "publicatioName": "publicationName"})
    print(f"  {len(reviews):,} review rows")

    reviews["creationDate"] = pd.to_datetime(reviews["creationDate"], errors="coerce")
    reviews = reviews.dropna(subset=["creationDate", "criticName"])

    rng = np.random.default_rng(SEED)

    print("Parsing originalScore ...")
    reviews["score_frac"] = parse_series(reviews["originalScore"])

    print("Deduplicating critic names ...")
    reviews, n_merges = dedup_critics(reviews)
    print(f"  merged {n_merges} name variants into canonical critics")

    # Data-quality floor: keep only critics with > 5 scored reviews.
    scored_counts = (reviews[reviews["score_frac"].notna()]
                     .groupby("critic_id").size())
    keep_critics = scored_counts[scored_counts >= MIN_CRITIC_MOVIES].index
    n_critics_before = int(scored_counts.size)
    n_rows_before = int(len(reviews))
    reviews = reviews[reviews["critic_id"].isin(keep_critics)].copy()
    print(f"  n>5 filter: kept {len(keep_critics):,}/{n_critics_before:,} critics "
          f"(with a parseable score), {len(reviews):,}/{n_rows_before:,} rows")

    # Standardize every parsed score onto the common {0..5} ordinal scale.
    print(f"Standardizing scores onto a 0-{SCORE_LEVELS} scale (random half-rounding) ...")
    reviews["score_std"] = standardize_to_levels(reviews["score_frac"],
                                                 SCORE_LEVELS, rng)

    # Per-critic diagnostic z-scale from all available standardized scores.
    scored = reviews[reviews["score_std"].notna()]
    stats = scored.groupby("critic_id")["score_std"].agg(["mean", "std", "count"])
    stats = stats[(stats["count"] >= MIN_HISTORY) & (stats["std"] > 1e-6)]
    print(f"  {len(stats):,} critics with >= {MIN_HISTORY} scored reviews and non-degenerate scale")

    reviews["critic_mu"] = reviews["critic_id"].map(stats["mean"])
    reviews["critic_sigma"] = reviews["critic_id"].map(stats["std"])
    reviews["z"] = (reviews["score_std"] - reviews["critic_mu"]) / reviews["critic_sigma"]
    reviews["year"] = reviews["creationDate"].dt.year

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(REVIEWS_PARQUET, index=False)
    print(f"Wrote {REVIEWS_PARQUET} ({len(reviews):,} rows)")

    movies = pd.read_csv(MOVIES_CSV, usecols=["id", "title", "audienceScore", "tomatoMeter",
                                              "genre", "releaseDateTheaters"])
    movies = movies.rename(columns={"id": "movie_id"})
    movies.to_parquet(MOVIES_PARQUET, index=False)
    print(f"Wrote {MOVIES_PARQUET} ({len(movies):,} rows)")

    # Persist the merge count and filter stats for the audit
    (DATA_PROCESSED / "dedup_merges.txt").write_text(str(n_merges))
    (DATA_PROCESSED / "filter_stats.json").write_text(json.dumps({
        "min_critic_movies": MIN_CRITIC_MOVIES,
        "score_levels": SCORE_LEVELS,
        "n_critics_with_score_before_filter": n_critics_before,
        "n_critics_after_filter": int(len(keep_critics)),
        "n_critics_dropped": int(n_critics_before - len(keep_critics)),
        "n_rows_before_filter": n_rows_before,
        "n_rows_after_filter": int(len(reviews)),
    }, indent=2))


if __name__ == "__main__":
    main()
