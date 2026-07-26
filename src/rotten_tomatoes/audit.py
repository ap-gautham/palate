"""Data audit — run before any modeling commitment (brief section: Data audit).

Reports the funnel from raw rows to the all-time usable matrix:
  1. parse rate of originalScore
    2. critics with >=20 parseable scored reviews across all available years
    3. movies with >=5 / >=20 critic scores across all available years
    4. median pairwise overlap between randomly drawn eligible critics
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from rotten_tomatoes.config import DATA_PROCESSED, MIN_HISTORY, REVIEWS_PARQUET, VALUE_COL, SEED, TABLES


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(REVIEWS_PARQUET)
    audit: dict = {}

    n_rows = len(df)
    n_with_string = df["originalScore"].notna().sum()
    n_parsed = df["score_frac"].notna().sum()
    audit["total_review_rows"] = int(n_rows)
    audit["rows_with_originalScore"] = int(n_with_string)
    audit["rows_parsed"] = int(n_parsed)
    audit["parse_rate_of_nonnull"] = round(float(n_parsed / n_with_string), 4)
    audit["parse_rate_of_all_rows"] = round(float(n_parsed / n_rows), 4)
    audit["dedup_merges"] = int((DATA_PROCESSED / "dedup_merges.txt").read_text())
    fs = DATA_PROCESSED / "filter_stats.json"
    if fs.exists():
        audit["data_quality_filter"] = json.loads(fs.read_text())

    scored = df[df[VALUE_COL].notna()]
    counts = scored.groupby("critic_id").size()
    eligible = counts[counts >= MIN_HISTORY]
    audit["critics_ge20_all_time"] = int(len(eligible))
    movie_counts = scored.groupby("movie_id").size()
    audit["movies_all_time_ge20_scores"] = int((movie_counts >= 20).sum())
    audit["movies_all_time_ge5_scores"] = int((movie_counts >= 5).sum())
    audit["scored_reviews_all_time"] = int(len(scored))

    audit["reviews_by_year_tail"] = {int(k): int(v) for k, v in
                                     scored["year"].value_counts().sort_index().tail(6).items()}

    # Median pairwise overlap among all-time eligible critics.
    pool = scored[scored["critic_id"].isin(eligible.index)]
    critics = pool["critic_id"].astype("category")
    movies = pool["movie_id"].astype("category")
    M = sparse.csr_matrix(
        (np.ones(len(pool)), (critics.cat.codes, movies.cat.codes)),
        shape=(len(critics.cat.categories), len(movies.cat.categories)),
    )
    M.data[:] = 1.0
    n_crit = M.shape[0]
    idx_a = rng.integers(0, n_crit, 4000)
    idx_b = rng.integers(0, n_crit, 4000)
    keep = idx_a != idx_b
    overlaps = np.asarray(M[idx_a[keep]].multiply(M[idx_b[keep]]).sum(axis=1)).ravel()
    audit["median_pairwise_overlap_eligible"] = float(np.median(overlaps))
    audit["mean_pairwise_overlap_eligible"] = round(float(overlaps.mean()), 2)
    audit["p90_pairwise_overlap_eligible"] = float(np.percentile(overlaps, 90))
    audit["frac_pairs_overlap_ge3"] = round(float((overlaps >= 3).mean()), 4)

    # Score-distribution diagnostics (histograms + dispersion-threshold table)
    from . import score_distributions
    audit["score_distribution"] = score_distributions.main()

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "audit.json"
    out.write_text(json.dumps(audit, indent=2))
    print("\n" + json.dumps({k: v for k, v in audit.items()
                             if k != "score_distribution"}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
