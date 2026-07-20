# Letterboxd community-rating project

This package mirrors the Rotten Tomatoes project while preserving the important
data difference: a Letterboxd rating is already a direct member score on the
integer **1–10** scale. There is no score parsing, critic publication, or
Tomatometer feature.

## Contract shared with the Rotten Tomatoes project

| Stage | Rotten Tomatoes | Letterboxd |
|---|---|---|
| Rater | professional critic | Letterboxd member |
| Rating scale | parsed and standardized 0–5 | native 1–10 |
| Eligibility | at least 10 scored films | at least 5 rated films |
| Design 1 | shrunken Pearson neighbourhood | shrunken Pearson neighbourhood |
| Design 2 | XGBoost | XGBoost |
| Design 3 | residual neural ensemble | architecture written; deliberately untrained |
| App catalog | 1,000 popular films | 1,000 popular films |

## Reproduction

Place the Kaggle export files in `data/letterboxd/raw/`, then run from `src/`:

```bash
../.venv/bin/python -m letterboxd.preprocess --max-users 100000
../.venv/bin/python -m letterboxd.train_analytic --max-test-users 300 --max-seen 50
../.venv/bin/python -m letterboxd.train_xgboost
../.venv/bin/python -m letterboxd.train_neural
```

The preprocessing command records each requested scale in
`data/letterboxd/processed/metadata.json`. The supplied export has fewer than
100,000 eligible users, so 100k, 1M, and 10M requested caps all resolve to the
same complete local matrix; no rows are randomly discarded.

## Evaluation and leakage rules

Both designs with results use a held-out rating as the label. The held-out
rating is removed from that member's seen profile. Design 1 builds signed
Pearson alignment and a magnitude multiplier only from seen films, then applies
the same movie-mean-centred formula used by the Rotten Tomatoes project. Its
evaluation caps a profile at 50 seen films to keep a 11M-row sparse matrix
interactive; the peer matrix is never subsampled.

Design 2 removes one rating per member before forming member and movie
aggregates. Its deterministic member split keeps a member entirely in either
train or test. The current feature contract is intentionally smaller than RT's
38-feature critic contract: `user_mean`, `user_count`, `movie_mean`,
`movie_std`, and `movie_count`. Letterboxd has no Tomatometer or critic
publication/genre-derived calibration feature. This is a clean baseline, not a
claim that the two RMSE values are directly comparable across their different
rating scales and test protocols.

## Current artifacts

The analysis writes only inside `results/letterboxd/`:

- `analytic_results.json`
- `xgboost_results.json`
- `models/letterboxd_xgboost.json`

The Streamlit app exposes both projects through its Project selector. The
website exposes the matching Letterboxd project tab and explicitly labels
Design 3 as untrained.
