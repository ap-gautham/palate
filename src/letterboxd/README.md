# Letterboxd community-rating project

This package mirrors the Rotten Tomatoes project (`src/rotten_tomatoes/`)
directory-for-directory, while preserving the important data difference: a
Letterboxd rating is already a direct member score on the integer **1–10**
scale. There is no score parsing, critic publication, or Tomatometer feature.
All three designs adopt the same **37-feature similarity-decile contract** as
RT's Design 2/3 (RT's 38-feature contract minus the Tomatometer), so the two
projects' Design 2/3 pairs are a fair architectural comparison, not just a
shared Design 1 formula.

## Contract shared with the Rotten Tomatoes project

| Stage | Rotten Tomatoes | Letterboxd |
|---|---|---|
| Rater | professional critic (pseudo-user) | real Letterboxd member |
| Rating scale | parsed and standardized 0–5 | native 1–10 |
| Eligibility | at least 10 scored films | at least 5 rated films |
| Design 1 | shrunken Pearson neighbourhood + magnitude | same formula |
| Design 2 | 38-feature XGBoost (incl. Tomatometer) | 37-feature XGBoost (same contract, no Tomatometer) |
| Design 3 | trained residual neural ensemble | trained residual neural ensemble (same architecture) |
| Evaluation | paired/nested seen-history sweep | paired/nested seen-history sweep (same protocol) |
| App catalog | 1,000 popular films, live in browser + Streamlit | 1,000 popular films, live in browser + Streamlit |

## Reproduction

Place the Kaggle export files in `data/letterboxd/raw/`, then run from `src/`:

```bash
../.venv/bin/python -m letterboxd.preprocess --max-users 100000
../.venv/bin/python -m letterboxd.train_xgboost
../.venv/bin/python -m letterboxd.train_neural
../.venv/bin/python -m letterboxd.analyze
../.venv/bin/python -m letterboxd.web_export
```

The preprocessing command records each requested scale in
`data/letterboxd/processed/metadata.json`. The supplied export has fewer than
100,000 eligible members, so 100k, 1M, and 10M requested caps all resolve to
the same complete local matrix (7,420 members, 11,078,045 ratings, 286,069
films) — no rows are randomly discarded.

## Feature contract (`features.py`)

Per (member profile, target film) episode: ten deciles of `similarity x
(rater score - that rater's leave-one-out all-time mean)` — mean, count, and
std per decile (30 columns) — plus a 7-column tail: `n_observed`,
`mean_overlap`, `max_overlap`, `n_reviewers`, `dispersion`, `genre_id`,
`user_mean`. `features.py` provides both a sparse-matrix path (`build_data`,
`similarity`, `episode_feature_row`, `generate_rows`/`generate_paired_rows`)
used for training and analysis, and a DataFrame app-path (`app_similarity`,
`app_features`) used by the Streamlit app and mirrored by the browser
TypeScript port in `web/src/lib/letterboxd/`.

## Design 3: inductive, not transductive

Design 3 is a residual tabular MLP (`network.py`, identical architecture to
RT's `design3_neural/network.py`) trained on the same 37 engineered features —
**inductive**, so it scores a brand-new member's live ratings in the browser
or Streamlit, unlike a member/movie embedding model (which can only predict
for members it saw during training). It is genuinely trained (`train_neural.py`,
a 3-model ensemble), not a placeholder.

## Evaluation and leakage rules

`letterboxd.analyze` runs the same **paired/nested seen-history sweep** as RT's
`comparison.analysis`: a fixed set of 8 target films per test member (300
deterministic members with more than 50 rated films), 3 popularity-ordered
seen-order redraws, with every seen-count `n` (and every baseline) scored on an
identical (member, target, draw) set. The held-out rating is removed from that
member's own profile and from the target film's peer mean (leave-one-out);
peer means for Design 1/2/3 use each rater's all-time statistics, computed
outside the current episode.

## Results

Full-history RMSE (1–10 scale), from `results/letterboxd/nsweep_summary.csv`:

| method | 3 | 5 | 10 | 20 | 50 | all |
|---|---:|---:|---:|---:|---:|---:|
| B1 global mean | 2.108 | 2.108 | 2.108 | 2.108 | 2.108 | 2.108 |
| B3 mean of all members | 1.639 | 1.639 | 1.639 | 1.639 | 1.639 | 1.639 |
| B4 mean of top-10 similar | 1.705 | 1.686 | 1.671 | 1.653 | 1.657 | 1.622 |
| Design 1 analytic | 1.765 | 1.670 | 1.597 | 1.561 | 1.545 | **1.507** |
| Design 2 XGBoost | 1.663 | 1.609 | 1.588 | 1.562 | 1.533 | **1.501** |
| Design 3 neural net | 1.676 | 1.620 | 1.593 | 1.561 | 1.543 | **1.515** |

**This is not directly comparable to RT's RMSE** — Letterboxd's target is 1–10,
RT's is 0–5, and the two use different (though structurally parallel)
evaluation populations. Normalizing by rating range
(`results/letterboxd/cross_dataset_comparison.csv`) puts them on the same
footing: RT's normalized RMSE is ~0.158–0.163, Letterboxd's is ~0.167–0.168.
The two are **comparable**, with RT marginally *better* normalized despite its
sparser critic pseudo-user profiles — an honest result, not a confirmation
that more (real) rating data produces a lower normalized error here.

## Current artifacts

`results/letterboxd/`:

- `nsweep_summary.csv`, `model_summary.csv`, `cross_dataset_comparison.csv`
- `figures/plotA_rmse_vs_n.png`, `figures/score_distributions.png`
- `xgboost_results.json`, `neural_results.json` + their `*_test_predictions.parquet`
- `models/letterboxd_xgboost.json` (+ `_meta.json`), `models/letterboxd_neural.pt` (+ `_meta.json`)

The Streamlit app exposes both projects through its Project selector, with all
three Letterboxd designs serving live predictions (identical UI to Rotten
Tomatoes, aside from a 1–10 rating stepper instead of 5 stars and no
Tomatometer column). The website's dataset switcher runs both projects fully
interactively, client-side, in the browser (`web/src/lib/letterboxd/`,
exported by `web_export.py`).
