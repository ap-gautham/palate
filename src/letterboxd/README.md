# Letterboxd community-rating project

This package mirrors the Rotten Tomatoes project (`src/rotten_tomatoes/`)
module-for-module — same flat layout, same file names, same split between
`pseudo_users.py` (matrix + similarity + episode protocol) and `features.py`
(the feature contract) — while preserving the important data difference: a
Letterboxd rating is already a direct member score on the integer **1–10**
scale. There is no score parsing, critic publication, or Tomatometer feature.
All three designs adopt the same **similarity-decile-plus-movie-facet
contract** as RT's Design 2/3 (RT's contract minus the Tomatometer), so the
two projects' Design 2/3 pairs are a fair architectural comparison, not just a
shared Design 1 formula. Design 1 also has a second **top-|sim| variant**
(restricted to the 10 largest-|similarity| peers), identical to RT's.

## Contract shared with the Rotten Tomatoes project

| Stage | Rotten Tomatoes | Letterboxd |
|---|---|---|
| Rater | professional critic (pseudo-user) | real Letterboxd member |
| Rating scale | parsed and standardized 0–5 | native 1–10 |
| Eligibility | at least 10 scored films | at least 5 rated films |
| Design 1 | shrunken Pearson neighbourhood + magnitude, plus a top-\|sim\| variant | same formula and variant |
| Design 2 | ~111-feature XGBoost (incl. Tomatometer) | ~110-feature XGBoost (same contract, no Tomatometer) |
| Design 3 | trained residual neural ensemble | trained residual neural ensemble (same architecture) |
| Evaluation | paired/nested seen-history sweep | paired/nested seen-history sweep (same protocol) |
| App catalog | 1,000 popular films, live in browser | 1,000 popular films, live in browser |

**Movie facets.** Both catalogs are joined to the `gsimonx37/letterboxd`
Kaggle metadata dump (genre, theme, studio, director, actor, decade, language,
country) by normalized (title, year) — see `movie_features.py` — giving every
trained design a per-member affinity term for each facet, built leave-target-
out from seen films only. The join matches 94.3% of Letterboxd's full catalog
and 99.8% of the 1,000-film app catalog (LB's own `movie_id` slug already
encodes the year, so the join is more reliable here than on RT).

## Reproduction

Place the Kaggle export files in `data/letterboxd/raw/`, then from the
repository root:

```bash
make lb            # preprocess -> train_xgboost -> train_neural -> analyze
make lb-export     # browser data for web/
```

Or one stage at a time from `src/` (what the make targets run):

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
std per decile (30 columns) — plus a tail of `n_observed`, `mean_overlap`,
`max_overlap`, `n_reviewers`, `dispersion`, `user_mean`, the per-genre /
theme / actor / director affinity blocks (77 columns), and a small numeric
tail (runtime, external rating, facet counts) — 115 columns in total. `features.py` provides both a sparse-matrix path
(`build_data`, `similarity`, `episode_feature_row`,
`generate_rows`/`generate_paired_rows`) used for training and analysis, and a
DataFrame app-path (`app_similarity`, `app_features`) used by the browser
TypeScript port in `web/src/lib/letterboxd/`. `movie_features.py` builds and
caches the gsimonx37 join that feeds the facet columns.

## Design 3: inductive, not transductive

Design 3 is a residual tabular MLP (`network.py`, identical architecture to
RT's `network.py`) trained on the same ~110 engineered
features — **inductive**, so it scores a brand-new member's live ratings in
the browser, unlike a member/movie embedding model (which can only predict
for members it saw during training). It is genuinely trained
(`train_neural.py`, a 3-model ensemble), not a placeholder.

## Evaluation and leakage rules

`letterboxd.analyze` runs the same **paired/nested seen-history sweep** as RT's
`rotten_tomatoes.analyze`: a fixed set of 8 target films per test member (300
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
| B1 global mean | 2.104 | 2.104 | 2.104 | 2.104 | 2.104 | 2.104 |
| B3 mean of all members | 1.654 | 1.654 | 1.654 | 1.654 | 1.654 | 1.654 |
| B4 mean of top-10 similar | 1.710 | 1.716 | 1.680 | 1.671 | 1.654 | 1.614 |
| Design 1 analytic (full) | 1.715 | 1.648 | 1.575 | 1.545 | 1.527 | **1.502** |
| Design 1 analytic (top-\|sim\|) | 1.806 | 1.723 | 1.633 | 1.591 | 1.553 | 1.501 |
| Design 2 XGBoost | 1.611 | 1.586 | 1.548 | 1.521 | 1.480 | **1.417** |
| Design 3 neural net | 1.610 | 1.579 | 1.541 | 1.518 | 1.494 | **1.418** |

Against B3 (quoting each film's mean member rating), the best model cuts
full-history RMSE by **14.3%** (1.417 vs. 1.654); the analytic formula by
9.2%. The top-|sim| variant trails the full-neighbourhood formula at every
finite seen-count, reaching only a statistical dead heat at full history
(1.501 vs. 1.502) — still no positive case for restricting the neighbourhood,
the same honest negative result as on Rotten Tomatoes. **This is not directly
comparable to RT's RMSE** — Letterboxd's target is 1–10, RT's is 0–5, and the
two use different (though structurally parallel) evaluation populations.
Normalizing by rating range
(`results/letterboxd/cross_dataset_comparison.csv`) puts them on the same
footing: RT's normalized RMSE is ~0.157–0.162, Letterboxd's is ~0.157–0.167.
The two are **comparable**: the trained models land at essentially identical
normalized error on both, and the analytic formula normalizes marginally
*better* on RT despite its sparser critic pseudo-user profiles — an honest
result, not a confirmation that more (real) rating data produces a lower
normalized error here, even with the richer movie-facet features added to
both.

## Isolating scale: the z-score track

Each design also has a parallel z-score variant: every member is standardized
to their own all-time mean/std, the model predicts pure deviation (not level),
and the prediction is converted back to the raw 1–10 scale before scoring
(`results/letterboxd/raw_vs_z.csv`, full-history RMSE, both on the raw scale):

| design | raw | z | z − raw |
|---|---:|---:|---:|
| Design 1 analytic (full) | 1.502 | 1.607 | +0.105 |
| Design 1 analytic (top-\|sim\|) | 1.501 | 1.501 | −0.001 |
| Design 2 XGBoost | 1.417 | 1.412 | −0.005 |
| Design 3 neural net | 1.418 | 1.406 | −0.013 |

Unlike RT (where the z-track trails raw at every design), Letterboxd's trained
models (Design 2/3) come out **flat-to-slightly-better** in z-space at full
history, and even the top-|sim| variant is nearly indifferent — removing each
member's own rating level lets a model with spare capacity spend it on the
deviation signal instead of re-learning it. The full-neighbourhood analytic
formula, which has no learned capacity to reallocate, is worse in z-space. See
`results/letterboxd/figures/plotC_raw_vs_z.png` for the full sweep.

## "Movies like this one" suggestions

The app's "films to predict" table has an expandable dropdown per row (green
if you've already rated one of that film's `k_neighbors` nearest content
neighbours, red otherwise) suggesting similar films to rate. Neighbours are
precomputed offline with K-means (`movie_features.top_similar`) over a
standardized vector of genre/decade multi-hot, a catalog-scoped top-k
multi-hot over studio/director/actor/theme/language/country, and numeric
runtime/rating/year/consensus (mean member rating here, in place of RT's
Tomatometer) — identical method to Rotten Tomatoes'.

`k_neighbors` was tuned with a paired-episode sweep (`similar_k_sweep.py`):
6,000 held-out (member, n=10 seen, target) episodes were bucketed by how many
of the target's k nearest neighbours were already in the seen set, for
`k_neighbors` ∈ {10, 20, 30} — the same 6,000 Design-1 predictions are reused
for every k, so only the bucketing differs. Neither dataset shows a clean
beneficial signal: RMSE is flat within noise up to 2 similar films seen, then
rises on buckets thin enough (a handful of episodes) to be unreliable —
rating a target's nearest content neighbours first did not measurably sharpen
its prediction in this sample, on either dataset. Reported honestly as a
negative result: `k_neighbors=20` is kept on *both* projects as a UI choice
(how many suggestions to show), independent of this null predictive-value
finding.
Full numbers: `results/letterboxd/similar_k_sweep.csv`,
`figures/plotD_similar_k_sweep.png`.

## Current artifacts

`results/letterboxd/`:

- `nsweep_summary.csv`, `model_summary.csv`, `cross_dataset_comparison.csv`, `raw_vs_z.csv`, `similar_k_sweep.csv`
- `figures/plotA_rmse_vs_n.png`, `figures/plotC_raw_vs_z.png`, `figures/plotD_similar_k_sweep.png`, `figures/score_distributions.png`
- `xgboost_results.json`, `neural_results.json` + their `*_test_predictions.parquet`
- `models/letterboxd_xgboost.json` (+ `_meta.json`), `models/letterboxd_neural.pt` (+ `_meta.json`)
- `models/letterboxd_xgboost_z.json` (+ `_meta.json`), `models/letterboxd_neural_z.pt` (+ `_meta.json`)

The website's dataset switcher runs both projects fully interactively,
client-side, in the browser (`web/src/lib/letterboxd/`, exported by
`web_export.py`), including both raw and z-score variants of all three
designs.
