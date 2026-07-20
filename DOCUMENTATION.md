# Critic-Matched Movie Prediction: Technical Documentation

This document describes the current all-time random-holdout implementation.
[movie_rec_engine_brief.md](movie_rec_engine_brief.md) is historical planning
context; it contains superseded temporal and aggregation proposals.

> **Two parallel projects.** The original sections below document the
> Rotten Tomatoes critic project. The repository now also contains a separate
> Letterboxd member-rating project at `src/letterboxd/`. It uses the same
> three-design vocabulary, held-out-rating discipline, sparse neighbourhood
> idea, and app flow, but it does not mix data, model artifacts, or score
> scales with RT. See [`src/letterboxd/README.md`](src/letterboxd/README.md)
> for the Letterboxd-specific contract, commands, metrics, and caveats.

## Parallel-project comparison

| Concern | Rotten Tomatoes | Letterboxd |
|---|---|---|
| Source | critic reviews | community member ratings |
| Native target | heterogeneous source scores -> 0–5 | native integer 1–10 |
| Processed data | `data/processed/` | `data/letterboxd/processed/` |
| Results | `results/tables/`, `results/models/` | `results/letterboxd/` |
| Design 1 | analytic critic match | analytic member match |
| Design 2 | 38-feature XGBoost | 5-feature XGBoost baseline |
| Design 3 | trained residual MLP ensemble | written but intentionally untrained |
| Interactive UI | Critic Match | Community Match — Letterboxd |

RMSE values must be interpreted within a project because the outcome scales,
sources, and current evaluation episode counts differ. The UI keeps the
workflows visually parallel while making those differences explicit.

## 1. Reproduction and Layout

Each of the three designs lives in its **own self-contained package** under
`src/`: the pseudo-user substrate (`pseudo_users.py`) and, for the learned
models, the feature construction (`features.py`) are duplicated verbatim inside
each design so a reader can study any one design top-to-bottom without jumping
to a shared module. The copies are identical (same file, same seeds), so the
paired test episodes and training features are byte-identical across designs
and the cross-design comparison is valid. Only the constants in `config.py`
(paths, seeds, the evaluation grid) are shared -- constants, not functions.

```text
src/config.py                     Shared paths + constants (no logic)
src/preprocessing/
    parse_scores.py               Parse the six score formats; standardize to 0-5
    build_dataset.py              Ingest, dedup, filter, standardize -> reviews_scored.parquet
    audit.py                      All-time data funnel and score diagnostics
    score_distributions.py        Distribution figures + dispersion table
src/design1_analytic/             SELF-CONTAINED analytic design
    pseudo_users.py               Matrix, similarity+magnitude, paired episodes, partition
    analytic.py                   Shrinkage, movie-mean-centered formula, top-k baseline
    run.py                        k-sweep + paired test + baselines -> tables
    attribution.py                Formula-component decomposition
    predict.py                    App inference (similarity table + formula)
src/design2_xgboost/              SELF-CONTAINED XGBoost design
    pseudo_users.py               (identical substrate copy)
    features.py                   Decile feature contract + generation + app builders
    train.py                      XGBoost training + paired test eval
    predict.py                    App inference
src/design3_neural/               SELF-CONTAINED neural-net design
    pseudo_users.py               (identical substrate copy)
    features.py                   (identical feature copy)
    network.py                    Residual tabular MLP
    train.py                      Ensemble training on the GPU + paired test eval
    predict.py                    App inference
src/comparison/analysis.py        Cross-design figures, summary + dispersion tables
src/app_catalog/export.py         1,000-movie app catalog
app/streamlit_app.py              Star-table rating UI; live formula + XGBoost + neural net
tests/                            Formula, holdout, paired-episode, partition, feature tests
```

Run from `src/` as modules (so package imports resolve) after placing the
Kaggle CSVs in `data/raw/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd src
../.venv/bin/python -m preprocessing.build_dataset
../.venv/bin/python -m preprocessing.audit
../.venv/bin/python -m design1_analytic.run
../.venv/bin/python -m design2_xgboost.train
../.venv/bin/python -m design3_neural.train
../.venv/bin/python -m design1_analytic.attribution
../.venv/bin/python -m app_catalog.export
../.venv/bin/python -m comparison.analysis
cd ..
.venv/bin/python -m unittest discover -s tests -v
```

All sampling uses fixed seeds. The current artifacts use 1,444,963 raw review
rows, 992,954 parsed scores, 3,715 critics with at least 10 scored reviews, and
3,704 pseudo-users with at least 10 distinct scored movies. (The critic floor
was lowered from 20 to 10 to widen the pool.)

## 2. Score Representation

Source scores include fractions, letter grades, percentages, and star ratings.
Each parseable score is mapped to the ordinal set `{0, 1, 2, 3, 4, 5}`. This
standardized raw score is the target for both active models.

The parquet also stores an all-time per-critic z-score:

```text
z[c, i] = (r[c, i] - mu[c]) / sigma[c]
```

- `r[c, i]`: critic `c`'s standardized raw score for movie `i`.
- `mu[c]`: critic `c`'s all-time mean standardized score.
- `sigma[c]`: critic `c`'s all-time score standard deviation.
- `z[c, i]`: score distance from `mu[c]`, measured in units of `sigma[c]`.

Z-scores are diagnostic only. The analytic formula and trained model predict
the raw standardized score because rating magnitude is an explicit input.

## 3. All-Time Random-Holdout Protocol

### 3.1 Pseudo-user identities

The full critic-by-movie matrix is built from every scored review, not a
date-limited subset. After averaging repeat critic/movie reviews, a critic becomes
a pseudo-user if they have at least 10 distinct scored movies. Critic identities
are deterministically partitioned:

| role | pseudo-users | purpose |
|---|---:|---|
| train | 2,592 | model fitting (XGBoost + neural net) |
| validation | 556 | shrinkage and early stopping |
| test | 556 | held-out evaluation of all models |

The trained models never fit on rows whose pseudo-user critic appears in the
validation or test partition. Repeating profiles therefore increases samples
without allowing a model to memorize a pseudo-user identity.

### 3.2 Training episodes and paired test episodes

**Training/validation (unpaired, for variety).** A profile samples `n` distinct
movies as seen ratings, then uniformly selects a distinct remaining movie as
the target (at least three other critics must have rated it). Similarity and
magnitude use only the sampled seen movies; the pseudo-user's target rating is
the label and is excluded from every aggregation. `n` ranges over
`{3, 5, 10, 20, 50, all}` with 32 training and 8 validation profiles each. This
gave 436,533 training and 23,639 validation rows.

**Test (paired and nested, for a clean comparison).** Sampling a fresh random
target per `n` made even the seen-independent baselines wobble column to column.
The test instead fixes the target and the seen order across the whole grid:
for each test pseudo-user with more than 50 movies, 8 target movies are chosen
once; for each target and each of 3 redraws, the remaining movies are placed in
one popularity-weighted order, and the seen set at seen-count `n` is the first
`n` of that order (all of them for `n = all`). A single deterministic routine
(`pseudo_users.iter_paired_episodes`, copied in each design) generates these 41,472 episodes, and Design 1,
XGBoost, and the neural net are all scored on byte-identical
`(user, target, n, draw)` keys. Because the target and seen order are fixed
across `n`, every seen-count and every baseline is scored on the same films, so
the baselines are exactly flat and the model curves are smooth.

For `n = all`, one target is selected first and every other pseudo-user rating
is seen. The protocol treats any movie absent from a user's seen set as out of
sample, so it supports catalog prediction but is not a temporal future-release
forecast: peer reviews for the target movie may come from any date.

### 3.3 Leakage controls

The focused tests verify that a sampled target is never included in its seen
set, that paired seen sets are prefix-nested and deterministic, that critic
partitions are disjoint and complete, and that the analytic
formula's denominator does not accidentally include `mag_sim`.

## 4. Similarity, Magnitude, and Design 1

Let `O[u, c]` be movies both pseudo-user `u` and critic `c` rated among the
sampled seen movies. Pearson correlation is:

```text
rho[u, c] =
  sum over i in O[u, c] of
    (r[u, i] - r_bar[u, O]) * (r[c, i] - r_bar[c, O])
  ----------------------------------------------------------------
  sqrt(sum over i of (r[u, i] - r_bar[u, O])^2) *
  sqrt(sum over i of (r[c, i] - r_bar[c, O])^2)
```

- `r[u, i]`, `r[c, i]`: standardized raw scores on overlap movie `i`.
- `r_bar[u, O]`, `r_bar[c, O]`: user and critic means over `O[u, c]`.
- `rho[u, c]`: Pearson correlation from `-1` to `+1`.
- `sum`: addition over all overlap movies.
- `sqrt`: square root.

The shrunken signed similarity is:

```text
s[u, c] = rho[u, c] * min(n_overlap[u, c], k) / k
```

`n_overlap[u, c]` is the overlap size, `min` takes the smaller input, and
`k = 8` is selected by random-holdout validation. `abs(s[u, c])` is the
nonnegative magnitude of signed similarity.

The rating-magnitude multiplier is:

```text
g[u, c] = sum over i in O[u, c] of r[u, i] * r[c, i]
          -----------------------------------------------
          sum over i in O[u, c] of r[c, i]^2
```

`g[u, c]` is named `mag_sim` in code. It is the least-squares through-origin
scale from critic ratings to user ratings. A relationship
`r[u, i] = 1.25 * r[c, i]` over the overlap produces `g[u, c] = 1.25`.

For held-out target movie `m`, `C[m]` is its other critics and:

```text
r_bar[m] = sum over c in C[m] of r[c, m] / number of critics in C[m]
```

`r_bar[m]` is the unweighted mean target rating for movie `m`, not a career
mean and not the Tomatometer. The requested prediction is:

```text
r_hat[u, m] =
  sum over c in C[m] of
    (abs(s[u, c]) * r_bar[m] + s[u, c] * (r[c, m] - r_bar[m])) * g[u, c]
  -------------------------------------------------------------------------
                    sum over c in C[m] of abs(s[u, c])
```

- `r_hat[u, m]`: predicted 0-5 score.
- `r[c, m]`: critic `c`'s target score.
- `s[u, c]`: shrunken Pearson similarity.
- `g[u, c]`: magnitude multiplier.
- `abs`: absolute value.

Predictions are clipped to `[0, 5]`. A zero denominator falls back to
`r_bar[m]`.

## 5. Baselines and Metrics

- **B1 global mean:** all-time mean standardized score.
- **B2 Tomatometer -> score:** the Tomatometer is the percentage of Fresh
  reviews. A global linear mapping `a * T[m] + b` converts it to the 0-5 score
  scale; `T[m]` is movie `m`'s Tomatometer, `a` is slope, and `b` is intercept.
  Missing Tomatometers fall back to B3.
- **B3 reviewer mean:** `r_bar[m]`.
- **B4 top-10 similar mean:** unweighted average of up to ten critics with the
  largest positive `s[u, c]` values.
- **RMSE:** `sqrt(sum((prediction - label)^2) / N)`, where `N` is the number of
  fake-user episodes. Lower is better.
- **Dispersion:** standard deviation of other critics' target scores for one
  movie. Higher dispersion means more disagreement.
- **Spearman correlation:** within-user rank correlation between predictions
  and held-out ratings.
- **Precision@10:** fraction of a pseudo-user's predicted top 10 targets that
  also occur in their true top 10 held-out targets.

## 6. Design 2: XGBoost

Design 2 is an XGBoost regressor (native API, so no scikit-learn dependency)
trained on the repeated random-holdout rows. For every target reviewer it
computes a leave-one-target peer deviation:

```text
deviation[c, m] = r[c, m] - mean of critic c's other all-time scores
```

Reviewers are sorted by similarity. The model uses mean/count/standard
deviation summaries of `similarity * deviation` across ten ranked deciles,
then appends seen-count, overlap statistics, Tomatometer score, reviewer count,
movie dispersion, genre ID, and user mean. `src/design2_xgboost/features.py`
builds these columns for training and app inference alike.

The saved artifact is `results/models/design2_xgboost.json` (sliced to the
early-stopped best iteration); its metadata records exact feature order and
genre IDs in `results/models/design2_xgboost_meta.json`. A missing Tomatometer
is passed to XGBoost as a missing value (gradient boosting handles it natively);
`genre_id` enters as a plain numeric column (a low-importance feature). Verified
free of target leakage: peer deviations are leave-one-out and the pseudo-user
is excluded from every target aggregation. Full-history training episodes are
included so the model sees the whole seen-count range the paired test spans.
Design 2 and Design 3 generate their features independently from the same
seeds, so both train and are scored on byte-identical rows.

## 6b. Design 3: Neural Network

`src/design3_neural/train.py` trains a residual neural network on features it
generates itself (its own `features.py` copy), on the Apple GPU (MPS) when
available. It uses **all** the same features:
the 37 numeric features (log1p-compressed where they are counts, NaN-imputed
with the train mean for a missing Tomatometer, and standardized on train
statistics) plus a 24-dimensional embedding of the genre ID. The architecture
is an input batch-norm and embedding, a projection to width 512, **six**
pre-activation residual blocks (`Linear -> BatchNorm -> GELU -> Dropout(0.1)`
twice, with a skip; ~3M weights), and a linear head. Training uses AdamW, a
ReduceLROnPlateau schedule, and early stopping; **three independently seeded
networks are averaged** into an ensemble. The whole training set is moved to
the GPU once (per-batch host->device copies otherwise dominate MPS step time).
The artifact is `results/models/design3_mlp.pt` (all three state dicts plus
preprocessing stats). Log compression and full-history training rows are both
required: without them the net extrapolates badly at `n = all`, where
`n_observed` is far outside any finite-`n` training row (trees do not have this
problem).

**Scaling experiment (the honest result).** The network was deliberately pushed
-- wider (384 -> 512), deeper (4 -> 6 blocks), longer training, and 2x the
fake-user training data (16 -> 32 profiles per critic/seen-count, ~437k rows)
-- to look for a double-descent gain. It moved the loss by nothing: the network
(0.807 overall / 0.793 full history) still does not beat XGBoost (0.803 /
0.791), and XGBoost itself was unchanged by the 2x data. On this engineered
tabular feature set the ceiling is the information in the features, not model
capacity; extra parameters slightly hurt if anything. Reported, not tuned away.

## 7. Regenerated Results

All models are scored on the same 41,472 **paired, nested** test episodes
(Section 3), so the baselines are exactly flat in `n` and the curves are
directly comparable row by row.

### 7.1 RMSE by seen-count

| seen ratings `n` | 3 | 5 | 10 | 20 | 50 | all |
|---|---:|---:|---:|---:|---:|---:|
| B1 global mean | 1.054 | 1.054 | 1.054 | 1.054 | 1.054 | 1.054 |
| B2 Tomatometer | 0.841 | 0.841 | 0.841 | 0.841 | 0.841 | 0.841 |
| B3 reviewer mean | 0.827 | 0.827 | 0.827 | 0.827 | 0.827 | 0.827 |
| B4 top-10 similar | 0.891 | 0.892 | 0.877 | 0.865 | 0.859 | 0.849 |
| Design 1 formula | 0.957 | 0.925 | 0.869 | 0.842 | 0.825 | 0.814 |
| Design 2 XGBoost | 0.815 | 0.811 | 0.805 | 0.801 | 0.796 | **0.791** |
| Design 3 neural net | 0.820 | 0.816 | 0.809 | 0.804 | 0.798 | 0.793 |

Overall (all `n` pooled): XGBoost 0.803, neural net 0.807, Design 1 0.873.
The baselines are flat because they do not depend on the seen set; Design 1
falls smoothly and crosses them near `n = 50`. The trained models beat every
baseline at every `n`, and are tied with each other -- even a deeper residual
ensemble trained on the GPU does not beat the trees, the standard result for
gradient boosting on engineered tabular features. Both trained models lean
first on the Tomatometer (about 11x the next feature in XGBoost gain), then
`user_mean`, then the similarity deciles, which is why their loss is low and
only weakly `n`-dependent.

### 7.2 Formula attribution

Attribution over independent full-history episodes (`src/attribution.py`):

| predictor | RMSE |
|---|---:|
| reviewer mean / movie consensus | 0.850 |
| magnitude-scaled movie consensus | **0.833** |
| movie-centered signed similarity, `mag_sim = 1` | 0.867 |
| full Design 1 formula | 0.850 |

Magnitude matching (`mag_sim`) produces the entire gain over the consensus;
adding the signed similarity term on top gives it back. As with the trained
models, correlation-based taste matching contributes little -- the useful
personalization is scale/level calibration.

### 7.3 Dispersion and ranking

Full-history gain over the Tomatometer by critic-disagreement tercile:

| stratum | Design 1 | Design 2 | Design 3 |
|---|---:|---:|---:|
| low | +5.1% | +7.1% | +7.7% |
| mid | +4.6% | +6.1% | +5.4% |
| high | +1.3% | +5.1% | +4.8% |

All three beat the Tomatometer in every tercile; the trained models most,
including on high-disagreement films. Within-user Spearman is 0.558 for
Design 1 vs 0.566 for B3 -- ranking is barely changed, consistent with the
attribution: the win is calibration, not reordering a user's films.

## 8. Streamlit Workflow

The app catalog contains the 1,000 most-reviewed movies (3,387 critics appear
in them). It has three sections, and predictions update live as ratings change:

A sort control at the top orders both search lists **alphabetically by title
(default), by year (newest first), or by most reviewed**.

1. **Films you have seen** -- a searchable add box (it clears after each add)
   populates a table where each row has a five-star `st.feedback` widget
   (click a star, it and all before it fill gold; the rating shows beside it in
   a brightening gold) and a remove button.
2. **Films to predict** -- the same searchable star-table; scoring is optional.
3. **Predictions** -- for every target movie, all three model predictions
   (analytic formula, XGBoost, neural net) beside the movie's consensus mean
   and Tomatometer. When the user scores some target films, the app reports the
   **mean squared error of each method -- and of the consensus mean -- against
   the user's own score**, and marks the closest, answering "which method
   predicts *me* best?".

The two add boxes carry the full catalog minus already-chosen films and use
stable session-state keys, so editing one never resets the other. Each design's
app inference lives in its own package (`design1_analytic/predict.py`,
`design2_xgboost/predict.py`, `design3_neural/predict.py`); the app imports the
three and orchestrates the UI. If no reviewer has nonzero similarity, the
analytic formula returns the target movie mean rather than failing.

XGBoost and PyTorch each bundle their own OpenMP runtime; the app sets
`KMP_DUPLICATE_LIB_OK=TRUE` and pins both to a single thread so they coexist
without the macOS "Error #15" abort or a thread-contention hang. Inference is
tiny and deterministic, so this is safe.

## 9. Verification and Limitations

Verified in this workspace:

- Formula unit tests confirm `mag_sim = 1.25` for an exact `user = 1.25 * critic`
  overlap and verify the specified numerator/denominator.
- Random-holdout tests prove targets are excluded from seen inputs; a paired-
  episode test proves the nested seen sets are prefix-nested and deterministic.
- Partition tests prove train, validation, and test pseudo-user identities are
  disjoint and exhaustive.
- Shared-feature tests lock training and app feature column order.
- Design 1, Design 2, Design 3, attribution, export, analysis, audit, and PDF
  generation all ran from the all-time processed data; the XGBoost low loss
  was checked to be a leakage-free calibrated consensus (per-`n` flat, dominant
  Tomatometer feature, leave-one-out peer means).

Limitations:

- Random item holdout is not temporal forecasting; target peer reviews can
  postdate a pseudo-user's seen ratings.
- Lowering the critic floor to 10 movies widens the pool with noisier
  low-volume critics; they mostly matter only where they overlap a user.
- `mag_sim` is an unregularized through-origin slope and may be volatile on
  sparse or unusual overlaps.
- The neural net matches but does not beat the trees here; on this tabular
  feature set with a strong Tomatometer signal that is expected, and it is
  reported rather than tuned away.
- Scores are quantized to six levels. Pseudo-users are critics, not casual
  users. The Tomatometer is a current snapshot, not a historical value.
