# Critic-Matched Movie Prediction — Portfolio Project Brief

> **Historical planning brief.** This document preserves the project's initial
> design and contains superseded z-score and aggregation proposals. The current
> implementation, formula definitions, and regenerated results are in
> [README.md](README.md) and [DOCUMENTATION.md](DOCUMENTATION.md).

**One-line framing (use this, not "movie recommender"):** Estimating a correlation from 3–10 overlapping observations and shrinking it toward a prior by an empirically calibrated factor, then testing whether the resulting personalized prediction beats a flat aggregate on held-out future data.

## Thesis

Rotten Tomatoes' flat Tomatometer flattens individual taste, most damagingly in the high-dispersion band where critics disagree. If a user is matched to the specific critics whose scores correlate with theirs, a similarity-weighted prediction should beat the aggregate — but only above some minimum number of user ratings, and only on films where critics actually disagree.

The project's deliverable is not the recommender. It is the **measurement of where that claim is true and where it fails.**

## Data

**Single source:** `andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews` (Kaggle). ~1.4M critic reviews through early 2023.

Relevant fields:
- `movies`: id, title, tomatoMeter, audienceScore, genre, releaseDateTheaters
- `reviews`: reviewId, movieId, criticName, publicationName, isTopCritic, originalScore, reviewState, creationDate, reviewText

No scraping. No Metacritic. No TMDB. Static local dataset for the entire project.

### Data audit — do this first, before any design commitment

Run the counts before writing modeling code. Nominal size is misleading after filtering:

1. Parse rate of `originalScore` (expect ~50% missing/unparseable)
2. Critics with ≥20 parseable scored reviews pre-2022
3. Of those, how many have ≥5 scored reviews in 2023
4. Movies in 2023 with ≥20 critic scores
5. Median pairwise overlap between two randomly drawn eligible critics

If (3) yields only a few hundred critics, the experiment resizes and the n-sweep bins need widening. Report these numbers in the README — the funnel from 1.4M rows to the usable matrix is itself a credibility signal.

## Preprocessing

**Score parsing.** Regex the common formats (`X/Y`, `X.X/Y`, letter grades via lookup, raw percentages). ~6 patterns cover the large majority. Store the raw string AND the parsed fraction separately so a parser improvement doesn't require a full re-ingest.

**Per-critic z-score.** For each critic with ≥20 scored reviews: `z = (x − μ_critic) / σ_critic` over their full history. Fixes scale granularity (a 3/4 is not 60) and harsh/generous baselines in one step. All modeling operates on z, not raw score.

Do NOT naively divide by max. Do NOT impute 75/25 for fresh/rotten rows — the variance is fabricated. Exclude scoreless rows from the numeric pipeline entirely.

**Critic entity resolution.** Critic identity is a name string. Dedup on name + publication clustering BEFORE computing similarity, or similarity vectors fragment across name variants. Log how many merges occurred; it is a defensible preprocessing decision either way.

## Core idea: critics are the users

There are no real users and none are needed. Hold out a critic, treat their rating history as a pseudo-user, mask most of it, and predict the rest from the remaining critic pool. This yields thousands of test subjects with deep histories instead of zero.

**Eligibility for pseudo-user:** ≥20 parseable scored reviews pre-2022 AND ≥5 scored reviews in 2023. Use *all* eligible critics via leave-one-critic-out, not a random handful — per-user variance is large and tight error bars matter.

**Critical: drop the pseudo-user from the critic pool.** Otherwise self-correlation is 1.0 and the result is spectacular and meaningless.

## Design 1 — Heuristic (non-ML) pipeline

### Split semantics (state this explicitly in the README; it looks like leakage and is not)

The temporal cut applies to the **pseudo-user's** ratings, not the critic pool's. At deployment, critic reviews for the target film exist — the user simply hasn't seen the film yet.

- **Similarity from:** pseudo-user's pre-2022 ratings ∩ each critic's pre-2022 ratings
- **Prediction for movie m (2023) uses:** other critics' 2023 ratings of m
- **Held out:** the pseudo-user's own 2023 rating of m

Cutting the critic pool at 2022 would leave zero critic coverage of the test movies.

### Similarity

Pearson correlation on overlapping z-scores. (Pearson, not cosine: cosine on raw scores is distorted by baseline differences, and cosine on mean-centered scores reduces to Pearson anyway.)

**Significance weighting / shrinkage:**
`sim' = sim × min(n_overlap, k) / k`

Do not fix k by intuition. Sweep k ∈ {3, 5, 8, 12, 20, 30} and select on validation — the calibrated value of k is a reportable result, not a hyperparameter to bury.

### Prediction

`predicted_z = Σ(critic_z × sim') / Σ(|sim'|)`

Map back through the pseudo-user's own μ, σ for display on their personal scale. Consider restricting to top-k critics by sim' and sweeping that too.

### The n-sweep — this is the headline experiment

An active critic has 500+ pre-2022 ratings; a real user has ~10. Computing similarity from a full history produces a good number that says nothing about cold start.

Subsample each pseudo-user's observed history to **n ∈ {3, 5, 10, 20, 50, all}**, with ≥5 random draws per n per user, and report the full curve with error bars.

**Sample the truncated history with popularity weighting, not uniformly.** A real user's 10 films are 10 well-known films; uniform sampling from a critic's long tail of festival titles produces an unrealistically low overlap structure.

### Baselines (all four are mandatory)

1. Global mean z (trivially 0 — sanity check)
2. **The movie's Tomatometer, converted to z** ← the baseline that matters
3. Unweighted mean z of all critics who reviewed the film
4. Unweighted mean of the top-k most similar critics (isolates the value of *weighting* vs. *selection*)

Beating (1) is meaningless. Beating (2) is the entire project.

## Design 2 — Learned aggregator

Design 1 hardcodes `Σ(r × s)/Σ(s)`. There is no reason that form is optimal. Design 2 **learns the aggregation function** and is only interesting as a delta against Design 1.

### The raw idea

One row per (pseudo-user, movie). Nothing global, nothing per-user-shaped:

```
X = [Pearson correlation of pseudo-user against all critics,
     critic scores for movie i]
Y =  pseudo-user's score for movie i
```

Pseudo-user construction: sample `n` movies at random from the `N` that critic rated, compute similarity from those `n`, and draw target movie `i` from the remaining `N − n`. Vary `n` across rows so the model sees the full cold-start range.

The critic history matrix (140k × 15k) appears nowhere in X. It is a constant across rows — zero variance, therefore zero information to any model. It is a *resource used to compute features*, not a feature. Correct to drop it.

### Why the raw form doesn't train

Three problems, each with a fix that costs one line of preprocessing:

**Concatenation can't express the interaction.** `[sim, score]` through a linear layer gives `W₁·sim + W₂·score` — additive. The quantity that matters is `sim[j] × score[j]`, a product. Reachable only through nonlinearities, which won't be found in 15,000 dimensions on ~700k rows.

**Fixed critic indexing doesn't generalize.** If column 4,832 always means "critic 4832," the model fits 15,000 near-independent weights, each supported by that critic's handful of films, and can transfer nothing between two behaviorally identical critics.

**Variable-length input.** Different movies have different numbers of reviewing critics; trees and dense nets need fixed width.

### Revised feature construction

1. Compute `sim` from the `n` sampled movies only
2. Take elementwise product `sim ⊙ score` over critics who reviewed movie `i`
3. **Sort by `sim` descending**, bin into deciles
4. Per decile: mean, count, std

```
X = [per-decile mean / count / std                    (30 cols)
     n_observed, mean overlap, max overlap            (user)
     tomatometer, n_critics, critic dispersion, genre (movie)]
Y = pseudo-user's z-score for movie i
```

**Model:** gradient boosting (LightGBM/XGBoost). **Temporal split:** pre-2021 train / 2021–2022 validation / 2022–2023 test.

Remaining headroom over the heuristic: nonlinear discounting of low-overlap similarities, and letting dispersion modulate how much personalization applies at all.

### Approximations imposed — state these in the README

These are hand-designed inductive biases, not neutral preprocessing. Each buys tractability by removing something the model would otherwise have had to learn. All are defensible at this data volume; none are free.

| Approximation | What it assumes | What is given up |
|---|---|---|
| Feed `sim × score` rather than `sim` and `score` separately | Similarity acts multiplicatively on critic score | Any non-multiplicative interaction the model might have discovered |
| Sort critics by similarity; index by rank, not identity | Only a critic's similarity rank matters, not who they are | Critic-specific idiosyncrasy (a critic reliably wrong in a learnable direction) |
| Decile-bin the sorted critics | Within-decile detail is noise | Fine structure at the very top of the ranking — the 1st vs 5th most similar critic collapse into one bin |
| Mean/count/std as the per-bin summary | First two moments capture the bin | Skew, bimodality — a split critical reaction inside one bin |
| Pearson as the similarity metric | Agreement is linear in z-score | Rank-based or nonlinear notions of taste agreement |

The honest framing: this is a **hand-designed approximation of a bilinear model**, with the interaction term supplied rather than learned. Justified because full bilinear `uᵀWc` is 2.1B parameters against ~700k training rows — unlearnable by three orders of magnitude. Stating that ratio in the README is the argument for every row of the table above.

**Ablations worth running** (cheap, and they turn assumptions into measurements):
- Sorted-decile features vs. raw `[sim, score]` concatenation — confirms the interaction term matters
- Deciles vs. top-k explicit slots (k = 5) — tests whether top-of-ranking detail was worth keeping
- Pearson vs. Spearman similarity — tests the linearity assumption

### Leakage guards — report both

- **Similarity/target separation:** `sim` computed from the `n` sampled movies ONLY; target drawn from the held-out `N − n`. Computing similarity over all `N` bakes the answer into the features and produces excellent, meaningless metrics. This is the failure most likely to go unnoticed.
- **Group-wise by pseudo-user:** multiple rows are generated per critic and are correlated. Cross-validate grouped by critic, never by random row split.
- **Temporal:** matches deployment.

If temporal is strong and group-wise collapses, the model memorized individual critics rather than learning an aggregator. Informative negative result; report it.

### Expected outcome

`Σ(sim × score)/Σ(sim)` is already a closed-form answer to this problem. The GBM's job is to beat it via second-order corrections. Run the heuristic first and put its RMSE on the wall.

If the GBM lands within noise of it, that is the result — the analytic weighting was already near-optimal, demonstrated rather than assumed. Report it as such; do not manufacture a difference.

## Metrics

**Primary:** RMSE on held-out z, stratified by critic score dispersion.

Overall RMSE is dominated by consensus films where the Tomatometer is already near-optimal and personalization is definitionally impossible — a small aggregate improvement there is uninterpretable. Bin test movies into dispersion terciles (or explicitly isolate the 70–80% Tomatometer band from the original thesis) and report improvement over baseline (2) **within each stratum**.

**Secondary:** Spearman ρ on within-user ranking; precision@10 for "user's top-rated films."

**Target claim:**
> On high-dispersion films, critic-matched prediction reduces RMSE by X% over the Tomatometer once the user has rated ≥n films. On low-dispersion films it adds nothing — correctly.

The second sentence carries as much weight as the first. A model that knows when not to help is a stronger artifact than one claiming uniform gains.

## Explicitly out of scope for v1

- **Phase 3 NLP / dealbreaker tagging** — separate project; no ground truth to validate against
- **Any scraping or incremental pipeline** — pure ops maintenance, demonstrates nothing about modeling
- **Postgres** — dataset fits in memory; parquet + scipy sparse is the honest choice. Keep the schema diagram in the README as design thinking; skip the deployment
- **Metacritic / TMDB / multi-source ingestion** — score normalization is a feature of this project, not an obstacle to route around
- **Surprise library** — unmaintained, and SVD learns latent factors rather than the interpretable user↔critic similarity that is the actual product
- **Matrix factorization / LightFM** — possible Phase 4 blend, not a starting point

## Stack

Python, pandas/polars, scipy.sparse, numpy, LightGBM (Design 2 only), Streamlit (demo). Parquet on disk. No database, no ORM, no orchestrator.

Phases 1–2 are on the order of 50 lines of sparse matrix code: build the critic × movie z-matrix once, correlate the user vector against it. The eval harness is the bulk of the work.

## Build order

1. Load, parse scores, z-score, dedup critics — **run the data audit and publish the funnel**
2. Build pseudo-user population (eligibility filter, leave-one-critic-out scaffolding)
3. Baselines 1–4 wired to the eval harness before any similarity code exists
4. Design 1 similarity + shrinkage; sweep k
5. n-sweep with popularity-weighted truncation → **headline plot**
6. Dispersion-stratified results table
7. *Only then:* Design 2 feature construction and GBM, reported as a delta
8. Streamlit demo

Realistic effort: ~3–4 focused weekends through step 6.

## Deliverable shape

Repo + README where the **top third is a results table and two plots**, not an architecture diagram:

- **Plot A:** RMSE vs. n_observed, one line per method (4 baselines + Design 1 + Design 2), error bars from the repeated draws
- **Plot B:** improvement over Tomatometer vs. critic dispersion, showing the gain concentrating in the high-dispersion band

Then a small Streamlit app: rate 15 films, get back top critic matches with alignment score and overlap count. The eval makes it credible; the demo makes it memorable and gets clicked.

## Known risks

- **Usable matrix smaller than expected** after filtering → widen n bins, relax the ≥5-in-2023 requirement, report the tradeoff
- **Heuristic and GBM within noise of each other** → a fine result, reported as such; do not manufacture a difference
- **Popularity-weighted truncation is a modeling assumption** → state it, and run one uniform-sampling ablation to show sensitivity
- **Critic dedup is imperfect** → report merge counts and run the eval both with and without dedup
