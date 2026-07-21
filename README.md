# Palate — Critic-Matched Movie Prediction

Live at **[ap-gautham.github.io/palate](https://ap-gautham.github.io/palate/)**.

A portfolio project that predicts the rating *you* would give a movie from the
scores other people gave it — and tests, honestly, where that personalization
actually beats a flat consensus average and where it just pretends to. Full
methodology, formulas, leakage controls, and results live in the typeset
report: **[report/report.pdf](report/report.pdf)**. This README is a map of
the repo and how to reproduce it.

## Two symmetric projects

The repository holds two intentionally isolated, identically-structured movie
rating predictors, so the comparison between them is fair by construction, not
by convention. Every directory is mirrored — `src/{rotten_tomatoes,
letterboxd}/`, `data/{rotten_tomatoes,letterboxd}/`,
`results/{rotten_tomatoes,letterboxd}/`, `web/public/data/{rotten_tomatoes,
letterboxd}/` — and neither project reads the other's raw data, processed
files, or trained models.

| | Rotten Tomatoes | Letterboxd |
|---|---|---|
| rater population | ~4,400 professional critics (pseudo-users) | 7,420 real members |
| target scale | parsed to 0–5 | native 1–10 |
| eligibility floor | ≥10 scored films | ≥5 rated films |
| Design 1 | analytic critic-match formula | analytic member-match formula (same formula) |
| Design 2 | XGBoost (incl. Tomatometer) | XGBoost (same ~110-feature contract, no Tomatometer) |
| Design 3 | trained residual-MLP ensemble | trained residual-MLP ensemble (same architecture) |
| interactive catalog | 1,000 popular films, live in browser | 1,000 popular films, live in browser |

There are no real per-user histories in the Rotten Tomatoes source data, so
each eligible critic is treated as a pseudo-user: sampled ratings are "movies
they've seen," a different rated movie is the unseen target. Letterboxd has
genuine per-member histories instead. Both designs 2/3 share the identical
~110-feature similarity-decile-plus-movie-facet contract (minus Rotten
Tomatoes' Tomatometer feature), so their Design 2/3 pairs are a fair
architectural comparison, not just a shared Design 1 formula. See
[`src/letterboxd/README.md`](src/letterboxd/README.md) for the
Letterboxd-specific contract and commands.

**Movie facets.** Both catalogs are joined (by normalized title + year) to the
`gsimonx37/letterboxd` Kaggle metadata dump — genre, theme, studio, director,
actor, decade, language, country (posters excluded, unneeded) — giving every
trained model a per-user affinity term for each facet ("this user over-rates
A24 films," not just "this user and critic X agree"), built leave-target-out
from seen films only. Design 1 also gets a new **top-|sim| variant**: instead
of the full peer neighbourhood, it restricts to the 10 peers with the largest
`|similarity|` (aligned *or* anti-aligned), computed identically for both
datasets. It underperforms the full formula on both — a negative result,
reported as found rather than tuned away.

## Headline results

Full-history RMSE on the shared paired/nested seen-history protocol (see the
report for the full per-`n` tables and figures):

| design | Rotten Tomatoes (0–5) | Letterboxd (1–10) |
|---|---:|---:|
| Design 1 analytic (full neighbourhood) | 0.796 | 1.507 |
| Design 1 analytic (top-\|sim\| variant) | 0.808 | 1.526 |
| Design 2 XGBoost | 0.770 | 1.455 |
| Design 3 neural net | 0.776 | 1.468 |

**Honest cross-dataset finding.** The two scales aren't directly comparable,
so normalizing by rating range (`RMSE / (max − min)`) puts them on the same
footing — Rotten Tomatoes (~0.154–0.162) and Letterboxd (~0.162–0.170) come out
**comparable**, with Rotten Tomatoes marginally *better* despite its far
sparser critic pseudo-user profiles. Letterboxd's real value is dense,
genuine per-member histories, not a lower headline RMSE — "more real data
performs much better" is not supported by this matched-protocol comparison,
even with the richer movie-facet features added to both.

**Isolating scale (the z-score experiment).** Both projects also train a
parallel z-score track: each rater is standardized to their own scale (the
model predicts pure *deviation*, not level), and predictions are converted
back to the raw scale before scoring — the direct test of whether the RMSE
gain above is level-calibration or genuine taste-matching. The result is
**not uniform**: on Rotten Tomatoes the z-track trails raw at every design
(full formula +0.130, top-\|sim\| +0.009, design2 +0.008, design3 +0.005
RMSE); on Letterboxd the trained models come out *flat-to-slightly-better* in
z-space (design2 +0.0004, design3 −0.004), the top-\|sim\| variant is nearly
indifferent (+0.001), while the full-neighbourhood formula is worse (+0.134)
— reported as found. See
`results/{rotten_tomatoes,letterboxd}/raw_vs_z.csv` / `plotC_raw_vs_z.png` and
the report for the full breakdown.

## Website

Built with Vite + React + TypeScript (`web/`), hosted as a static GitHub Pages
site from `docs/` — no server. The app defaults to the interactive tab, with a
dataset switcher between Rotten Tomatoes and Letterboxd. **All four model
variants (analytic full, analytic top-\|sim\|, XGBoost, neural net), each with
raw and z-score tracks, for both datasets, run client-side in the browser**:
each analytic formula is plain arithmetic, each XGBoost model is a
from-scratch JSON tree-walker (`web/src/lib/xgboost.ts`, shared by both
datasets via a parameterized clamp range), and each neural net is a
hand-written forward pass (`web/src/lib/neuralnet.ts`, also shared) over the
ensemble's raw weights — including the movie-facet affinity computation
(`web/src/lib/features.ts`), ported from Python's `_facet_tail`. Rotten
Tomatoes' export lives under `src/rotten_tomatoes/web_export/export.py`;
Letterboxd's mirrors it at `src/letterboxd/web_export.py`. Both TypeScript
ports are checked against their Python inference paths in
`web/scripts/validate*.ts` (the analytic formulas match to float precision;
the trained models agree within a documented tie-break tolerance — see the
comment in each project's `features.ts` for why).

Each catalog holds its **1,000 most-rated films**; a sort control orders the
search lists alphabetically (default), by year, or by rating count, a
**genre filter** narrows the list to a single gsimonx37 genre, and the
dropdown scrolls through every match rather than truncating. Three sections:
**films you've seen** (star widget for Rotten Tomatoes, a 1–10 stepper for
Letterboxd), **films to predict** (scoring optional), and **predictions** —
every method's raw and z-score prediction side by side with the consensus
mean, plus (rating any predict film) the mean squared error of each variant
against your own score, with the closest marked.

**"Movies like this one" suggestions.** Each row in "films to predict" has an
expandable dropdown — green if you've already rated one of that film's 20
nearest content neighbours, red otherwise — listing 20 similar films
(K-means over genre/decade/theme/studio/director/actor/runtime/rating/year,
`movie_features.top_similar`) with a tip to rate them to improve that
prediction. The neighbour-list length (`k_neighbors`) was itself tuned: a
paired-episode sweep (`rotten_tomatoes.design1_analytic.similar_k_sweep`,
`letterboxd.similar_k_sweep`) bucketed 6,000 held-out predictions by how many
of the target's k nearest neighbours were already in the rater's seen set,
for `k_neighbors` ∈ {10, 20, 30}. On Rotten Tomatoes RMSE trends down (noisily,
non-monotonically) as the similar-seen count rises, and the episode-weighted
RMSE across the ≥3-similar-seen buckets is lowest at `k=20` (1.121, n=43, vs.
`k=30`'s 1.525/n=99 and `k=10`'s barely-sampled 3.966/n=7) — modest evidence,
but pointing to `k=20`. Letterboxd's own sweep was noisier still and did not
cleanly discriminate among the three values at this sample size — reported
honestly rather than forced to agree — so `k_neighbors=20` was set for both
projects on the strength of Rotten Tomatoes' (relatively) cleaner result. See
`results/{rotten_tomatoes,letterboxd}/similar_k_sweep.csv` and the report's
"Similar-film suggestions" section for the full breakdown.

To rebuild the site after retraining a model:

```bash
cd src
python -m rotten_tomatoes.web_export.export
python -m letterboxd.web_export
cd ../web && npm install && npm run build   # writes into docs/
```

## Reproduce

Both projects' movie-facet join reads `gsimonx37/letterboxd` (Kaggle) —
`movies.csv`, `genres.csv`, `themes.csv`, `studios.csv`, `actors.csv`,
`crew.csv`, `countries.csv`, `languages.csv` (skip the posters file, it's
unused and huge) — placed once in `data/letterboxd/raw/gsimonx37/` and read
independently by both projects. The join result is cached to
`data/{project}/processed/movie_facets.pkl` after the first run (a ~1–2
minute join over the full catalog); delete the cache to force a rebuild.

Rotten Tomatoes (place the Kaggle CSVs in `data/rotten_tomatoes/raw/`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd src
for m in rotten_tomatoes.preprocessing.build_dataset rotten_tomatoes.preprocessing.audit \
         rotten_tomatoes.design1_analytic.run rotten_tomatoes.design2_xgboost.train \
         rotten_tomatoes.design3_neural.train rotten_tomatoes.design1_analytic.attribution \
         rotten_tomatoes.app_catalog.export rotten_tomatoes.comparison.analysis; do
  ../.venv/bin/python -m "$m"
done
cd ..
.venv/bin/python -m unittest discover -s tests -v
```

Letterboxd (place the Kaggle CSVs in `data/letterboxd/raw/` — see
[`src/letterboxd/README.md`](src/letterboxd/README.md) for the full contract):

```bash
cd src
for m in letterboxd.preprocess letterboxd.train_xgboost letterboxd.train_neural letterboxd.analyze; do
  ../.venv/bin/python -m "$m"
done
```

All random seeds are fixed. The pseudo-user substrate (`pseudo_users.py`) and
the feature contract (`features.py`) each live once, at the top of
`src/rotten_tomatoes/`, and are imported by all three Rotten Tomatoes designs
(`design1_analytic/`, `design2_xgboost/`, `design3_neural/`) — so the
cross-design comparison is exact by construction. `src/letterboxd/` mirrors
this layout with its own `features.py`. Full methodology, every formula term
explained, leakage controls, and the complete results (including the z-score
experiment for both projects) are in **[report/report.pdf](report/report.pdf)**.
