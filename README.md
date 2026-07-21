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
| Design 2 | XGBoost (incl. Tomatometer) | XGBoost (same 37-feature contract, no Tomatometer) |
| Design 3 | trained residual-MLP ensemble | trained residual-MLP ensemble (same architecture) |
| interactive catalog | 1,000 popular films, live in browser | 1,000 popular films, live in browser |

There are no real per-user histories in the Rotten Tomatoes source data, so
each eligible critic is treated as a pseudo-user: sampled ratings are "movies
they've seen," a different rated movie is the unseen target. Letterboxd has
genuine per-member histories instead. Both designs 2/3 share the identical
37-feature similarity-decile contract (minus Rotten Tomatoes' Tomatometer
feature), so their Design 2/3 pairs are a fair architectural comparison, not
just a shared Design 1 formula. See [`src/letterboxd/README.md`](src/letterboxd/README.md)
for the Letterboxd-specific contract and commands.

## Headline results

Full-history RMSE on the shared paired/nested seen-history protocol (see the
report for the full per-`n` tables and figures):

| design | Rotten Tomatoes (0–5) | Letterboxd (1–10) |
|---|---:|---:|
| Design 1 analytic | 0.796 | 1.507 |
| Design 2 XGBoost | 0.775 | 1.502 |
| Design 3 neural net | 0.773 | 1.518 |

**Honest cross-dataset finding.** The two scales aren't directly comparable,
so normalizing by rating range (`RMSE / (max − min)`) puts them on the same
footing — Rotten Tomatoes (~0.155–0.169) and Letterboxd (~0.167–0.180) come out
**comparable**, with Rotten Tomatoes marginally *better* despite its far
sparser critic pseudo-user profiles. Letterboxd's real value is dense,
genuine per-member histories, not a lower headline RMSE — "more real data
performs much better" is not supported by this matched-protocol comparison.

**Isolating scale (the z-score experiment).** Both projects also train a
parallel z-score track: each rater is standardized to their own scale (the
model predicts pure *deviation*, not level), and predictions are converted
back to the raw scale before scoring — the direct test of whether the RMSE
gain above is level-calibration or genuine taste-matching. The result is
**not uniform**: on Rotten Tomatoes the z-track trails raw at every design
(design1 +0.130, design2 +0.008, design3 +0.015 RMSE); on Letterboxd the
trained models (design2/3) come out *slightly better* in z-space (−0.010,
−0.021) while the analytic formula is worse (+0.134) — reported as found. See
`results/{rotten_tomatoes,letterboxd}/raw_vs_z.csv` / `plotC_raw_vs_z.png` and
the report for the full breakdown.

## Website

Built with Vite + React + TypeScript (`web/`), hosted as a static GitHub Pages
site from `docs/` — no server. The app defaults to the interactive tab, with a
dataset switcher between Rotten Tomatoes and Letterboxd. **All three models,
raw and z-score variants, for both datasets, run client-side in the
browser**: each analytic formula is plain arithmetic, each XGBoost model is a
from-scratch JSON tree-walker (`web/src/lib/xgboost.ts`, shared by both
datasets via a parameterized clamp range), and each neural net is a
hand-written forward pass (`web/src/lib/neuralnet.ts`, also shared) over the
ensemble's raw weights. Rotten Tomatoes' export lives under
`src/rotten_tomatoes/web_export/export.py`; Letterboxd's mirrors it at
`src/letterboxd/web_export.py`. Both TypeScript ports are checked against
their Python inference paths in `web/scripts/validate*.ts` (the analytic
formula matches to float precision; the trained models agree to within
~0.01–0.05 on their respective scales — see the comment in each project's
`features.ts` for why).

Each catalog holds its **1,000 most-rated films**; a sort control orders the
search lists alphabetically (default), by year, or by rating count, and the
dropdown scrolls through every match rather than truncating. Three sections:
**films you've seen** (star widget for Rotten Tomatoes, a 1–10 stepper for
Letterboxd), **films to predict** (scoring optional), and **predictions** —
every method's raw and z-score prediction side by side with the consensus
mean, plus (rating any predict film) the mean squared error of each variant
against your own score, with the closest marked.

To rebuild the site after retraining a model:

```bash
cd src
python -m rotten_tomatoes.web_export.export
python -m letterboxd.web_export
cd ../web && npm install && npm run build   # writes into docs/
```

## Reproduce

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
