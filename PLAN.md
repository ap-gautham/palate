# Plan: Symmetric two-project repo (Rotten Tomatoes + Letterboxd) + website/app

> **Status (living document).**
> - ✅ **Phase 0 — symmetric restructure**: complete and verified (RT moved to
>   `src/rotten_tomatoes/`, data/results/web dirs mirrored, imports/paths fixed;
>   6 unit tests pass, web build + RT JS↔Python parity unchanged).
> - ✅ **Phase 1 — Letterboxd feature contract + all 3 designs + analysis**:
>   complete. `features.py` (37-feature contract), `network.py`,
>   `train_xgboost.py`, `train_neural.py`, `analyze.py`. Full-history RMSE
>   (1–10 scale): Design 1 = 1.507, Design 2 XGBoost = 1.501, Design 3 neural =
>   1.515; baselines global-mean 2.108, consensus 1.639. Honest cross-dataset
>   finding: normalized RMSE ≈ 0.167 (LB) vs 0.158 (RT) — comparable, RT slightly
>   better normalized, so **not** "much better" on Letterboxd. Artifacts in
>   `results/letterboxd/` (nsweep_summary.csv, model_summary.csv,
>   cross_dataset_comparison.csv, figures/plotA_rmse_vs_n.png,
>   figures/score_distributions.png).
> - ✅ **Phase 2 — Letterboxd interactive browser demo**: complete.
>   `letterboxd/web_export.py` exports the top-1000 submatrix + both models to
>   `web/public/data/letterboxd/`; all three designs ported to TypeScript under
>   `web/src/lib/letterboxd/`, reusing the now-parameterized `xgboost.ts`/
>   `neuralnet.ts` (clamp range [1,10]). `LetterboxdApp.tsx` replaces the old
>   static panel — live in the project switcher, verified end-to-end with
>   headless Chrome (ratings → live 3-design predictions, MSE-vs-you, closest
>   members). JS↔Python parity confirmed (analytic ~1e-7, XGBoost/NN within
>   ~0.01–0.05 on the 1–10 scale). RT test suite and RT parity unaffected.
> - ✅ **Phase 3 — Streamlit parity**: complete. `run_letterboxd_app` now uses
>   the shared `letterboxd.features` module (37-feature contract) for both
>   XGBoost and the trained neural net (via `letterboxd.analyze.load_nn`/
>   `nn_predict`), reusing the exact analytic formula. Added the "Which method
>   predicts you best?" MSE-vs-you block and a "Your closest members" expander,
>   mirroring RT. All "untrained"/"Not trained" language removed. Verified
>   end-to-end with headless Chrome: rating films across both Rotten Tomatoes
>   and Letterboxd tabs produces live 3-design predictions, correct MSE-vs-you
>   winner, and a populated closest-members table.
> - ✅ **Phase 4 — verify already-done asks**: confirmed in code (app defaults
>   to the app tab unless `#about`; autocomplete returns the full filtered list,
>   no 10-item cap) and via screenshots taken throughout Phase 2/3 testing.
> - ✅ **Phase 5 — docs**: complete. README rewritten (symmetric layout, correct
>   LB numbers, honest normalized-RMSE comparison table, updated commands);
>   DOCUMENTATION.md parallel-project table + path fixes; `src/letterboxd/README.md`
>   fully rewritten (feature contract, inductive Design 3, results, honest
>   comparison); `report/report.tex` gained a new "Second dataset: Letterboxd
>   community ratings" section (table + figure + honest finding), recompiled to
>   a 7-page PDF with no warnings, copied to both `docs/assets/` and
>   `web/public/assets/`; `About.tsx` gained a matching "Second dataset" section
>   with LB cards, plot, and the honest cross-dataset callout — verified via
>   screenshot. Repo-wide sweep confirms no remaining "untrained" language or
>   stale 1.695/1.628 numbers.
> - ⏳ **Phase 6**: not started (final build/verify/ship).

## Context

The repo hosts a Rotten Tomatoes (RT) critic-matched rating predictor with a
Vite/React site (`web/` → `docs/`, live on GitHub Pages) and a Streamlit app. A
prior run added an isolated Letterboxd (LB) project (`src/letterboxd/`,
`data/letterboxd/`, `results/letterboxd/`): 7,420 members / 11.08M ratings on a
1–10 scale, Design 1 analytic (RMSE 1.664), a 5-feature Design 2 XGBoost, a
functional interactive LB Streamlit tab, and a static website panel.

The user wants the two projects to be **perfectly symmetric siblings in every
directory** (not just LB isolated while RT sits at `src/` top level), LB
completed to **full parity** with RT (same feature contract, all three designs
trained and live, same analysis), and everything framed **honestly**.

**Chosen folder names** (flag if you want different): `rotten_tomatoes` and
`letterboxd` (snake_case), used identically under `src/`, `data/`, `results/`,
`web/public/data/`, `docs/data/`.

**Two decisions already made with the user:**
- LB Design 3 neural = **inductive feature-MLP** (mirrors RT's Design 3; runs
  live in the browser + Streamlit for any new user; a proper curve in the
  n-sweep). NOT the transductive embedding-CF.
- Full parallel analysis + full interactive browser demo + honest framing.

**Consequence baked into this plan:** to make the inductive MLP meaningful and
truly parallel, LB adopts RT's **similarity-decile feature contract** (minus
Tomatometer), and **Design 2 XGBoost is retrained on that same contract** so
Designs 2 & 3 are compared on identical features exactly like RT. This changes
LB's XGBoost number (recompute; do not reuse 1.628).

**Reality correction to propagate honestly:** the dataset is 7,420 *users* with
11M *ratings*, not 11M users. Compare via **normalized RMSE = RMSE/(max−min)**
and state plainly whether LB is better or merely comparable — never claim
superiority the matched-protocol numbers don't support. LB's genuine advantage
is dense real-user histories vs sparse critic pseudo-users.

---

## Phase 0 — Symmetric restructure (do FIRST; verify before continuing)

Highest-risk phase (import paths). Execute exactly, then pass the 0.8 checkpoint
before any feature work.

**0.1 Move RT source into a package** (`git mv` if tracked else `mv`):
`src/config.py` and the dirs `preprocessing/ design1_analytic/ design2_xgboost/
design3_neural/ comparison/ app_catalog/ web_export/` → under
`src/rotten_tomatoes/`. Add `src/rotten_tomatoes/__init__.py` (mirror
`src/letterboxd/__init__.py`). `src/letterboxd/` stays.

**0.2 Fix Python imports** (run everything from `src/`):
- 14 files do `from config import …` → `from rotten_tomatoes.config import …`
  (`design{1,2,3}_*/…`, `preprocessing/{audit,build_dataset,score_distributions}`,
  `app_catalog/export`, `comparison/analysis`).
- Absolute cross-package imports (`from design1_analytic import`, …
  `design2_xgboost`, `design3_neural`, `app_catalog`, `comparison`,
  `preprocessing`) → prefix `rotten_tomatoes.` in: `app/streamlit_app.py`,
  `tests/test_harness.py`, `tests/test_design2_features.py`,
  `src/rotten_tomatoes/web_export/validate_against_js.py`. Intra-package
  relative imports are unchanged.

**0.3 `src/rotten_tomatoes/config.py` paths** (`ROOT =
Path(__file__).resolve().parents[2]`): `DATA_RAW/DATA_PROCESSED` →
`data/rotten_tomatoes/{raw,processed}`; `RESULTS = ROOT/"results"/
"rotten_tomatoes"`; FIGURES/TABLES/MODELS under it.

**0.4 Move data + results on disk**: `data/raw`→`data/rotten_tomatoes/raw`,
`data/processed`→`data/rotten_tomatoes/processed`;
`results/{tables,models,figures}`→`results/rotten_tomatoes/{…}`,
`results/*.log`→`results/rotten_tomatoes/`. LB dirs already correct.

**0.5 Web data dirs symmetric**: move RT `web/public/data/{*.bin,*.json}` →
`web/public/data/rotten_tomatoes/`; set `web/src/lib/data.ts`
`BASE = import.meta.env.BASE_URL + "data/rotten_tomatoes/"`; point RT
`web_export/export.py` output there. (LB export writes `…/data/letterboxd/`.)

**0.6 `.gitignore`** → `data/rotten_tomatoes/{raw,processed}`,
`data/letterboxd/{raw,processed}`, `results/rotten_tomatoes/models`,
`results/letterboxd/models`, `web/node_modules/`, scratch. Small
`results/**/{tables,figures,*.json,*.csv}` stay committed.

**0.7 Update RT commands in docs** to package form, from `src/`, e.g.
`python -m rotten_tomatoes.preprocessing.build_dataset`,
`…rotten_tomatoes.design1_analytic.run`, etc., in `README.md` + `DOCUMENTATION.md`.

**0.8 CHECKPOINT (must pass before Phase 1):**
`cd src && ../.venv/bin/python -c "import rotten_tomatoes.config"` + each
design's `predict` import; `python -m unittest discover -s tests -v` (6) passes;
`cd web && npm run build` succeeds with `docs/data/rotten_tomatoes/*`; RT app
screenshot still interactive. Fix any failure before continuing.

---

## Phase 1 — Letterboxd: shared feature contract, train all 3 designs, full analysis

**1.1 New `src/letterboxd/features.py`** mirroring RT
`src/rotten_tomatoes/design2_xgboost/features.py`, adapted to members and 1–10,
**without Tomatometer**: 10 deciles × {mean,cnt,std} of `sim×deviation` (30) +
tail `n_observed, mean_overlap, max_overlap, n_reviewers, dispersion, genre_id,
user_mean` (7) = 37 features; genre from `movie_data.csv` first genre (build a
genre→id map like RT). Reuse the CSR sparse member-similarity already in
`src/letterboxd/train_analytic.py` for the decile inputs; provide episode-row
generation (train/val) and paired-episode generation (test) like RT.

**1.2 Retrain Design 2 XGBoost** — update `src/letterboxd/train_xgboost.py` to
use the 37-feature contract (drop the old 5-feature set); clip [1,10]; save
`results/letterboxd/models/letterboxd_xgboost.json` + meta (feature_columns,
genre_to_id). Recompute RMSE.

**1.3 Rewrite Design 3 as an inductive MLP** — replace the embedding-CF in
`src/letterboxd/train_neural.py` with an RT-style network (mirror
`src/rotten_tomatoes/design3_neural/network.py`: input BatchNorm + genre
embedding + residual/plain blocks + linear head), over the same 37-feature
contract (36 numeric + genre embedding), clip [1,10]. **Train it** (small
ensemble, e.g. 1–3 members; MPS if available; may take ~10–30 min — run in
background and report). Keep a `--train` guard but the reproduce commands now
pass `--train`. Save `results/letterboxd/models/letterboxd_neural.pt` + meta
(mu/sd/impute/log_cols/layer offsets) shaped for a browser export like RT.
Remove all "untrained"/"not trained" language project-wide.

**1.4 New `src/letterboxd/analyze.py`** (mirror RT
`src/rotten_tomatoes/comparison/analysis.py` style): paired/nested n-sweep
(`N_GRID=[3,5,10,20,50,None]`, 8 targets, 3 draws, SEED 42) over ~300–500 test
members with >50 ratings; baselines B1 global mean, B3 movie/consensus mean
(leave-one-out), B4 top-k neighbour mean (no B2 Tomatometer); Designs 1/2/3 all
as curves. Outputs `results/letterboxd/nsweep_summary.csv`, `model_summary.csv`,
and `results/letterboxd/figures/{plotA_rmse_vs_n.png,score_distributions.png}`
styled identically to RT. Also `results/letterboxd/cross_dataset_comparison.csv`
(RT vs LB full-history RMSE, range, normalized RMSE; read RT's from
`results/rotten_tomatoes/tables/model_summary.csv`; do not hardcode).

Keep everything fast/vectorised over the sparse matrix; never densify.

---

## Phase 2 — Letterboxd interactive browser demo (all 3 designs live)

Mirror RT's full browser stack (`src/rotten_tomatoes/web_export/export.py` +
`web/src/lib/*`), now INCLUDING the neural net.

**2.1 New `src/letterboxd/web_export.py`** → `web/public/data/letterboxd/`:
`movies.json` (top-1000: id/title/year/nScores/genreId), `members.json`,
`ratings_member_idx.bin`/`ratings_movie_idx.bin` (Uint16) +
`ratings_score.bin` (Float32) for the top-1000 submatrix (~23 MB),
`xgb_model.json` (compact tree dump, 37 cols), `nn_meta.json` +
`nn_weights_member*.bin` (mirror RT's neural export), `meta.json`
(ratingMin 1, ratingMax 10, kShrink 8, genreToId, counts). Reuse RT export
helpers (`export_xgboost`, the NN layer-flatten order, `allow_nan=False`,
slug-title fallback).

**2.2 TS port under `web/src/lib/letterboxd/`** (parallel modules; leave RT
files intact): `types.ts`, `data.ts` (lazy-load), `matches.ts` (member
similarity), `features.ts` (deciles + tail, no Tomatometer — copy RT
`features.ts`), `design1.ts` (clip [1,10]), `neuralnet.ts` (reuse RT
`web/src/lib/neuralnet.ts` forward pass; clip [1,10]), `predict.ts` (Designs
1+2+3). Reuse generic `web/src/lib/xgboost.ts` but **parameterize its clamp
range** (optional lo,hi; default 0,5) so LB passes [1,10]; RT unchanged. Extend
`web/scripts/validate.ts` + a Python validator with an LB case (JS vs Python
within ~0.02).

**2.3 Frontend** — replace static `LetterboxdTab` in `web/src/App.tsx` with
`web/src/pages/LetterboxdApp.tsx` mirroring `PredictApp.tsx` (seen/predict/
predictions + MSE-vs-you + closest-members expander), reusing `FilmAutocomplete`
/`FilmTable`; add `web/src/components/RatingInput.tsx` (1–10 stepper) for LB (RT
keeps `StarRating`). Columns: Analytic, XGBoost, Neural net, Consensus (mean),
Your score (no Tomatometer). Add
`web/src/lib/letterboxd/useLetterboxdData.ts` (mirror `useAppData.ts`) with the
loading spinner for the ~23 MB + NN stream.

---

## Phase 3 — Streamlit parity

`app/streamlit_app.py` LB path (`run_letterboxd_app`): run **all three designs
live** (load the new LB XGBoost + neural, compute the 37-feature row like RT’s
LB `features.py`); add the **"Which method predicts you best?"** MSE block and a
**"Your closest members"** expander (reuse RT helpers). Keep the accurate
"7,420 members · 286,069 films · 11.08M ratings" caption; no "11M users",
no "untrained".

---

## Phase 4 — Verify already-done asks (1) & (2)

Confirm in the build/screenshot pass: app is the default tab, About + `Palate.`
brand clickable (`web/src/App.tsx`); autocomplete lists the full filtered
catalog and scrolls (`FilmAutocomplete.tsx`).

---

## Phase 5 — Docs: honest, parallel, corrected (numbers from Phase 1 outputs)

- `web/src/pages/About.tsx` + LB page copy: use the freshly computed LB
  analytic/XGBoost/neural RMSEs; add an honest cross-dataset note (comparable
  normalized; LB edge = dense real-user histories; cross-scale RMSE not directly
  comparable); optional compact table from `cross_dataset_comparison.csv`.
- `README.md`: correct all LB numbers; evidence-led framing (no "much better"/
  "untrained"); n-sweep table + `python -m letterboxd.{analyze,train_xgboost,
  train_neural --train}`; "7,420 members, 11.08M ratings"; new symmetric layout
  + package commands.
- `DOCUMENTATION.md`: expand LB subsection (paired/nested n-sweep, baselines,
  37-feature contract, leakage, honest comparison); update RT/LB table + paths.
- `src/letterboxd/README.md`: add features/analyze/train steps; keep honest
  "not directly comparable" caveat.
- `report/report.tex` (0 LB mentions): new section "Second dataset: Letterboxd
  community ratings" — dataset, 1–10 scale, symmetric pipeline, 37-feature
  contract, the n-sweep table + LB figures, all three designs trained, honest
  normalized comparison. Recompile (`pdflatex` ×2) → refresh
  `docs/assets/palate-report.pdf`.

---

## Phase 6 — Build, verify, ship

1. From `src/` (env `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`):
   `python -m letterboxd.train_xgboost`, `python -m letterboxd.train_neural
   --train`, `python -m letterboxd.analyze`, `python -m letterboxd.web_export`.
2. Parity: `cd web && npx tsx scripts/validate.ts` (RT + LB) + Python validators.
3. `cd web && npm run build`; confirm `docs/data/{rotten_tomatoes,letterboxd}/`;
   browser data ≈ 41 MB (RT) + ~48 MB (LB incl NN) — still within limits.
4. Screenshot RT + LB interactive (rate → live 3-design predictions), About,
   switcher (headless Chrome) — no console errors.
5. `.venv/bin/python -m unittest discover -s tests -v` (regression guard).
6. `git add -A`; review `git status` (no `web/node_modules/`, no `data/**/raw`
   or `processed`, no scratch; `docs/`, `web/` sources, small
   `results/**/{tables,figures,*.json,*.csv}` committed). Commit (Co-Authored-By
   trailer) and push; Pages redeploys from `docs/`.

## Guardrails
- Phase 0 gate: don't start Phase 1 until 0.8 is green.
- Projects fully isolated: neither reads the other's inputs or writes its dirs.
- LB neural is now trained (inductive) — no "untrained" text anywhere.
- Never "11 million users" (7,420 members / 11.08M ratings). Only real,
  freshly-computed numbers; no invented RMSEs.
- `data/**/{raw,processed}`, `web/node_modules/` stay gitignored.

## Key files
- Phase 0: all of `src/*` → `src/rotten_tomatoes/`, `config.py` paths,
  `web/src/lib/data.ts`, `.gitignore`, `app/streamlit_app.py`, `tests/*`,
  `README.md`, `DOCUMENTATION.md`.
- New: `src/letterboxd/{features,analyze,web_export}.py`,
  `web/src/lib/letterboxd/*`, `web/src/pages/LetterboxdApp.tsx`,
  `web/src/components/RatingInput.tsx`, `results/letterboxd/figures/*`.
- Edit: `src/letterboxd/{train_xgboost,train_neural}.py` (37-feature contract;
  train neural), `web/src/App.tsx`, `web/src/lib/xgboost.ts` (clamp range),
  `web/src/pages/About.tsx`, `web/scripts/validate.ts`, `app/streamlit_app.py`,
  `report/report.tex`, `docs/assets/palate-report.pdf`.
- Reuse: RT `design3_neural/network.py` + `web/src/lib/neuralnet.ts` (LB neural),
  RT `design2_xgboost/features.py` (LB features), RT `comparison/analysis.py`
  (style), RT `web_export/export.py` (export pattern),
  `src/letterboxd/train_analytic.py` (sparse similarity).
