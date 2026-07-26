"""Letterboxd Design 2: XGBoost over the shared feature contract (the RT
schema minus the Tomatometer), on the 1-10 member scale.

Also trains a second, z-score-track model: the target and peer features are
expressed in z-space (each rater standardized by their own scale), and
predictions are converted back to the raw scale before scoring, so the two
tracks' RMSE is directly comparable (see features.py's ``episode_feature_row_z``
for the exact standardization rule -- the user's own seen-set mean/std, never
a member's all-time stats).

Self-contained and isolated from Rotten Tomatoes. Loads the cached row pool
built by ``build_rows`` (`make lb-rows`), so this trainer and the neural net
fit on byte-identical rows; the authoritative seen-history sweep across all
designs lives in ``letterboxd.analyze``.

Run from src/:  python -m letterboxd.train_xgboost
Outputs: results/letterboxd/models/letterboxd_xgboost{,_z}.json (+ meta),
         results/letterboxd/xgboost_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import MODELS, MOVIES_PARQUET, RATING_MAX, RATING_MIN, RESULTS, SEED
from . import features as F
from . import pseudo_users as PU
from .build_rows import load_rows
from .pseudo_users import rmse
from .plots import plot_rmse_by_n, rmse_by_n, score_other_design

# scikit-learn-style hyperparameters (XGBRegressor names: eta -> learning_rate,
# seed -> random_state, num_boost_round -> n_estimators).
XGB_PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse",
              "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 20,
              "subsample": 0.8, "colsample_bytree": 0.9, "tree_method": "hist",
              "random_state": SEED}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--plot-file", type=str, default=None,
                        help="write a scratch RMSE-by-n plot here after training "
                             "(default: results/letterboxd/figures/temp_design2.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the scratch plot entirely (used by `make`)")
    args = parser.parse_args()

    started = time.time()
    rows = load_rows()
    (tr_x, tr_y, _), (tr_z_x, tr_z_y, _) = rows["train"], rows["train_z"]
    (va_x, va_y, _), (va_z_x, va_z_y, _) = rows["val"], rows["val_z"]
    (te_x, te_y, te_meta), (te_z_x, te_z_y, te_z_meta) = rows["test"], rows["test_z"]
    te_mu, te_sigma = rows["te_mu"], rows["te_sigma"]
    print(f"loaded cached rows: train {len(tr_y):,} val {len(va_y):,} "
          f"test {len(te_y):,} ({time.time()-started:.0f}s)")
    movies = pd.read_parquet(MOVIES_PARQUET)

    # XGBoost through its scikit-learn estimator API: fit on the feature frame
    # directly, and predict() stops at best_iteration once early stopping fired.
    model = XGBRegressor(n_estimators=args.rounds, early_stopping_rounds=40, **XGB_PARAMS)
    model.fit(tr_x[F.FEATURE_COLS], tr_y, eval_set=[(va_x[F.FEATURE_COLS], va_y)], verbose=False)
    best = model.best_iteration
    te_pred = np.clip(model.predict(te_x[F.FEATURE_COLS]), RATING_MIN, RATING_MAX)
    test_rmse = rmse(te_pred - te_y)
    print(f"raw track: best_iteration {best}  paired test RMSE {test_rmse:.4f}")

    model_z = XGBRegressor(n_estimators=args.rounds, early_stopping_rounds=40, **XGB_PARAMS)
    model_z.fit(tr_z_x[F.FEATURE_COLS], tr_z_y,
                eval_set=[(va_z_x[F.FEATURE_COLS], va_z_y)], verbose=False)
    best_z = model_z.best_iteration
    te_pred_z_raw = np.clip(
        te_mu + te_sigma * model_z.predict(te_z_x[F.FEATURE_COLS]), RATING_MIN, RATING_MAX)
    test_rmse_z = rmse(te_pred_z_raw - te_y)  # te_y: same episodes' raw ground truth
    print(f"z track:   best_iteration {best_z}  paired test RMSE {test_rmse_z:.4f} "
          f"(raw scale after convert-back)")

    _, genre_to_id, unknown_genre_id = PU.make_genre_maps(movies)
    MODELS.mkdir(parents=True, exist_ok=True)
    # Save the booster truncated to the early-stopping optimum -- the same
    # trees predict() uses here, so the JS tree-walker (which walks every tree
    # in the dump) agrees with Python instead of carrying the extra rounds.
    model.get_booster()[: best + 1].save_model(str(MODELS / "letterboxd_xgboost.json"))
    (MODELS / "letterboxd_xgboost_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_xgboost.json", "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "best_iteration": int(best), "rating_scale": [RATING_MIN, RATING_MAX]}, indent=2))

    model_z.get_booster()[: best_z + 1].save_model(str(MODELS / "letterboxd_xgboost_z.json"))
    (MODELS / "letterboxd_xgboost_z_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_xgboost_z.json", "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "best_iteration": int(best_z), "rating_scale": [RATING_MIN, RATING_MAX],
        "note": "target is (raw - mu_user)/sigma_user; mu_user/sigma_user come "
                "from the user's own seen-set ratings, not a member's all-time "
                "stats. rmse below is already converted back to the raw scale.",
    }, indent=2))

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_xgb"] = te_pred.astype(np.float32)
    out["pred_xgb_z"] = te_pred_z_raw.astype(np.float32)
    out.to_parquet(RESULTS / "xgboost_test_predictions.parquet", index=False)
    (RESULTS / "xgboost_results.json").write_text(json.dumps({
        "model": "xgboost", "rating_scale": [RATING_MIN, RATING_MAX],
        "feature_columns": F.FEATURE_COLS, "train_rows": int(len(tr_y)),
        "test_rows": int(len(te_y)), "rmse": test_rmse, "rmse_z": test_rmse_z,
        "best_iteration": int(best), "best_iteration_z": int(best_z)}, indent=2))
    per_n = (out.assign(se=lambda d: (d["pred_xgb"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    per_n_z = (out.assign(se=lambda d: (d["pred_xgb_z"] - d["y"]) ** 2)
               .groupby("n")["se"].mean().pipe(np.sqrt))
    print("XGBoost test RMSE by seen-count (raw track):")
    print(per_n.round(4).to_string())
    print("XGBoost test RMSE by seen-count (z track, converted back):")
    print(per_n_z.round(4).to_string())
    print(f"Done in {time.time()-started:.0f}s")

    if not args.no_plot:
        n_col = te_meta["n"].to_numpy()
        curves = {
            "design2": rmse_by_n(te_pred, te_y, n_col),
            "design2_z": rmse_by_n(te_pred_z_raw, te_y, n_col),
            "zero": rmse_by_n(np.full(len(te_y), rows["global_mean"]), te_y, n_col),
            "movie_mean": rmse_by_n(rows["movie_mean"][te_meta["tcol"].to_numpy()], te_y, n_col),
        }
        nn_path = MODELS / "letterboxd_neural.pt"
        if nn_path.exists():
            try:
                # scored in a fresh process -- see plots.score_other_design
                other_pred = score_other_design("nn", nn_path, te_x, F.FEATURE_COLS,
                                                RATING_MIN, RATING_MAX)
                if other_pred is not None:
                    curves["design3"] = rmse_by_n(other_pred, te_y, n_col)
                else:
                    print("  (scratch plot: skipping design3 curve -- feature contract changed)")
            except Exception as e:
                print(f"  (scratch plot: skipping design3 curve -- {e})")
        plot_path = Path(args.plot_file) if args.plot_file else RESULTS / "figures" / "temp_design2.png"
        plot_rmse_by_n(curves, plot_path,
                       "Letterboxd Design 2: scratch RMSE by seen-count (this run)",
                       "RMSE on paired test episodes (1-10)")
        print(f"Scratch plot written to {plot_path} "
              f"(canonical figures are only produced by `make lb-analyze`)")


if __name__ == "__main__":
    main()
