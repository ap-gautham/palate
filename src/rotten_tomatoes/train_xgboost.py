"""Design 2: an XGBoost regressor over the engineered episode features.

Loads the cached row pool built by ``build_rows`` (`make rt-rows`), so this
trainer and the neural net fit on byte-identical rows and score the same
paired test episodes. Trains with early stopping on validation RMSE.

Also trains a second, z-score-track model: the target and the peer features
are expressed in z-space (each rater standardized by their own scale), and
predictions are converted back to the raw scale before scoring, so the two
tracks' RMSE is directly comparable (see features.py's
``episode_feature_row_z`` for the exact standardization rule -- the user's own
seen-set mean/std, never a critic's all-time stats).

Run from src/:  python -m rotten_tomatoes.train_xgboost

Outputs: results/models/design2_xgboost{,_z}.json (+ meta),
         results/tables/design2_{results.json,test_predictions.parquet,
         feature_importance.csv}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from rotten_tomatoes.config import MODELS, MOVIES_PARQUET, SEED, TABLES, VALUE_COL
from rotten_tomatoes import features as F
from rotten_tomatoes.build_rows import load_rows
from rotten_tomatoes.pseudo_users import rmse
from rotten_tomatoes.plots import plot_rmse_by_n, rmse_by_n, score_other_design

MODEL_FILE = MODELS / "design2_xgboost.json"
MODEL_META_FILE = MODELS / "design2_xgboost_meta.json"
MODEL_FILE_Z = MODELS / "design2_xgboost_z.json"
MODEL_META_FILE_Z = MODELS / "design2_xgboost_z_meta.json"


# scikit-learn-style hyperparameters (XGBRegressor names: eta -> learning_rate,
# lambda -> reg_lambda, seed -> random_state, num_boost_round -> n_estimators).
XGB_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse", "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 20, "subsample": 0.8,
    "colsample_bytree": 0.9, "reg_lambda": 1.0, "tree_method": "hist",
    "random_state": SEED,
}


def train_xgb(train_x, train_y, val_x, val_y):
    # XGBoost through its scikit-learn estimator API: a plain fit/predict
    # estimator that takes the feature frame directly. NaN (a missing
    # Tomatometer, or any of the affinity blocks' NaN-able columns) is
    # handled natively via XGBoost's missing-value split direction.
    model = XGBRegressor(n_estimators=3000, early_stopping_rounds=100,
                         importance_type="gain", **XGB_PARAMS)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    # predict() already stops at model.best_iteration once early stopping fired.
    return model, model.best_iteration


def best_booster(model):
    """The underlying Booster truncated to the early-stopping optimum -- what
    gets saved, so the JS tree-walker scores exactly like predict() does here."""
    return model.get_booster()[: model.best_iteration + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-file", type=str, default=None,
                        help="write a scratch RMSE-by-n plot here after training "
                             "(default: results/rotten_tomatoes/figures/temp_design2.png)")
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
          f"test {len(te_y):,} ({time.time() - started:.0f}s)")
    movies = pd.read_parquet(MOVIES_PARQUET)

    print("Training XGBoost (raw track) ...")
    model, best_iter = train_xgb(tr_x, tr_y, va_x, va_y)
    preds = np.clip(model.predict(te_x), 0, 5)
    test_rmse = rmse(preds - te_y)
    print(f"  test RMSE {test_rmse:.4f} (best iteration {best_iter})")

    print("Training XGBoost (z-score track) ...")
    model_z, best_iter_z = train_xgb(tr_z_x, tr_z_y, va_z_x, va_z_y)
    preds_z_raw = np.clip(te_mu + te_sigma * model_z.predict(te_z_x), 0, 5)
    # te_z_y is the z-space target; te_y (raw frame) is the same episodes'
    # ground truth on the raw scale, which is what a converted-back prediction
    # must be compared against.
    test_rmse_z = rmse(preds_z_raw - te_y)
    print(f"  test RMSE {test_rmse_z:.4f} (best iteration {best_iter_z}, raw scale after convert-back)")

    _, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    MODELS.mkdir(parents=True, exist_ok=True)
    best = best_booster(model)
    best.save_model(str(MODEL_FILE))
    MODEL_META_FILE.write_text(json.dumps({
        "model_file": MODEL_FILE.name, "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "test_rmse": float(test_rmse), "best_iteration": int(best_iter),
        "train_rows": int(len(tr_y)), "value_col": VALUE_COL,
    }, indent=2))

    best_booster(model_z).save_model(str(MODEL_FILE_Z))
    MODEL_META_FILE_Z.write_text(json.dumps({
        "model_file": MODEL_FILE_Z.name, "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "test_rmse": float(test_rmse_z), "best_iteration": int(best_iter_z),
        "train_rows": int(len(tr_z_y)), "value_col": "z",
        "note": "target is (raw - mu_user)/sigma_user; mu_user/sigma_user come "
                "from the user's own seen-set ratings, not a critic's all-time "
                "stats. test_rmse above is already converted back to the raw scale.",
    }, indent=2))

    gain = best.get_score(importance_type="gain")
    importance = pd.Series({c: gain.get(c, 0.0) for c in F.FEATURE_COLS})
    importance.sort_values(ascending=False).to_csv(TABLES / "design2_feature_importance.csv")

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_main"] = preds.astype(np.float32)
    out["pred_main_z"] = preds_z_raw.astype(np.float32)
    out.to_parquet(TABLES / "design2_test_predictions.parquet", index=False)
    (TABLES / "design2_results.json").write_text(json.dumps({
        "main": {"test_rmse": float(test_rmse), "best_iteration": best_iter},
        "main_z": {"test_rmse": float(test_rmse_z), "best_iteration": best_iter_z},
        "model": "xgboost", "train_rows": int(len(tr_y)), "test_rows": int(len(te_y)),
    }, indent=2))

    per_n = (out.assign(se=lambda d: (d["pred_main"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    per_n_z = (out.assign(se=lambda d: (d["pred_main_z"] - d["y"]) ** 2)
               .groupby("n")["se"].mean().pipe(np.sqrt))
    print("\nXGBoost test RMSE by seen-count (raw track):")
    print(per_n.round(4).to_string())
    print("\nXGBoost test RMSE by seen-count (z track, converted back):")
    print(per_n_z.round(4).to_string())
    print(f"\nSaved {MODEL_FILE} and {MODEL_FILE_Z}. Done in {time.time() - started:.0f}s")

    if not args.no_plot:
        n_col = te_meta["n"].to_numpy()
        curves = {
            "design2": rmse_by_n(preds, te_y, n_col),
            "design2_z": rmse_by_n(preds_z_raw, te_y, n_col),
            "zero": rmse_by_n(np.full(len(te_y), rows["global_mean"]), te_y, n_col),
            "movie_mean": rmse_by_n(rows["movie_mean"][te_meta["tcol"].to_numpy()], te_y, n_col),
        }
        nn_path = MODELS / "design3_mlp.pt"
        if nn_path.exists():
            try:
                # scored in a fresh process -- see plots.score_other_design
                other_pred = score_other_design("nn", nn_path, te_x, F.FEATURE_COLS, 0.0, 5.0)
                if other_pred is not None:
                    curves["design3"] = rmse_by_n(other_pred, te_y, n_col)
                else:
                    print("  (scratch plot: skipping design3 curve -- feature contract changed)")
            except Exception as e:
                print(f"  (scratch plot: skipping design3 curve -- {e})")
        plot_path = Path(args.plot_file) if args.plot_file else TABLES.parent / "figures" / "temp_design2.png"
        plot_rmse_by_n(curves, plot_path,
                       "Rotten Tomatoes Design 2: scratch RMSE by seen-count (this run)",
                       "RMSE on paired test episodes (0-5)")
        print(f"Scratch plot written to {plot_path} "
              f"(canonical figures are only produced by `make rt-analyze`)")


if __name__ == "__main__":
    main()
