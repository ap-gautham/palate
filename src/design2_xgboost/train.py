"""Design 2: an XGBoost regressor over the engineered episode features.

Generates its own training/validation rows (unpaired random profiles) and its
own paired/nested test rows, so it is fully self-contained; the deterministic
seeds make its training data and test rows match the neural net's. Trains with
early stopping on validation RMSE and evaluates on the shared test episodes.

Run from src/:  python -m design2_xgboost.train

Outputs: results/models/design2_xgboost.json (+ meta),
         results/tables/design2_{results.json,test_predictions.parquet,
         feature_importance.csv}
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODELS, MOVIES_PARQUET, REVIEWS_PARQUET, SEED, TABLES, VALUE_COL
from . import features as F
from .pseudo_users import build_split, partition_pseudo_users, rmse

N_GRID_TRAIN = [3, 5, 10, 20, 50, None]
TRAIN_PROFILES_PER_N = 32
VALIDATION_PROFILES_PER_N = 8
MODEL_FILE = MODELS / "design2_xgboost.json"
MODEL_META_FILE = MODELS / "design2_xgboost_meta.json"


XGB_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.05,
    "max_depth": 6, "min_child_weight": 20, "subsample": 0.8,
    "colsample_bytree": 0.9, "lambda": 1.0, "tree_method": "hist", "seed": SEED,
}


def train_xgb(train_x, train_y, val_x, val_y):
    # Native XGBoost API (no scikit-learn dependency). NaN (a missing
    # Tomatometer) is treated as missing by DMatrix, just as LightGBM handled
    # it. genre_id enters as a plain numeric column -- the neural net embeds it
    # instead; it is a low-importance feature, so the ordinal encoding is cheap.
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=3000,
                        evals=[(dval, "validation")], early_stopping_rounds=100,
                        verbose_eval=False)
    return booster[: booster.best_iteration + 1], booster.best_iteration


def main() -> None:
    started = time.time()
    rng = np.random.default_rng(SEED + 1)
    scored = pd.read_parquet(REVIEWS_PARQUET)
    scored = scored[scored[VALUE_COL].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET)
    genre_of_movie, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)

    print("Building all-time matrix ...")
    split = build_split(scored, movies)
    parts = partition_pseudo_users(split)
    print(f"  train/val/test = {len(parts['train'])}/{len(parts['validation'])}"
          f"/{len(parts['test'])} critics")

    print("Generating training rows ...")
    tr_x, tr_y, _ = F.generate_rows(split, parts["train"], rng, genre_of_movie,
                                    N_GRID_TRAIN, TRAIN_PROFILES_PER_N, unknown_genre_id)
    print(f"  {len(tr_y):,} train rows ({time.time() - started:.0f}s)")
    va_x, va_y, _ = F.generate_rows(split, parts["validation"], rng, genre_of_movie,
                                    N_GRID_TRAIN, VALIDATION_PROFILES_PER_N, unknown_genre_id)
    print(f"  {len(va_y):,} validation rows ({time.time() - started:.0f}s)")
    print("Generating paired test rows ...")
    te_x, te_y, te_meta = F.generate_paired_rows(split, parts["test"],
                                                 genre_of_movie, unknown_genre_id)
    print(f"  {len(te_y):,} test rows ({time.time() - started:.0f}s)")

    print("Training XGBoost ...")
    model, best_iter = train_xgb(tr_x, tr_y, va_x, va_y)
    preds = np.clip(model.predict(xgb.DMatrix(te_x)), 0, 5)
    test_rmse = rmse(preds - te_y)
    print(f"  test RMSE {test_rmse:.4f} (best iteration {best_iter})")

    MODELS.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    MODEL_META_FILE.write_text(json.dumps({
        "model_file": MODEL_FILE.name, "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "test_rmse": float(test_rmse), "best_iteration": int(best_iter),
        "train_rows": int(len(tr_y)), "value_col": VALUE_COL,
    }, indent=2))

    gain = model.get_score(importance_type="gain")
    importance = pd.Series({c: gain.get(c, 0.0) for c in F.FEATURE_COLS})
    importance.sort_values(ascending=False).to_csv(TABLES / "design2_feature_importance.csv")
    out = te_meta.copy()
    out["y"] = te_y
    out["pred_main"] = preds.astype(np.float32)
    out.to_parquet(TABLES / "design2_test_predictions.parquet", index=False)
    (TABLES / "design2_results.json").write_text(json.dumps({
        "main": {"test_rmse": float(test_rmse), "best_iteration": best_iter},
        "model": "xgboost", "train_rows": int(len(tr_y)), "test_rows": int(len(te_y)),
    }, indent=2))

    per_n = (out.assign(se=lambda d: (d["pred_main"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    print("\nXGBoost test RMSE by seen-count:")
    print(per_n.round(4).to_string())
    print(f"\nSaved {MODEL_FILE}. Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
