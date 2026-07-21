"""Letterboxd Design 2: XGBoost over the shared 37-feature contract (the RT
schema minus the Tomatometer), on the 1-10 member scale.

Also trains a second, z-score-track model: the target and peer features are
expressed in z-space (each rater standardized by their own scale), and
predictions are converted back to the raw scale before scoring, so the two
tracks' RMSE is directly comparable (see features.py's ``episode_feature_row_z``
for the exact standardization rule -- the user's own seen-set mean/std, never
a member's all-time stats).

Self-contained and isolated from Rotten Tomatoes. Trains on random-holdout
member profiles and reports a paired held-out RMSE; the authoritative
seen-history sweep across all designs lives in ``letterboxd.analyze``.

Run from src/:  python -m letterboxd.train_xgboost
Outputs: results/letterboxd/models/letterboxd_xgboost{,_z}.json (+ meta),
         results/letterboxd/xgboost_results.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (MODELS, MOVIES_PARQUET, RATING_MAX, RATING_MIN, RATINGS_PARQUET,
                     RESULTS, SEED)
from . import features as F

N_GRID = [3, 5, 10, 20, 50, None]
XGB_PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.05,
              "max_depth": 6, "min_child_weight": 20, "subsample": 0.8,
              "colsample_bytree": 0.9, "tree_method": "hist", "seed": SEED}


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-members", type=int, default=1500)
    parser.add_argument("--val-members", type=int, default=300)
    parser.add_argument("--test-members", type=int, default=200)
    parser.add_argument("--profiles-per-n", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")

    started = time.time()
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    data = F.build_data(ratings, movies)
    data_z = F.build_data(ratings, movies, value="z")
    parts = F.partition_members(data)
    rng = np.random.default_rng(SEED + 1)
    print(f"built matrix {data.n_members}x{data.n_movies} ({time.time()-started:.0f}s)")

    ((tr_x, tr_y, _), (tr_z_x, tr_z_y, _), _, _) = F.generate_rows(
        data, parts["train"][:args.train_members], rng, N_GRID, args.profiles_per_n, data_z=data_z)
    ((va_x, va_y, _), (va_z_x, va_z_y, _), _, _) = F.generate_rows(
        data, parts["validation"][:args.val_members], rng, N_GRID, 2, data_z=data_z)
    ((te_x, te_y, te_meta), (te_z_x, te_z_y, te_z_meta), te_mu, te_sigma) = F.generate_paired_rows(
        data, parts["test"][:args.test_members], N_GRID, 8, 3, 50, data_z=data_z)
    print(f"rows: train {len(tr_y):,} val {len(va_y):,} test {len(te_y):,} "
          f"({time.time()-started:.0f}s)")

    dtrain = xgb.DMatrix(tr_x[F.FEATURE_COLS], label=tr_y)
    dval = xgb.DMatrix(va_x[F.FEATURE_COLS], label=va_y)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=args.rounds,
                        evals=[(dval, "val")], early_stopping_rounds=40, verbose_eval=False)
    best = booster.best_iteration
    te_pred = np.clip(booster.predict(xgb.DMatrix(te_x[F.FEATURE_COLS]),
                                      iteration_range=(0, best + 1)), RATING_MIN, RATING_MAX)
    test_rmse = rmse(te_pred, te_y)
    print(f"raw track: best_iteration {best}  paired test RMSE {test_rmse:.4f}")

    dtrain_z = xgb.DMatrix(tr_z_x[F.FEATURE_COLS], label=tr_z_y)
    dval_z = xgb.DMatrix(va_z_x[F.FEATURE_COLS], label=va_z_y)
    booster_z = xgb.train(XGB_PARAMS, dtrain_z, num_boost_round=args.rounds,
                          evals=[(dval_z, "val")], early_stopping_rounds=40, verbose_eval=False)
    best_z = booster_z.best_iteration
    te_pred_z_raw = np.clip(
        te_mu + te_sigma * booster_z.predict(xgb.DMatrix(te_z_x[F.FEATURE_COLS]),
                                             iteration_range=(0, best_z + 1)),
        RATING_MIN, RATING_MAX)
    test_rmse_z = rmse(te_pred_z_raw, te_y)  # te_y: same episodes' raw ground truth
    print(f"z track:   best_iteration {best_z}  paired test RMSE {test_rmse_z:.4f} "
          f"(raw scale after convert-back)")

    _, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / "letterboxd_xgboost.json"))
    (MODELS / "letterboxd_xgboost_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_xgboost.json", "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "best_iteration": int(best), "rating_scale": [RATING_MIN, RATING_MAX]}, indent=2))

    booster_z.save_model(str(MODELS / "letterboxd_xgboost_z.json"))
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


if __name__ == "__main__":
    main()
