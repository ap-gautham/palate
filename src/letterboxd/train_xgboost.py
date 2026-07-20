"""Letterboxd Design 2: XGBoost over the shared 37-feature contract (the RT
schema minus the Tomatometer), on the 1-10 member scale.

Self-contained and isolated from Rotten Tomatoes. Trains on random-holdout
member profiles and reports a paired held-out RMSE; the authoritative
seen-history sweep across all designs lives in ``letterboxd.analyze``.

Run from src/:  python -m letterboxd.train_xgboost
Outputs: results/letterboxd/models/letterboxd_xgboost.json (+ meta),
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
    parts = F.partition_members(data)
    rng = np.random.default_rng(SEED + 1)
    print(f"built matrix {data.n_members}x{data.n_movies} ({time.time()-started:.0f}s)")

    tr_x, tr_y, _ = F.generate_rows(data, parts["train"][:args.train_members], rng,
                                    N_GRID, args.profiles_per_n)
    va_x, va_y, _ = F.generate_rows(data, parts["validation"][:args.val_members], rng,
                                    N_GRID, 2)
    te_x, te_y, te_meta = F.generate_paired_rows(data, parts["test"][:args.test_members],
                                                 N_GRID, 8, 3, 50)
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
    print(f"best_iteration {best}  paired test RMSE {test_rmse:.4f}")

    _, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / "letterboxd_xgboost.json"))
    (MODELS / "letterboxd_xgboost_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_xgboost.json", "feature_columns": F.FEATURE_COLS,
        "genre_to_id": genre_to_id, "unknown_genre_id": unknown_genre_id,
        "best_iteration": int(best), "rating_scale": [RATING_MIN, RATING_MAX]}, indent=2))

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_xgb"] = te_pred.astype(np.float32)
    out.to_parquet(RESULTS / "xgboost_test_predictions.parquet", index=False)
    (RESULTS / "xgboost_results.json").write_text(json.dumps({
        "model": "xgboost", "rating_scale": [RATING_MIN, RATING_MAX],
        "feature_columns": F.FEATURE_COLS, "train_rows": int(len(tr_y)),
        "test_rows": int(len(te_y)), "rmse": test_rmse,
        "best_iteration": int(best)}, indent=2))
    per_n = (out.assign(se=lambda d: (d["pred_xgb"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    print("XGBoost test RMSE by seen-count:")
    print(per_n.round(4).to_string())
    print(f"Done in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
