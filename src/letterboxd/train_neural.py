"""Letterboxd Design 3: an inductive residual tabular MLP over the shared
37-feature contract (same architecture as the Rotten Tomatoes network), on the
1-10 member scale.

Inductive by construction: it predicts from engineered features of a member's
seen profile, so it scores brand-new users live in the browser and Streamlit
(unlike a transductive embedding model). Trains a small seeded ensemble on the
Apple GPU (MPS) when available. Self-contained and isolated from RT.

Run from src/:  python -m letterboxd.train_neural
Outputs: results/letterboxd/models/letterboxd_neural.pt (+ meta),
         results/letterboxd/neural_results.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import (MODELS, MOVIES_PARQUET, RATING_MAX, RATING_MIN, RATINGS_PARQUET,
                     RESULTS, SEED)
from . import features as F
from .network import TabularResNet

N_GRID = [3, 5, 10, 20, 50, None]
NUMERIC_COLS = [c for c in F.FEATURE_COLS if c != "genre_id"]
GENRE_COL = "genre_id"
LOG_COLS = ["n_observed", "mean_overlap", "max_overlap", "n_reviewers"] + \
           [f"d{i}_cnt" for i in range(10)]
LOG_IDX = np.array([NUMERIC_COLS.index(c) for c in LOG_COLS])

EMB_DIM, WIDTH, DEPTH, DROPOUT = 24, 512, 6, 0.1
ENSEMBLE_SIZE, MAX_EPOCHS, PATIENCE = 3, 300, 20
BATCH, LR, WEIGHT_DECAY = 8192, 2e-3, 1e-5


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def to_arrays(frame, target, meta):
    return (frame[NUMERIC_COLS].to_numpy(np.float32),
            frame[GENRE_COL].to_numpy(np.int64), target.astype(np.float32), meta)


def preprocess(tr, va, te):
    for arr in (tr, va, te):
        arr[:, LOG_IDX] = np.log1p(np.clip(arr[:, LOG_IDX], 0, None))
    mu_impute = np.nanmean(tr, axis=0)
    for arr in (tr, va, te):
        mask = np.isnan(arr)
        arr[mask] = np.take(mu_impute, np.where(mask)[1])
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (tr - mu) / sd, (va - mu) / sd, (te - mu) / sd, mu_impute, mu, sd


@torch.no_grad()
def predict(model, numeric, genre, device):
    model.eval()
    out = []
    for start in range(0, len(numeric), 16384):
        nb = torch.from_numpy(numeric[start:start + 16384]).to(device)
        gb = torch.from_numpy(genre[start:start + 16384]).to(device)
        out.append(model(nb, gb).cpu().numpy())
    return np.clip(np.concatenate(out), RATING_MIN, RATING_MAX)


def train_one(seed, tr, va, n_genres, device):
    tr_num, tr_gen, tr_y = tr
    va_num, va_gen, va_y = va
    torch.manual_seed(seed)
    model = TabularResNet(tr_num.shape[1], n_genres, EMB_DIM, WIDTH, DEPTH, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    tr_num_t = torch.from_numpy(tr_num).to(device)
    tr_gen_t = torch.from_numpy(tr_gen).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    va_num_t = torch.from_numpy(va_num).to(device)
    va_gen_t = torch.from_numpy(va_gen).to(device)
    n = len(tr_y)
    best_val, best_state, stale, epoch = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(tr_num_t[idx], tr_gen_t[idx]), tr_y_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_rmse = rmse(np.clip(model(va_num_t, va_gen_t).cpu().numpy(),
                                    RATING_MIN, RATING_MAX), va_y)
        sched.step(val_rmse)
        if val_rmse < best_val - 1e-5:
            best_val, stale = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_state, best_val, epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-members", type=int, default=1500)
    parser.add_argument("--val-members", type=int, default=300)
    parser.add_argument("--test-members", type=int, default=200)
    parser.add_argument("--profiles-per-n", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=ENSEMBLE_SIZE)
    args = parser.parse_args()
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")

    started = time.time()
    device = pick_device()
    print(f"device: {device}")
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    _, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    n_genres = len(genre_to_id)
    data = F.build_data(ratings, movies)
    parts = F.partition_members(data)
    rng = np.random.default_rng(SEED + 1)
    print(f"built matrix {data.n_members}x{data.n_movies} ({time.time()-started:.0f}s)")

    tr = to_arrays(*F.generate_rows(data, parts["train"][:args.train_members], rng,
                                    N_GRID, args.profiles_per_n))
    va = to_arrays(*F.generate_rows(data, parts["validation"][:args.val_members], rng,
                                    N_GRID, 2))
    te = to_arrays(*F.generate_paired_rows(data, parts["test"][:args.test_members],
                                           N_GRID, 8, 3, 50))
    tr_num, tr_gen, tr_y, _ = tr
    va_num, va_gen, va_y, _ = va
    te_num, te_gen, te_y, te_meta = te
    print(f"rows: train {len(tr_y):,} val {len(va_y):,} test {len(te_y):,} "
          f"({time.time()-started:.0f}s)")

    tr_num, va_num, te_num, mu_impute, mu, sd = preprocess(tr_num, va_num, te_num)
    tr_p, va_p = (tr_num, tr_gen, tr_y), (va_num, va_gen, va_y)

    states, val_scores = [], []
    ens = np.zeros(len(te_y), dtype=np.float64)
    for member in range(args.ensemble):
        state, best_val, epochs = train_one(SEED + member, tr_p, va_p, n_genres, device)
        states.append(state)
        val_scores.append(best_val)
        model = TabularResNet(len(NUMERIC_COLS), n_genres, EMB_DIM, WIDTH, DEPTH, DROPOUT).to(device)
        model.load_state_dict(state)
        member_pred = predict(model, te_num, te_gen, device)
        ens += member_pred
        print(f"  member {member+1}/{args.ensemble}: val {best_val:.4f}, "
              f"test {rmse(member_pred, te_y):.4f}, {epochs+1} epochs ({time.time()-started:.0f}s)")
    ens /= args.ensemble
    test_rmse = rmse(ens, te_y)
    print(f"\nEnsemble ({args.ensemble}) paired test RMSE {test_rmse:.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": states, "numeric_cols": NUMERIC_COLS,
                "genre_col": GENRE_COL, "log_cols": LOG_COLS, "mu_impute": mu_impute,
                "mu": mu, "sd": sd, "n_genres": n_genres, "emb_dim": EMB_DIM,
                "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT,
                "rating_min": RATING_MIN, "rating_max": RATING_MAX},
               MODELS / "letterboxd_neural.pt")
    (MODELS / "letterboxd_neural_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_neural.pt", "feature_columns": F.FEATURE_COLS,
        "ensemble_size": args.ensemble, "test_rmse": float(test_rmse),
        "mean_member_val_rmse": float(np.mean(val_scores)),
        "rating_scale": [RATING_MIN, RATING_MAX],
        "architecture": {"embedding_dim": EMB_DIM, "width": WIDTH, "depth": DEPTH,
                         "dropout": DROPOUT}}, indent=2))

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_nn"] = ens.astype(np.float32)
    out.to_parquet(RESULTS / "neural_test_predictions.parquet", index=False)
    (RESULTS / "neural_results.json").write_text(json.dumps({
        "model": "residual_mlp", "rating_scale": [RATING_MIN, RATING_MAX],
        "ensemble_size": args.ensemble, "test_rows": int(len(te_y)),
        "rmse": float(test_rmse),
        "mean_member_val_rmse": float(np.mean(val_scores))}, indent=2))
    per_n = (out.assign(se=lambda d: (d["pred_nn"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    print("Neural net test RMSE by seen-count:")
    print(per_n.round(4).to_string())
    print(f"Done in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
