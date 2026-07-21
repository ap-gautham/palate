"""Letterboxd Design 3: an inductive residual tabular MLP over the shared
37-feature contract (same architecture as the Rotten Tomatoes network), on the
1-10 member scale.

Inductive by construction: it predicts from engineered features of a member's
seen profile, so it scores brand-new users live in the browser (unlike a
transductive embedding model). Trains a small seeded ensemble on the Apple GPU
(MPS) when available. Self-contained and isolated from RT.

Also trains a second, z-score-track ensemble: the target and peer features are
expressed in z-space (each rater standardized by their own scale -- the user
side by THIS episode's own seen-set mean/std, never a member's all-time
stats), and predictions are converted back to the raw scale before scoring.

Run from src/:  python -m letterboxd.train_neural
Outputs: results/letterboxd/models/letterboxd_neural{,_z}.pt (+ meta),
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
    return np.concatenate(out)  # clip happens after any z convert-back


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
            val_pred = model(va_num_t, va_gen_t).cpu().numpy()
        # Unclipped: va_y is raw [1,10] for the raw track but unbounded
        # z-space for the z track, so a single clip boundary can't serve both.
        val_rmse = rmse(val_pred, va_y)
        sched.step(val_rmse)
        if val_rmse < best_val - 1e-5:
            best_val, stale = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_state, best_val, epoch


def train_ensemble(tr_p, va_p, te_num, te_gen, n_genres, device, ensemble_size, seed_offset=0):
    states, val_scores = [], []
    ens = np.zeros(len(te_num), dtype=np.float64)
    for member in range(ensemble_size):
        state, best_val, epochs = train_one(SEED + seed_offset + member, tr_p, va_p, n_genres, device)
        states.append(state)
        val_scores.append(best_val)
        model = TabularResNet(len(NUMERIC_COLS), n_genres, EMB_DIM, WIDTH, DEPTH, DROPOUT).to(device)
        model.load_state_dict(state)
        member_pred = predict(model, te_num, te_gen, device)
        ens += member_pred
        print(f"  member {member+1}/{ensemble_size}: val {best_val:.4f}, {epochs+1} epochs")
    return ens / ensemble_size, states, val_scores


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

    tr = to_arrays(tr_x, tr_y, None)
    va = to_arrays(va_x, va_y, None)
    te = to_arrays(te_x, te_y, te_meta)
    tr_z = to_arrays(tr_z_x, tr_z_y, None)
    va_z = to_arrays(va_z_x, va_z_y, None)
    te_z = to_arrays(te_z_x, te_z_y, te_z_meta)

    tr_num, tr_gen, tr_y, _ = tr
    va_num, va_gen, va_y, _ = va
    te_num, te_gen, te_y, te_meta = te
    tr_z_num, tr_z_gen, tr_z_y, _ = tr_z
    va_z_num, va_z_gen, va_z_y, _ = va_z
    te_z_num, te_z_gen, te_z_y, te_z_meta = te_z
    print(f"rows: train {len(tr_y):,} val {len(va_y):,} test {len(te_y):,} "
          f"({time.time()-started:.0f}s)")

    tr_num, va_num, te_num, mu_impute, mu, sd = preprocess(tr_num, va_num, te_num)
    tr_z_num, va_z_num, te_z_num, mu_impute_z, mu_z, sd_z = preprocess(tr_z_num, va_z_num, te_z_num)

    print("Training raw-track ensemble ...")
    ens, states, val_scores = train_ensemble(
        (tr_num, tr_gen, tr_y), (va_num, va_gen, va_y), te_num, te_gen,
        n_genres, device, args.ensemble, seed_offset=0)
    ens = np.clip(ens, RATING_MIN, RATING_MAX)
    test_rmse = rmse(ens, te_y)
    print(f"Ensemble ({args.ensemble}) raw paired test RMSE {test_rmse:.4f} ({time.time()-started:.0f}s)")

    print("Training z-score-track ensemble ...")
    ens_z, states_z, val_scores_z = train_ensemble(
        (tr_z_num, tr_z_gen, tr_z_y), (va_z_num, va_z_gen, va_z_y), te_z_num, te_z_gen,
        n_genres, device, args.ensemble, seed_offset=100)
    preds_z_raw = np.clip(te_mu + te_sigma * ens_z, RATING_MIN, RATING_MAX)
    test_rmse_z = rmse(preds_z_raw, te_y)
    print(f"Ensemble ({args.ensemble}) z paired test RMSE {test_rmse_z:.4f} "
          f"(raw scale after convert-back, {time.time()-started:.0f}s)")

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

    torch.save({"state_dicts": states_z, "numeric_cols": NUMERIC_COLS,
                "genre_col": GENRE_COL, "log_cols": LOG_COLS, "mu_impute": mu_impute_z,
                "mu": mu_z, "sd": sd_z, "n_genres": n_genres, "emb_dim": EMB_DIM,
                "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT,
                "rating_min": RATING_MIN, "rating_max": RATING_MAX},
               MODELS / "letterboxd_neural_z.pt")
    (MODELS / "letterboxd_neural_z_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_neural_z.pt", "feature_columns": F.FEATURE_COLS,
        "ensemble_size": args.ensemble, "test_rmse": float(test_rmse_z),
        "mean_member_val_rmse": float(np.mean(val_scores_z)),
        "rating_scale": [RATING_MIN, RATING_MAX],
        "architecture": {"embedding_dim": EMB_DIM, "width": WIDTH, "depth": DEPTH,
                         "dropout": DROPOUT},
        "note": "target is (raw - mu_user)/sigma_user; mu_user/sigma_user come "
                "from the user's own seen-set ratings, not a member's all-time "
                "stats. test_rmse above is already converted back to the raw scale.",
    }, indent=2))

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_nn"] = ens.astype(np.float32)
    out["pred_nn_z"] = preds_z_raw.astype(np.float32)
    out.to_parquet(RESULTS / "neural_test_predictions.parquet", index=False)
    (RESULTS / "neural_results.json").write_text(json.dumps({
        "model": "residual_mlp", "rating_scale": [RATING_MIN, RATING_MAX],
        "ensemble_size": args.ensemble, "test_rows": int(len(te_y)),
        "rmse": float(test_rmse), "rmse_z": float(test_rmse_z),
        "mean_member_val_rmse": float(np.mean(val_scores))}, indent=2))
    per_n = (out.assign(se=lambda d: (d["pred_nn"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    per_n_z = (out.assign(se=lambda d: (d["pred_nn_z"] - d["y"]) ** 2)
               .groupby("n")["se"].mean().pipe(np.sqrt))
    print("Neural net test RMSE by seen-count (raw track):")
    print(per_n.round(4).to_string())
    print("Neural net test RMSE by seen-count (z track, converted back):")
    print(per_n_z.round(4).to_string())
    print(f"Done in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
