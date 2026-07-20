"""Design 3: a residual neural-network ensemble over the same episode features
as Design 2.

Self-contained: it generates its own training/validation rows and paired test
rows (the shared seed SEED+1 makes the training data identical to Design 2's,
and the deterministic paired episodes make the test rows byte-identical). It
trains on the Apple GPU (MPS) when available, with AdamW + a ReduceLROnPlateau
schedule and early stopping, and averages three independently seeded networks.

Run from src/:  python -m design3_neural.train

Outputs: results/models/design3_mlp.pt (+ meta),
         results/tables/design3_{results.json,test_predictions.parquet}
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from rotten_tomatoes.config import MODELS, MOVIES_PARQUET, REVIEWS_PARQUET, SEED, TABLES, VALUE_COL
from . import features as F
from .network import TabularResNet
from .pseudo_users import build_split, partition_pseudo_users

MODEL_FILE = MODELS / "design3_mlp.pt"
MODEL_META_FILE = MODELS / "design3_mlp_meta.json"

N_GRID_TRAIN = [3, 5, 10, 20, 50, None]
TRAIN_PROFILES_PER_N = 32
VALIDATION_PROFILES_PER_N = 8

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


def to_arrays(frame: pd.DataFrame, target: np.ndarray, meta: pd.DataFrame):
    numeric = frame[NUMERIC_COLS].to_numpy(dtype=np.float32)
    genre = frame[GENRE_COL].to_numpy(dtype=np.int64)
    return numeric, genre, target.astype(np.float32), meta


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


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
    return np.clip(np.concatenate(out), 0.0, 5.0)


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

    best_val, best_state, stale = np.inf, None, 0
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
            val_rmse = rmse(np.clip(model(va_num_t, va_gen_t).cpu().numpy(), 0, 5), va_y)
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
    started = time.time()
    np.random.seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    rng = np.random.default_rng(SEED + 1)
    scored = pd.read_parquet(REVIEWS_PARQUET)
    scored = scored[scored[VALUE_COL].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET)
    genre_of_movie, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    n_genres = len(genre_to_id)

    split = build_split(scored, movies)
    parts = partition_pseudo_users(split)
    print("Generating features (own copy; identical to Design 2 by seed) ...")
    tr = to_arrays(*F.generate_rows(split, parts["train"], rng, genre_of_movie,
                                    N_GRID_TRAIN, TRAIN_PROFILES_PER_N, unknown_genre_id))
    va = to_arrays(*F.generate_rows(split, parts["validation"], rng, genre_of_movie,
                                    N_GRID_TRAIN, VALIDATION_PROFILES_PER_N, unknown_genre_id))
    te = to_arrays(*F.generate_paired_rows(split, parts["test"], genre_of_movie, unknown_genre_id))
    tr_num, tr_gen, tr_y, _ = tr
    va_num, va_gen, va_y, _ = va
    te_num, te_gen, te_y, te_meta = te
    print(f"  rows: train {len(tr_y):,}  val {len(va_y):,}  test {len(te_y):,} "
          f"({time.time() - started:.0f}s)")
    print(f"  features: {len(NUMERIC_COLS)} numeric + genre embedding")

    tr_num, va_num, te_num, mu_impute, mu, sd = preprocess(tr_num, va_num, te_num)
    tr_p = (tr_num, tr_gen, tr_y)
    va_p = (va_num, va_gen, va_y)

    states, val_scores = [], []
    ens = np.zeros(len(te_y), dtype=np.float64)
    for member in range(ENSEMBLE_SIZE):
        state, best_val, epochs = train_one(SEED + member, tr_p, va_p, n_genres, device)
        states.append(state)
        val_scores.append(best_val)
        model = TabularResNet(len(NUMERIC_COLS), n_genres, EMB_DIM, WIDTH, DEPTH, DROPOUT).to(device)
        model.load_state_dict(state)
        member_pred = predict(model, te_num, te_gen, device)
        ens += member_pred
        print(f"  member {member + 1}/{ENSEMBLE_SIZE}: val {best_val:.4f}, "
              f"test {rmse(member_pred, te_y):.4f}, {epochs + 1} epochs "
              f"({time.time() - started:.0f}s)")
    ens /= ENSEMBLE_SIZE
    test_rmse = rmse(ens, te_y)
    print(f"\nEnsemble ({ENSEMBLE_SIZE}) test RMSE {test_rmse:.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": states, "numeric_cols": NUMERIC_COLS,
                "genre_col": GENRE_COL, "log_cols": LOG_COLS, "mu_impute": mu_impute,
                "mu": mu, "sd": sd, "n_genres": n_genres, "emb_dim": EMB_DIM,
                "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT}, MODEL_FILE)
    MODEL_META_FILE.write_text(json.dumps({
        "model_file": MODEL_FILE.name, "feature_columns": F.FEATURE_COLS,
        "ensemble_size": ENSEMBLE_SIZE, "test_rmse": float(test_rmse),
        "mean_member_val_rmse": float(np.mean(val_scores)),
        "architecture": {"embedding_dim": EMB_DIM, "width": WIDTH, "depth": DEPTH,
                         "dropout": DROPOUT}}, indent=2))

    out = te_meta.copy()
    out["y"] = te_y
    out["pred_nn"] = ens.astype(np.float32)
    out.to_parquet(TABLES / "design3_test_predictions.parquet", index=False)
    (TABLES / "design3_results.json").write_text(json.dumps({
        "test_rmse": float(test_rmse), "ensemble_size": ENSEMBLE_SIZE,
        "test_rows": int(len(te_y))}, indent=2))
    per_n = (out.assign(se=lambda d: (d["pred_nn"] - d["y"]) ** 2)
             .groupby("n")["se"].mean().pipe(np.sqrt))
    print("\nNeural net test RMSE by seen-count:")
    print(per_n.round(4).to_string())
    print(f"\nSaved {MODEL_FILE}. Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
