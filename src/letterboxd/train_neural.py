"""Letterboxd Design 3: an inductive residual tabular MLP over the shared
feature contract (same architecture as the Rotten Tomatoes network), on the
1-10 member scale. No genre embedding: the per-genre affinity block already
gives the model per-genre information directly (see network.py).

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
from pathlib import Path

import numpy as np
# NOTE: never import xgboost in this process. torch and xgboost each bundle
# their own OpenMP runtime, and a process that loads both segfaults the moment
# either does real parallel work; the scratch plot scores the XGBoost sibling
# in a clean subprocess instead (see plots.score_other_design).
import torch
from torch import nn

from .config import MODELS, RATING_MAX, RATING_MIN, RESULTS, SEED
from . import features as F
from .build_rows import load_rows
from .pseudo_users import rmse
from .network import TabularResNet
from .plots import plot_rmse_by_n, rmse_by_n, score_other_design

NUMERIC_COLS = F.FEATURE_COLS
LOG_COLS = ["n_observed", "mean_overlap", "max_overlap", "n_reviewers"] + \
           [f"d{i}_cnt" for i in range(10)]
LOG_IDX = np.array([NUMERIC_COLS.index(c) for c in LOG_COLS])

# Fixed per-column NaN sentinel for the network (XGBoost splits on NaN
# natively via missing-value direction; the network needs a value it can
# learn as "this affinity has no evidence"). Rating-scale averages get -1
# (impossible on the 1-10 scale); z-scores get 0 (their own neutral value,
# and the "no evidence" case coincides with the column's own zero point).
# Every other numeric column never contains NaN, so its entry is unused.
_Z_SCORE_COLS = (F.GENRE_Z_COLS + F.ACTOR_BYRATING_Z_COLS
                 + F.ACTOR_BYCOUNT_Z_COLS + ["user_director_z"])
_RATING_AVG_COLS = ["user_theme_avg"] + F.CAST_OVERLAP_RATING_COLS
_IMPUTE_OVERRIDES = {**{c: 0.0 for c in _Z_SCORE_COLS},
                     **{c: -1.0 for c in _RATING_AVG_COLS}}
IMPUTE_VALUE = np.array([_IMPUTE_OVERRIDES.get(c, 0.0) for c in NUMERIC_COLS],
                        dtype=np.float32)

WIDTH, DEPTH, DROPOUT = 512, 6, 0.1
ENSEMBLE_SIZE, MAX_EPOCHS, PATIENCE = 3, 300, 20
BATCH, LR, WEIGHT_DECAY = 8192, 2e-3, 1e-5


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def to_arrays(frame, target, meta):
    return frame[NUMERIC_COLS].to_numpy(np.float32), target.astype(np.float32), meta


def preprocess(tr, va, te):
    """Impute with the fixed ``IMPUTE_VALUE`` sentinel (not the column mean --
    see its definition) so a missing affinity always lands at the same
    out-of-range or neutral point regardless of what this particular split's
    training rows happened to average."""
    for arr in (tr, va, te):
        arr[:, LOG_IDX] = np.log1p(np.clip(arr[:, LOG_IDX], 0, None))
    for arr in (tr, va, te):
        mask = np.isnan(arr)
        arr[mask] = np.take(IMPUTE_VALUE, np.where(mask)[1])
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (tr - mu) / sd, (va - mu) / sd, (te - mu) / sd, mu, sd


@torch.no_grad()
def predict(model, numeric, device):
    model.eval()
    out = []
    for start in range(0, len(numeric), 16384):
        nb = torch.from_numpy(numeric[start:start + 16384]).to(device)
        out.append(model(nb).cpu().numpy())
    return np.concatenate(out)  # clip happens after any z convert-back


def train_one(seed, tr, va, device):
    tr_num, tr_y = tr
    va_num, va_y = va
    torch.manual_seed(seed)
    model = TabularResNet(tr_num.shape[1], WIDTH, DEPTH, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    tr_num_t = torch.from_numpy(tr_num).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    va_num_t = torch.from_numpy(va_num).to(device)
    n = len(tr_y)
    best_val, best_state, stale, epoch = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(tr_num_t[idx]), tr_y_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(va_num_t).cpu().numpy()
        # Unclipped: va_y is raw [1,10] for the raw track but unbounded
        # z-space for the z track, so a single clip boundary can't serve both.
        val_rmse = rmse(val_pred - va_y)
        sched.step(val_rmse)
        if val_rmse < best_val - 1e-5:
            best_val, stale = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_state, best_val, epoch


def train_ensemble(tr_p, va_p, te_num, device, ensemble_size, seed_offset=0):
    states, val_scores = [], []
    ens = np.zeros(len(te_num), dtype=np.float64)
    for member in range(ensemble_size):
        state, best_val, epochs = train_one(SEED + seed_offset + member, tr_p, va_p, device)
        states.append(state)
        val_scores.append(best_val)
        model = TabularResNet(len(NUMERIC_COLS), WIDTH, DEPTH, DROPOUT).to(device)
        model.load_state_dict(state)
        member_pred = predict(model, te_num, device)
        ens += member_pred
        print(f"  member {member+1}/{ensemble_size}: val {best_val:.4f}, {epochs+1} epochs")
    return ens / ensemble_size, states, val_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--plot-file", type=str, default=None,
                        help="write a scratch RMSE-by-n plot here after training "
                             "(default: results/letterboxd/figures/temp_design3.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the scratch plot entirely (used by `make`)")
    args = parser.parse_args()

    started = time.time()
    device = pick_device()
    print(f"device: {device}")

    rows = load_rows()
    (tr_x, tr_y, _), (tr_z_x, tr_z_y, _) = rows["train"], rows["train_z"]
    (va_x, va_y, _), (va_z_x, va_z_y, _) = rows["val"], rows["val_z"]
    (te_x, te_y, te_meta), (te_z_x, te_z_y, te_z_meta) = rows["test"], rows["test_z"]
    te_mu, te_sigma = rows["te_mu"], rows["te_sigma"]
    print(f"loaded cached rows ({time.time()-started:.0f}s)")

    tr = to_arrays(tr_x, tr_y, None)
    va = to_arrays(va_x, va_y, None)
    te = to_arrays(te_x, te_y, te_meta)
    tr_z = to_arrays(tr_z_x, tr_z_y, None)
    va_z = to_arrays(va_z_x, va_z_y, None)
    te_z = to_arrays(te_z_x, te_z_y, te_z_meta)

    tr_num, tr_y, _ = tr
    va_num, va_y, _ = va
    te_num, te_y, te_meta = te
    tr_z_num, tr_z_y, _ = tr_z
    va_z_num, va_z_y, _ = va_z
    te_z_num, te_z_y, te_z_meta = te_z
    print(f"rows: train {len(tr_y):,} val {len(va_y):,} test {len(te_y):,} "
          f"({time.time()-started:.0f}s total)")

    tr_num, va_num, te_num, mu, sd = preprocess(tr_num, va_num, te_num)
    tr_z_num, va_z_num, te_z_num, mu_z, sd_z = preprocess(tr_z_num, va_z_num, te_z_num)

    print("Training raw-track ensemble ...")
    ens, states, val_scores = train_ensemble(
        (tr_num, tr_y), (va_num, va_y), te_num, device, args.ensemble, seed_offset=0)
    ens = np.clip(ens, RATING_MIN, RATING_MAX)
    test_rmse = rmse(ens - te_y)
    print(f"Ensemble ({args.ensemble}) raw paired test RMSE {test_rmse:.4f} ({time.time()-started:.0f}s)")

    print("Training z-score-track ensemble ...")
    ens_z, states_z, val_scores_z = train_ensemble(
        (tr_z_num, tr_z_y), (va_z_num, va_z_y), te_z_num, device, args.ensemble, seed_offset=100)
    preds_z_raw = np.clip(te_mu + te_sigma * ens_z, RATING_MIN, RATING_MAX)
    test_rmse_z = rmse(preds_z_raw - te_y)
    print(f"Ensemble ({args.ensemble}) z paired test RMSE {test_rmse_z:.4f} "
          f"(raw scale after convert-back, {time.time()-started:.0f}s)")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": states, "numeric_cols": NUMERIC_COLS,
                "log_cols": LOG_COLS, "mu_impute": IMPUTE_VALUE,
                "mu": mu, "sd": sd,
                "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT,
                "rating_min": RATING_MIN, "rating_max": RATING_MAX},
               MODELS / "letterboxd_neural.pt")
    (MODELS / "letterboxd_neural_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_neural.pt", "feature_columns": F.FEATURE_COLS,
        "ensemble_size": args.ensemble, "test_rmse": float(test_rmse),
        "mean_member_val_rmse": float(np.mean(val_scores)),
        "rating_scale": [RATING_MIN, RATING_MAX],
        "architecture": {"width": WIDTH, "depth": DEPTH, "dropout": DROPOUT}}, indent=2))

    torch.save({"state_dicts": states_z, "numeric_cols": NUMERIC_COLS,
                "log_cols": LOG_COLS, "mu_impute": IMPUTE_VALUE,
                "mu": mu_z, "sd": sd_z,
                "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT,
                "rating_min": RATING_MIN, "rating_max": RATING_MAX},
               MODELS / "letterboxd_neural_z.pt")
    (MODELS / "letterboxd_neural_z_meta.json").write_text(json.dumps({
        "model_file": "letterboxd_neural_z.pt", "feature_columns": F.FEATURE_COLS,
        "ensemble_size": args.ensemble, "test_rmse": float(test_rmse_z),
        "mean_member_val_rmse": float(np.mean(val_scores_z)),
        "rating_scale": [RATING_MIN, RATING_MAX],
        "architecture": {"width": WIDTH, "depth": DEPTH, "dropout": DROPOUT},
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

    if not args.no_plot:
        n_col = te_meta["n"].to_numpy()
        curves = {
            "design3": rmse_by_n(ens, te_y, n_col),
            "design3_z": rmse_by_n(preds_z_raw, te_y, n_col),
            "zero": rmse_by_n(np.full(len(te_y), rows["global_mean"]), te_y, n_col),
            "movie_mean": rmse_by_n(rows["movie_mean"][te_meta["tcol"].to_numpy()], te_y, n_col),
        }
        xgb_path = MODELS / "letterboxd_xgboost.json"
        if xgb_path.exists():
            try:
                # scored in a fresh process -- see plots.score_other_design
                other_pred = score_other_design("xgb", xgb_path, te_x, F.FEATURE_COLS,
                                                RATING_MIN, RATING_MAX)
                curves["design2"] = rmse_by_n(other_pred, te_y, n_col)
            except Exception as e:
                print(f"  (scratch plot: skipping design2 curve -- {e})")
        plot_path = Path(args.plot_file) if args.plot_file else RESULTS / "figures" / "temp_design3.png"
        plot_rmse_by_n(curves, plot_path,
                       "Letterboxd Design 3: scratch RMSE by seen-count (this run)",
                       "RMSE on paired test episodes (1-10)")
        print(f"Scratch plot written to {plot_path} "
              f"(canonical figures are only produced by `make lb-analyze`)")


if __name__ == "__main__":
    main()
