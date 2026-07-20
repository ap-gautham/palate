"""Letterboxd final analysis: the paired/nested seen-history sweep across all
three designs and baselines, styled to match the Rotten Tomatoes figures, plus
an honest cross-dataset comparison.

Mirrors rotten_tomatoes.comparison.analysis. One deterministic pass over paired
test-member episodes scores B1 global mean, B3 consensus (movie) mean, B4 top-k
similar mean, Design 1 analytic, Design 2 XGBoost, and Design 3 neural on an
identical (member, target, draw) set at every seen-count n.

Run from src/ (after train_xgboost + train_neural):  python -m letterboxd.analyze
Outputs: results/letterboxd/{nsweep_summary.csv, model_summary.csv,
         cross_dataset_comparison.csv},
         results/letterboxd/figures/{plotA_rmse_vs_n.png, score_distributions.png}
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (MODELS, MOVIES_PARQUET, RATING_MAX, RATING_MIN, RATINGS_PARQUET,
                     RESULTS, SEED)
from . import features as F

FIGURES = RESULTS / "figures"
RT_SUMMARY = RESULTS.parent / "rotten_tomatoes" / "tables" / "model_summary.csv"

N_GRID = [3, 5, 10, 20, 50, None]
N_ORDER = [3, 5, 10, 20, 50, -1]
N_TICK = ["3", "5", "10", "20", "50", "all"]
TOPK = 10

# dataviz palette (matches the RT figures)
C = {"design1": "#2a78d6", "design2": "#008300", "design3": "#4a3aa7",
     "critic_mean": "#eda100", "topk_similar_mean": "#1baf7a", "zero": "#e87ba4"}
LABEL = {"design1": "Design 1 · member mean + magnitude",
         "design2": "Design 2 · XGBoost",
         "design3": "Design 3 · neural net",
         "critic_mean": "B3 · mean of all members",
         "topk_similar_mean": "B4 · mean of top-10 similar",
         "zero": "B1 · global mean score"}
FLAT = {"zero", "critic_mean"}
SURFACE = "#fcfcfb"


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d8d7d3")
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)


# ---- model loaders ---------------------------------------------------------
def load_xgb():
    path = MODELS / "letterboxd_xgboost.json"
    if not path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def load_nn():
    path = MODELS / "letterboxd_neural.pt"
    if not path.exists():
        return None
    import torch
    torch.set_num_threads(1)
    return torch.load(path, map_location="cpu", weights_only=False)


def nn_predict(ckpt, feats: pd.DataFrame) -> np.ndarray:
    import torch
    from .network import TabularResNet
    numeric = feats[ckpt["numeric_cols"]].to_numpy(np.float32).copy()
    genre = feats["genre_id"].to_numpy(np.int64)
    log_idx = np.array([ckpt["numeric_cols"].index(c) for c in ckpt["log_cols"]])
    numeric[:, log_idx] = np.log1p(np.clip(numeric[:, log_idx], 0, None))
    nan = np.isnan(numeric)
    numeric[nan] = np.take(ckpt["mu_impute"], np.where(nan)[1])
    numeric = (numeric - ckpt["mu"]) / ckpt["sd"]
    num_t = torch.from_numpy(numeric.copy())
    gen_t = torch.from_numpy(genre.copy())
    preds = np.zeros(len(feats), dtype=np.float64)
    for state in ckpt["state_dicts"]:
        model = TabularResNet(len(ckpt["numeric_cols"]), ckpt["n_genres"], ckpt["emb_dim"],
                              ckpt["width"], ckpt["depth"], ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds += model(num_t, gen_t).numpy()
    return np.clip(preds / len(ckpt["state_dicts"]), RATING_MIN, RATING_MAX)


# ---- evaluation ------------------------------------------------------------
def evaluate(data: F.LBData, test_members: np.ndarray, xgb_model, nn_ckpt,
             targets_per_user=8, draws=3, n_max_finite=50):
    """One pass over paired episodes; returns a records DataFrame with per-method
    predictions and the collected feature rows for the learned models."""
    records, feature_rows = [], []
    for (member, target_col, y, n, draw, seen_cols, seen_vals) in F.iter_paired_episodes(
            data, test_members, N_GRID, targets_per_user, draws, n_max_finite):
        sim, mag, overlap = F.similarity(data, seen_cols, seen_vals, member)
        raters, values = F.target_raters(data, target_col, member)
        if len(raters) < F.MIN_OTHER_REVIEWERS:
            continue
        r_sim, r_mag = sim[raters], mag[raters]
        consensus = float(values.mean())                       # B3
        b1 = data.global_mean                                  # B1
        order = np.argsort(-r_sim)[:TOPK]                      # B4
        b4 = float(values[order].mean()) if len(order) else consensus
        weight = np.abs(r_sim)
        num = ((weight * consensus + r_sim * (values - consensus)) * r_mag).sum()
        analytic = consensus if weight.sum() == 0 else float(num / weight.sum())
        analytic = float(np.clip(analytic, RATING_MIN, RATING_MAX))

        positive_overlap = overlap[overlap > 0]
        tail = {"n_observed": int(len(seen_cols)),
                "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
                "max_overlap": float(overlap.max()) if len(overlap) else 0.0,
                "n_reviewers": int(len(raters)),
                "dispersion": float(data.movie_std[target_col]),
                "genre_id": int(data.genre_id[target_col]),
                "user_mean": float(np.mean(seen_vals))}
        feature_rows.append(F.main_feature_row(r_sim, F.target_deviations(data, raters, values), tail))
        records.append({"member": member, "n": n, "draw": draw, "y": y,
                        "zero": b1, "critic_mean": consensus, "topk_similar_mean": b4,
                        "design1": analytic})
    rec = pd.DataFrame(records)
    feats = pd.DataFrame(feature_rows, columns=F.FEATURE_COLS)
    feats["genre_id"] = feats["genre_id"].astype(int)
    if xgb_model is not None:
        rec["design2"] = np.clip(xgb_model.predict(xgb.DMatrix(feats[F.FEATURE_COLS])),
                                 RATING_MIN, RATING_MAX)
    if nn_ckpt is not None:
        rec["design3"] = nn_predict(nn_ckpt, feats)
    return rec


def per_draw_rmse(g, col):
    return g.groupby("draw").apply(lambda d: rmse(d[col] - d["y"]), include_groups=False)


def sweep_table(rec: pd.DataFrame, methods) -> pd.DataFrame:
    rows = {}
    for m in methods:
        means = []
        for n in N_ORDER:
            sub = rec[rec["n"] == n]
            means.append(per_draw_rmse(sub, m).mean() if len(sub) else np.nan)
        rows[m] = means
    return pd.DataFrame(rows, index=N_TICK).T


# ---- figures ---------------------------------------------------------------
def plot_a(rec, methods):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    curves = {}
    for m in methods:
        mean, sd = [], []
        for n in N_ORDER:
            sub = rec[rec["n"] == n]
            pr = per_draw_rmse(sub, m)
            mean.append(pr.mean()); sd.append(pr.std(ddof=0))
        curves[m] = (np.array(mean), np.array(sd))
    for m in methods:
        mean, sd = curves[m]
        flat = m in FLAT
        ax.plot(xs, mean, color=C[m], linewidth=2, linestyle="--" if flat else "-",
                marker="" if flat else "o", markersize=4.5, label=LABEL[m])
        if not flat:
            ax.fill_between(xs, mean - sd, mean + sd, color=C[m], alpha=0.15, linewidth=0)
    ax.set_ylim(top=max(c[0].max() for c in curves.values()) * 1.09)
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) - 0.4)
    ax.set_xlabel("n = seen ratings sampled for each member profile", fontsize=10, color="#0b0b0b")
    ax.set_ylabel("RMSE on random held-out rating (1–10)", fontsize=10, color="#0b0b0b")
    ax.set_title("Letterboxd: prediction quality as seen history grows",
                 fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "plotA_rmse_vs_n.png", facecolor=SURFACE)
    plt.close(fig)


def plot_score_distributions(data: F.LBData, ratings: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    axes[0].hist(ratings["rating"], bins=np.arange(0.5, 11.5, 1), color="#2a78d6",
                 edgecolor=SURFACE)
    axes[0].set_title("All member ratings (1–10)", fontsize=11, loc="left", color="#0b0b0b")
    member_mean = data.member_sum / np.maximum(data.member_count, 1)
    axes[1].hist(member_mean, bins=40, color="#eb6834", edgecolor=SURFACE)
    axes[1].set_title("Per-member mean rating", fontsize=11, loc="left", color="#0b0b0b")
    busy = data.movie_std[data.movie_count >= 20]
    axes[2].hist(busy, bins=40, color="#008300", edgecolor=SURFACE)
    axes[2].set_title("Per-film rating dispersion (≥20 raters)", fontsize=11, loc="left", color="#0b0b0b")
    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "score_distributions.png", facecolor=SURFACE)
    plt.close(fig)


# ---- cross-dataset honest comparison ---------------------------------------
def cross_dataset(lb_full: dict):
    rows = []
    lb_range = RATING_MAX - RATING_MIN
    for method, rmse_val in lb_full.items():
        rows.append({"dataset": "letterboxd", "scale": "1-10", "method": method,
                     "full_history_rmse": rmse_val,
                     "normalized_rmse": rmse_val / lb_range})
    if RT_SUMMARY.exists():
        rt = pd.read_csv(RT_SUMMARY)
        rt_range = 5.0
        for _, r in rt.iterrows():
            rows.append({"dataset": "rotten_tomatoes", "scale": "0-5", "method": r["method"],
                         "full_history_rmse": float(r["full_history_rmse"]),
                         "normalized_rmse": float(r["full_history_rmse"]) / rt_range})
    return pd.DataFrame(rows)


def main():
    started = time.time()
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    data = F.build_data(ratings, movies)
    parts = F.partition_members(data)
    test_members = parts["test"][:300]
    print(f"built matrix {data.n_members}x{data.n_movies}; {len(test_members)} test members "
          f"({time.time()-started:.0f}s)")

    xgb_model, nn_ckpt = load_xgb(), load_nn()
    rec = evaluate(data, test_members, xgb_model, nn_ckpt)
    print(f"evaluated {len(rec):,} episodes ({time.time()-started:.0f}s)")

    methods = ["zero", "critic_mean", "topk_similar_mean", "design1"]
    if xgb_model is not None:
        methods.append("design2")
    if nn_ckpt is not None:
        methods.append("design3")

    sweep = sweep_table(rec, methods)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(RESULTS / "nsweep_summary.csv")
    print("\nLetterboxd RMSE by seen-count:\n", sweep.round(4).to_string())

    full = {m: float(sweep.loc[m, "all"]) for m in methods}
    overall = {m: rmse(rec[m] - rec["y"]) for m in methods}
    pd.DataFrame({"method": methods,
                  "overall_rmse": [overall[m] for m in methods],
                  "full_history_rmse": [full[m] for m in methods]}
                 ).to_csv(RESULTS / "model_summary.csv", index=False)

    plot_a(rec, methods)
    plot_score_distributions(data, ratings)
    cross = cross_dataset(full)
    cross.to_csv(RESULTS / "cross_dataset_comparison.csv", index=False)
    print("\nCross-dataset (normalized RMSE = RMSE / rating-range):")
    print(cross.round(4).to_string(index=False))
    print(f"\nDone in {time.time()-started:.0f}s. Figures in {FIGURES}")


if __name__ == "__main__":
    main()
