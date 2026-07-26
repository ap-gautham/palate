"""Letterboxd final analysis: the paired/nested seen-history sweep across all
three designs and baselines, styled to match the Rotten Tomatoes figures, plus
an honest cross-dataset comparison.

Mirrors rotten_tomatoes.analyze. One deterministic pass over paired
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
from . import pseudo_users as PU
from .pseudo_users import rmse
from .analytic import predict_movie, predict_movie_topk_abs

FIGURES = RESULTS / "figures"
RT_SUMMARY = RESULTS.parent / "rotten_tomatoes" / "tables" / "model_summary.csv"

N_GRID = [3, 5, 10, 20, 50, None]
N_ORDER = [3, 5, 10, 20, 50, -1]
N_TICK = ["3", "5", "10", "20", "50", "all"]
TOPK = 10
TOPK_ABS = 10   # Design 1 top-|sim| variant: the k largest-|sim| peers, +/-

# dataviz palette (matches the RT figures)
C = {"design1": "#2a78d6", "design1_topk": "#7a4fd6", "design2": "#008300", "design3": "#4a3aa7",
     "critic_mean": "#eda100", "topk_similar_mean": "#1baf7a", "zero": "#e87ba4"}
LABEL = {"design1": "Design 1 · member mean + magnitude",
         "design1_topk": "Design 1 · top-|sim| (k=10)",
         "design2": "Design 2 · XGBoost",
         "design3": "Design 3 · neural net",
         "critic_mean": "B3 · mean of all members",
         "topk_similar_mean": "B4 · mean of top-10 similar",
         "zero": "B1 · global mean score"}
FLAT = {"zero", "critic_mean"}
SURFACE = "#fcfcfb"


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
def load_xgb(name="letterboxd_xgboost.json"):
    path = MODELS / name
    if not path.exists():
        return None
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def load_nn(name="letterboxd_neural.pt"):
    path = MODELS / name
    if not path.exists():
        return None
    import torch
    torch.set_num_threads(1)
    return torch.load(path, map_location="cpu", weights_only=False)


def nn_predict_raw(ckpt, feats: pd.DataFrame) -> np.ndarray:
    """Ensemble-averaged prediction, UNCLIPPED -- valid on the raw [1,10]
    scale or on unbounded z-space, depending on which checkpoint is passed."""
    import torch
    from .network import TabularResNet
    numeric = feats[ckpt["numeric_cols"]].to_numpy(np.float32).copy()
    log_idx = np.array([ckpt["numeric_cols"].index(c) for c in ckpt["log_cols"]])
    numeric[:, log_idx] = np.log1p(np.clip(numeric[:, log_idx], 0, None))
    nan = np.isnan(numeric)
    numeric[nan] = np.take(ckpt["mu_impute"], np.where(nan)[1])
    numeric = (numeric - ckpt["mu"]) / ckpt["sd"]
    num_t = torch.from_numpy(numeric.copy())
    preds = np.zeros(len(feats), dtype=np.float64)
    for state in ckpt["state_dicts"]:
        model = TabularResNet(len(ckpt["numeric_cols"]), ckpt["width"], ckpt["depth"], ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds += model(num_t).numpy()
    return preds / len(ckpt["state_dicts"])


def nn_predict(ckpt, feats: pd.DataFrame) -> np.ndarray:
    return np.clip(nn_predict_raw(ckpt, feats), RATING_MIN, RATING_MAX)


# ---- evaluation ------------------------------------------------------------
def evaluate(data: PU.LBData, test_members: np.ndarray, xgb_model, nn_ckpt, fc: F.FacetContext,
             targets_per_user=8, draws=3, n_max_finite=50,
             data_z: PU.LBData | None = None, xgb_model_z=None, nn_ckpt_z=None):
    """One pass over paired episodes; returns a records DataFrame with per-method
    predictions and the collected feature rows for the learned models. If
    ``data_z`` is given, also computes the z-score track (converted back to
    the raw scale) for Design 1/2/3, dropping only the z columns (to NaN) for
    episodes whose seen ratings have ~zero variance."""
    records, feature_rows = [], []
    z_feature_rows, z_mus, z_sigmas, z_valid = [], [], [], []
    for (member, target_col, y, n, draw, seen_cols, seen_vals) in PU.iter_paired_episodes(
            data, test_members, N_GRID, targets_per_user, draws, n_max_finite):
        sim, mag, overlap = PU.similarity(data, seen_cols, seen_vals, member)
        raters, values = F.target_raters(data, target_col, member)
        if len(raters) < F.MIN_OTHER_REVIEWERS:
            continue
        r_sim, r_mag = sim[raters], mag[raters]
        consensus = float(values.mean())                       # B3
        b1 = data.global_mean                                  # B1
        order = np.argsort(-r_sim)[:TOPK]                      # B4
        b4 = float(values[order].mean()) if len(order) else consensus
        analytic = predict_movie(r_sim, r_mag, values, RATING_MIN, RATING_MAX)
        topk_abs = float(np.clip(predict_movie_topk_abs(r_sim, r_mag, values, k=TOPK_ABS),
                                 RATING_MIN, RATING_MAX))

        positive_overlap = overlap[overlap > 0]
        tail = {"n_observed": int(len(seen_cols)),
                "mean_overlap": float(positive_overlap.mean()) if len(positive_overlap) else 0.0,
                "max_overlap": float(overlap.max()) if len(overlap) else 0.0,
                "n_reviewers": int(len(raters)),
                "dispersion": float(data.movie_std[target_col]),
                "user_mean": float(np.mean(seen_vals))}
        tail.update(F.facet_tail_from_context(fc, seen_cols, seen_vals, target_col))
        feature_rows.append(F.main_feature_row(r_sim, F.target_deviations(data, raters, values), tail))

        row = {"member": member, "n": n, "draw": draw, "y": y,
               "zero": b1, "critic_mean": consensus, "topk_similar_mean": b4,
               "design1": analytic, "design1_topk": topk_abs}

        if data_z is not None:
            mu = float(np.mean(seen_vals))
            sigma = float(np.std(seen_vals, ddof=0))
            if sigma > 1e-9:
                seen_z = (seen_vals - mu) / sigma
                sim_z, mag_z, _ = PU.similarity(data_z, seen_cols, seen_z, member)
                raters_z, values_z = F.target_raters(data_z, target_col, member)
                if len(raters_z) >= F.MIN_OTHER_REVIEWERS:
                    rz_sim, rz_mag = sim_z[raters_z], mag_z[raters_z]
                    # z-track: unclipped on the z scale -- predict_movie clips
                    # to [RATING_MIN, RATING_MAX], which is wrong here; clip
                    # only after mu + sigma * pred converts back to raw below.
                    consensus_z = float(values_z.mean())
                    weight_z = np.abs(rz_sim)
                    num_z = ((weight_z * consensus_z + rz_sim * (values_z - consensus_z)) * rz_mag).sum()
                    analytic_z = consensus_z if weight_z.sum() == 0 else float(num_z / weight_z.sum())
                    row["design1_z"] = float(np.clip(mu + sigma * analytic_z, RATING_MIN, RATING_MAX))
                    topk_abs_z = predict_movie_topk_abs(rz_sim, rz_mag, values_z, k=TOPK_ABS)
                    row["design1_topk_z"] = float(np.clip(mu + sigma * topk_abs_z, RATING_MIN, RATING_MAX))

                    z_result = F.episode_feature_row_z(data, data_z, seen_cols, seen_vals, target_col, member, fc)
                    if z_result is not None:
                        z_row, zmu, zsigma = z_result
                        z_feature_rows.append(z_row)
                        z_mus.append(zmu)
                        z_sigmas.append(zsigma)
                        z_valid.append(True)
                    else:
                        z_valid.append(False)
                else:
                    z_valid.append(False)
            else:
                z_valid.append(False)
            if not z_valid[-1]:
                row["design1_z"] = np.nan
                row["design1_topk_z"] = np.nan

        records.append(row)
    rec = pd.DataFrame(records)
    feats = pd.DataFrame(feature_rows, columns=F.FEATURE_COLS)
    if xgb_model is not None:
        rec["design2"] = np.clip(xgb_model.predict(feats[F.FEATURE_COLS]),
                                 RATING_MIN, RATING_MAX)
    if nn_ckpt is not None:
        rec["design3"] = nn_predict(nn_ckpt, feats)

    if data_z is not None:
        z_valid = np.asarray(z_valid, dtype=bool)
        rec["design2_z"] = np.nan
        rec["design3_z"] = np.nan
        if z_feature_rows:
            z_feats = pd.DataFrame(z_feature_rows, columns=F.FEATURE_COLS)
            z_mus_arr, z_sigmas_arr = np.asarray(z_mus), np.asarray(z_sigmas)
            valid_idx = rec.index[z_valid]
            if xgb_model_z is not None:
                pred_z = xgb_model_z.predict(z_feats[F.FEATURE_COLS])
                rec.loc[valid_idx, "design2_z"] = np.clip(
                    z_mus_arr + z_sigmas_arr * pred_z, RATING_MIN, RATING_MAX)
            if nn_ckpt_z is not None:
                pred_z = nn_predict_raw(nn_ckpt_z, z_feats)
                rec.loc[valid_idx, "design3_z"] = np.clip(
                    z_mus_arr + z_sigmas_arr * pred_z, RATING_MIN, RATING_MAX)
    return rec


def per_draw_rmse(g, col):
    g = g.dropna(subset=[col, "y"])
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
    ax.set_ylim(top=max(np.nanmax(c[0]) for c in curves.values()) * 1.09)
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


def plot_c(rec, methods):
    """Overlay each design's raw track (solid) against its z-score track
    (dashed, converted back to the raw scale) across seen-history n."""
    z_pairs = [("design1", "design1_z"), ("design1_topk", "design1_topk_z"),
              ("design2", "design2_z"), ("design3", "design3_z")]
    z_pairs = [(raw, z) for raw, z in z_pairs if raw in methods and z in rec.columns
              and rec[z].notna().any()]
    if not z_pairs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    for raw_key, z_key in z_pairs:
        raw_mean, z_mean = [], []
        for n in N_ORDER:
            sub = rec[rec["n"] == n]
            raw_mean.append(per_draw_rmse(sub, raw_key).mean())
            z_mean.append(per_draw_rmse(sub, z_key).mean())
        ax.plot(xs, raw_mean, color=C[raw_key], linewidth=2, marker="o", markersize=4.5,
                label=f"{LABEL[raw_key]} (raw)")
        ax.plot(xs, z_mean, color=C[raw_key], linewidth=2, linestyle="--", marker="s",
                markersize=4, alpha=0.75, label=f"{LABEL[raw_key]} (z, converted back)")
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) - 0.4)
    ax.set_xlabel("n = seen ratings sampled for each member profile", fontsize=10, color="#0b0b0b")
    ax.set_ylabel("RMSE on random held-out rating (1–10)", fontsize=10, color="#0b0b0b")
    ax.set_title("Letterboxd: raw track vs. z-score track\n"
                 "(z-space predictions converted back to the raw scale)",
                 fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "plotC_raw_vs_z.png", facecolor=SURFACE)
    plt.close(fig)


def plot_score_distributions(data: PU.LBData, ratings: pd.DataFrame):
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


def raw_vs_z_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Full-history raw-scale RMSE per design x {raw, z}, side by side."""
    rows = []
    for design in ["design1", "design1_topk", "design2", "design3"]:
        raw_row = summ[summ["method"] == design]
        z_row = summ[summ["method"] == f"{design}_z"]
        if raw_row.empty:
            continue
        rows.append({
            "design": design,
            "raw_full_history_rmse": float(raw_row["full_history_rmse"].iloc[0]),
            "z_full_history_rmse": float(z_row["full_history_rmse"].iloc[0]) if not z_row.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["z_minus_raw"] = out["z_full_history_rmse"] - out["raw_full_history_rmse"]
    out.to_csv(RESULTS / "raw_vs_z.csv", index=False)
    return out


def main():
    started = time.time()
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    data = PU.build_data(ratings, movies)
    data_z = PU.build_data(ratings, movies, value="z")
    parts = PU.partition_members(data)
    test_members = parts["test"][:300]
    print(f"built matrix {data.n_members}x{data.n_movies}; {len(test_members)} test members "
          f"({time.time()-started:.0f}s)")
    fc = F.build_facet_context(movies, data.movies, data.global_std)

    xgb_model, nn_ckpt = load_xgb(), load_nn()
    xgb_model_z = load_xgb("letterboxd_xgboost_z.json")
    nn_ckpt_z = load_nn("letterboxd_neural_z.pt")
    rec = evaluate(data, test_members, xgb_model, nn_ckpt, fc, data_z=data_z,
                   xgb_model_z=xgb_model_z, nn_ckpt_z=nn_ckpt_z)
    print(f"evaluated {len(rec):,} episodes ({time.time()-started:.0f}s)")

    methods = ["zero", "critic_mean", "topk_similar_mean", "design1", "design1_topk"]
    if xgb_model is not None:
        methods.append("design2")
    if nn_ckpt is not None:
        methods.append("design3")
    z_methods = [f"{m}_z" for m in ["design1", "design1_topk", "design2", "design3"]
                if f"{m}_z" in rec.columns]

    sweep = sweep_table(rec, methods + z_methods)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(RESULTS / "nsweep_summary.csv")
    print("\nLetterboxd RMSE by seen-count:\n", sweep.round(4).to_string())

    full = {m: float(sweep.loc[m, "all"]) for m in methods}
    overall = {m: rmse(rec.dropna(subset=[m, "y"])[m] - rec.dropna(subset=[m, "y"])["y"])
              for m in methods}
    rows = [{"method": m, "overall_rmse": overall[m], "full_history_rmse": full[m]} for m in methods]
    for zm in z_methods:
        valid = rec.dropna(subset=[zm, "y"])
        rows.append({"method": zm, "overall_rmse": rmse(valid[zm] - valid["y"]),
                     "full_history_rmse": float(sweep.loc[zm, "all"])})
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "model_summary.csv", index=False)

    plot_a(rec, methods)
    plot_c(rec, methods)
    plot_score_distributions(data, ratings)
    cross = cross_dataset(full)
    cross.to_csv(RESULTS / "cross_dataset_comparison.csv", index=False)
    raw_vs_z = raw_vs_z_table(summ)
    print("\nCross-dataset (normalized RMSE = RMSE / rating-range):")
    print(cross.round(4).to_string(index=False))
    if len(raw_vs_z):
        print("\nRaw vs. z-score track (full-history RMSE, both on the raw scale):")
        print(raw_vs_z.round(4).to_string(index=False))
    print(f"\nDone in {time.time()-started:.0f}s. Figures in {FIGURES}")


if __name__ == "__main__":
    main()
