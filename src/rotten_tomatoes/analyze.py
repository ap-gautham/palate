"""Final analysis: headline plots, dispersion-stratified tables, secondary
metrics. Consumes nsweep_records.parquet (Design 1 + baselines) and, when
present, design2_test_predictions.parquet.

Outputs:
  results/figures/plotA_rmse_vs_n.png
  results/figures/plotB_gain_vs_dispersion.png
  results/tables/stratified_rmse.csv
  results/tables/sampling_ablation.csv
  results/tables/secondary_metrics.csv
  results/tables/final_summary.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rotten_tomatoes.config import FIGURES, TABLES, VALUE_COL


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))

# Human-readable name of the value being predicted/scored.
VALUE_NAME = "ratings (0–5)" if VALUE_COL == "score_std" else "z"

# dataviz reference palette (validated, light mode)
C = {"design1": "#2a78d6", "design1_topk": "#7a4fd6", "design2": "#008300", "design3": "#4a3aa7",
     "tomatometer_z": "#eb6834", "critic_mean": "#8a8984",
     "topk_similar_mean": "#1baf7a", "zero": "#e87ba4"}
LABEL = {"design1": "Similarity model (analytic)",
         "design1_topk": "Similarity model · top-|sim| (k=10)",
         "design2": "XGBoost",
         "design3": "Neural network",
         "tomatometer_z": "Baseline: Tomatometer → score*",
         "critic_mean": "Baseline: mean of all reviewers",
         "topk_similar_mean": "Baseline: mean of top-10 similar",
         "zero": "Baseline: global mean score"}
SURFACE = "#fcfcfb"
N_ORDER = [3, 5, 10, 20, 50, -1]
N_TICK = ["3", "5", "10", "20", "50", "all"]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d8d7d3")
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)


def per_draw_rmse(g, col):
    return g.groupby("draw").apply(lambda d: rmse(d[col] - d["y"]),
                                   include_groups=False)


def load():
    rec = pd.read_parquet(TABLES / "nsweep_records.parquet")
    d2p = TABLES / "design2_test_predictions.parquet"
    d2 = pd.read_parquet(d2p) if d2p.exists() else None
    d3p = TABLES / "design3_test_predictions.parquet"
    d3 = pd.read_parquet(d3p) if d3p.exists() else None
    return rec, d2, d3


def plot_a(rec, d2, d3):
    """Clean headline figure: the consensus baseline (dashed) plus the three
    models, nothing else -- the other baselines and the top-|sim| variant live
    in the tables."""
    pop = rec[rec["sampling"] == "pop"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    curves = {}
    mean, sd = [], []
    for n in N_ORDER:
        pd_r = per_draw_rmse(pop[pop["n"] == n], "critic_mean")
        mean.append(pd_r.mean()); sd.append(pd_r.std(ddof=0))
    curves["critic_mean"] = (np.array(mean), np.array(sd))
    mean, sd = [], []
    for n in N_ORDER:
        pd_r = per_draw_rmse(pop[pop["n"] == n], "design1")
        mean.append(pd_r.mean()); sd.append(pd_r.std(ddof=0))
    curves["design1"] = (np.array(mean), np.array(sd))
    for key, frame, col in [("design2", d2, "pred_main"), ("design3", d3, "pred_nn")]:
        if frame is None:
            continue
        mean, sd = [], []
        for n in N_ORDER:
            pd_r = per_draw_rmse(frame[frame["n"] == n], col)
            mean.append(pd_r.mean()); sd.append(pd_r.std(ddof=0))
        curves[key] = (np.array(mean), np.array(sd))
    for m in [k for k in ["critic_mean", "design1", "design2", "design3"] if k in curves]:
        mean, sd = curves[m]
        flat = m == "critic_mean"
        ax.plot(xs, mean, color=C[m], linewidth=2,
                linestyle="--" if flat else "-",
                marker="" if flat else "o", markersize=4.5, label=LABEL[m])
        if not flat:
            ax.fill_between(xs, mean - sd, mean + sd, color=C[m], alpha=0.15,
                            linewidth=0)
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_xlabel("n = ratings the user has already given", fontsize=10,
                  color="#0b0b0b")
    ax.set_ylabel(f"RMSE on held-out {VALUE_NAME}", fontsize=10,
                  color="#0b0b0b")
    ax.set_title("Prediction error falls as a user rates more films",
                 fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "plotA_rmse_vs_n.png", facecolor=SURFACE)
    plt.close(fig)


def stratify(rec, d2, d3):
    """Dispersion terciles over unique test movies; RMSE within stratum."""
    pop = rec[rec["sampling"] == "pop"].copy()
    movie_disp = pop.drop_duplicates("tcol").set_index("tcol")["dispersion"]
    edges = movie_disp.quantile([1 / 3, 2 / 3]).to_numpy()
    def strat(d):
        return np.digitize(d, edges)  # 0 low, 1 mid, 2 high
    pop["stratum"] = strat(pop["dispersion"])
    rows = []
    for (n, s), g in pop.groupby(["n", "stratum"]):
        row = {"n": n, "stratum": ["low", "mid", "high"][s],
               "n_pred": len(g), "n_movies": g["tcol"].nunique(),
               "disp_range": f"{g['dispersion'].min():.2f}–{g['dispersion'].max():.2f}"}
        for m in ["design1", "tomatometer_z", "critic_mean", "topk_similar_mean"]:
            row[f"rmse_{m}"] = rmse(g[m] - g["y"])
        row["gain_vs_tm_pct"] = 100 * (1 - row["rmse_design1"] / row["rmse_tomatometer_z"])
        rows.append(row)
    out = pd.DataFrame(rows)
    for frame, col, tag in [(d2, "pred_main", "design2"), (d3, "pred_nn", "design3")]:
        if frame is None:
            continue
        merged = frame.merge(pop[["tcol"]].drop_duplicates().assign(
            stratum=lambda x: strat(movie_disp.reindex(x["tcol"]).to_numpy())),
            on="tcol", how="inner")
        for (n, s), g in merged.groupby(["n", "stratum"]):
            i = out[(out["n"] == n) & (out["stratum"] == ["low", "mid", "high"][s])].index
            out.loc[i, f"rmse_{tag}"] = rmse(g[col] - g["y"])
        out[f"gain_{tag}_vs_tm_pct"] = 100 * (1 - out[f"rmse_{tag}"] / out["rmse_tomatometer_z"])
    out.to_csv(TABLES / "stratified_rmse.csv", index=False)
    return out


def plot_b(strat_df):
    full = strat_df[strat_df["n"] == -1].set_index("stratum").loc[["low", "mid", "high"]]
    n50 = strat_df[strat_df["n"] == 50].set_index("stratum").loc[["low", "mid", "high"]]
    n10 = strat_df[strat_df["n"] == 10].set_index("stratum").loc[["low", "mid", "high"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(3)
    w = 0.26
    series = [("n = 10", n10, "#9ec5f4"), ("n = 50", n50, "#5598e7"),
              ("full history", full, "#2a78d6")]
    for i, (lab, d, col) in enumerate(series):
        vals = d["gain_vs_tm_pct"].to_numpy()
        bars = ax.bar(xs + (i - 1) * (w + 0.02), vals, width=w, color=col,
                      label=f"Design 1, {lab}", edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#52514e")
    ax.axhline(0, color="#52514e", linewidth=1)
    ax.set_xticks(xs, [f"{s}\ndispersion" for s in ["low", "mid", "high"]])
    ax.set_ylabel("RMSE improvement over Tomatometer (%)", fontsize=10,
                  color="#0b0b0b")
    ax.set_title("Where the formula helps: gain over the Tomatometer by\n"
                 "critic-disagreement stratum (random-holdout episodes)",
                 fontsize=12, color="#0b0b0b", loc="left", pad=10)
    style_ax(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "plotB_gain_vs_dispersion.png", facecolor=SURFACE)
    plt.close(fig)


def model_summary(rec, d2, d3):
    """Overall + full-history RMSE for every method on the shared episodes."""
    pop = rec[rec["sampling"] == "pop"]
    full = pop[pop["n"] == -1]
    rows = []
    for tag in ["design1", "design1_topk", "tomatometer_z", "critic_mean", "topk_similar_mean"]:
        rows.append({"method": tag, "overall_rmse": rmse(pop[tag] - pop["y"]),
                     "full_history_rmse": rmse(full[tag] - full["y"])})
    for tag in ["design1_z", "design1_topk_z"]:
        if tag in pop.columns:
            valid = pop.dropna(subset=[tag])
            full_valid = full.dropna(subset=[tag])
            rows.append({"method": tag, "overall_rmse": rmse(valid[tag] - valid["y"]),
                         "full_history_rmse": rmse(full_valid[tag] - full_valid["y"])})
    for tag, frame, col in [("design2", d2, "pred_main"), ("design3", d3, "pred_nn")]:
        if frame is None:
            continue
        rows.append({"method": tag, "overall_rmse": rmse(frame[col] - frame["y"]),
                     "full_history_rmse": rmse(
                         frame[frame["n"] == -1][col] - frame[frame["n"] == -1]["y"])})
        z_col = f"{col}_z"
        if z_col in frame.columns:
            rows.append({"method": f"{tag}_z", "overall_rmse": rmse(frame[z_col] - frame["y"]),
                         "full_history_rmse": rmse(
                             frame[frame["n"] == -1][z_col] - frame[frame["n"] == -1]["y"])})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "model_summary.csv", index=False)
    return out


def raw_vs_z_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Full-history raw RMSE per design x {raw, z}, side by side."""
    rows = []
    for design, raw_tag, z_tag in [("design1", "design1", "design1_z"),
                                    ("design1_topk", "design1_topk", "design1_topk_z"),
                                    ("design2", "design2", "design2_z"),
                                    ("design3", "design3", "design3_z")]:
        raw_row = summ[summ["method"] == raw_tag]
        z_row = summ[summ["method"] == z_tag]
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
    out.to_csv(TABLES / "raw_vs_z.csv", index=False)
    return out


def plot_c(rec, d2, d3):
    """Overlay each design's raw track (solid) against its z-score track
    (dashed, converted back to the raw scale) across seen-history n."""
    pop = rec[rec["sampling"] == "pop"]
    has_z = "design1_z" in pop.columns
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    series = [("design1", pop, "design1", "design1_z")]
    if d2 is not None and "pred_main_z" in d2.columns:
        series.append(("design2", d2, "pred_main", "pred_main_z"))
    if d3 is not None and "pred_nn_z" in d3.columns:
        series.append(("design3", d3, "pred_nn", "pred_nn_z"))
    plotted_any = False
    for key, frame, raw_col, z_col in series:
        raw_mean = []
        z_mean = []
        for n in N_ORDER:
            g = frame[frame["n"] == n]
            raw_mean.append(per_draw_rmse(g, raw_col).mean())
            g_valid = g.dropna(subset=[z_col])
            z_mean.append(rmse(g_valid[z_col] - g_valid["y"]) if len(g_valid) else np.nan)
        ax.plot(xs, raw_mean, color=C[key], linewidth=2, marker="o", markersize=4.5,
                label=f"{LABEL[key]} (raw)")
        ax.plot(xs, z_mean, color=C[key], linewidth=2, linestyle="--", marker="s",
                markersize=4, alpha=0.75, label=f"{LABEL[key]} (z, converted back)")
        plotted_any = True
    if not plotted_any or not has_z:
        plt.close(fig)
        return
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.set_xlabel("n = ratings the user has already given", fontsize=10, color="#0b0b0b")
    ax.set_ylabel(f"RMSE on held-out {VALUE_NAME}", fontsize=10, color="#0b0b0b")
    ax.set_title("Isolating scale: raw track vs. z-score track\n"
                 "(z-space predictions converted back to the raw scale)",
                 fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "plotC_raw_vs_z.png", facecolor=SURFACE)
    plt.close(fig)


def secondary_metrics(rec):
    full = rec[(rec["sampling"] == "pop") & (rec["n"] == -1)]
    rows = []
    for m in ["design1", "tomatometer_z", "critic_mean"]:
        sp, p10 = [], []
        for u, g in full.groupby("user"):
            if len(g) >= 5 and g["y"].nunique() > 1 and g[m].nunique() > 1:
                sp.append(stats.spearmanr(g[m], g["y"]).statistic)
            if len(g) >= 10:
                k = min(10, len(g))
                top_true = set(g.nlargest(k, "y")["tcol"])
                top_pred = set(g.nlargest(k, m)["tcol"])
                p10.append(len(top_true & top_pred) / k)
        rows.append({"method": m,
                 "mean_spearman": float(np.mean(sp)) if sp else np.nan,
                     "n_users_spearman": len(sp),
                 "precision_at_10": float(np.mean(p10)) if p10 else np.nan,
                 "n_users_p10": len(p10)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "secondary_metrics.csv", index=False)
    return out


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rec, d2, d3 = load()
    plot_a(rec, d2, d3)
    strat_df = stratify(rec, d2, d3)
    plot_b(strat_df)
    summ = model_summary(rec, d2, d3)
    sec = secondary_metrics(rec)
    plot_c(rec, d2, d3)
    raw_vs_z = raw_vs_z_table(summ)
    print("Model summary (overall / full-history RMSE):")
    print(summ.round(4).to_string(index=False))
    print("\nStratified (full history):")
    print(strat_df[strat_df["n"] == -1].round(4).to_string(index=False))
    print("\nSecondary metrics:")
    print(sec.round(4).to_string(index=False))
    if len(raw_vs_z):
        print("\nRaw vs. z-score track (full-history RMSE, both on the raw scale):")
        print(raw_vs_z.round(4).to_string(index=False))
    print("\nFigures written to", FIGURES)


if __name__ == "__main__":
    main()
