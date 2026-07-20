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

import json

from rotten_tomatoes.config import FIGURES, TABLES, VALUE_COL


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))

# Human-readable name of the value being predicted/scored.
VALUE_NAME = "raw score (0–5)" if VALUE_COL == "score_std" else "z"
_meta_path = TABLES / "value_meta.json"
GLOBAL_MEAN = (json.loads(_meta_path.read_text())["global_mean"]
               if _meta_path.exists() else 0.0)

# dataviz reference palette (validated, light mode)
C = {"design1": "#2a78d6", "design2": "#008300", "design3": "#4a3aa7",
     "tomatometer_z": "#eb6834", "critic_mean": "#eda100",
     "topk_similar_mean": "#1baf7a", "zero": "#e87ba4"}
LABEL = {"design1": "Design 1 · movie mean + magnitude",
         "design2": "Design 2 · XGBoost",
         "design3": "Design 3 · neural net",
         "tomatometer_z": "B2 · Tomatometer → score*",
         "critic_mean": "B3 · mean of all reviewers",
         "topk_similar_mean": "B4 · mean of top-10 similar",
         "zero": "B1 · global mean score"}
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
    pop = rec[rec["sampling"] == "pop"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    curves = {}
    for m in ["zero", "tomatometer_z", "critic_mean", "topk_similar_mean", "design1"]:
        pop2 = pop.copy(); pop2["zero"] = GLOBAL_MEAN
        mean, sd = [], []
        for n in N_ORDER:
            pd_r = per_draw_rmse(pop2[pop2["n"] == n], m)
            mean.append(pd_r.mean()); sd.append(pd_r.std(ddof=0))
        curves[m] = (np.array(mean), np.array(sd))
    for key, frame, col in [("design2", d2, "pred_main"), ("design3", d3, "pred_nn")]:
        if frame is None:
            continue
        mean, sd = [], []
        for n in N_ORDER:
            pd_r = per_draw_rmse(frame[frame["n"] == n], col)
            mean.append(pd_r.mean()); sd.append(pd_r.std(ddof=0))
        curves[key] = (np.array(mean), np.array(sd))
    offsets = {"design1": 14, "design2": -14, "design3": 3,
               "topk_similar_mean": 8}
    for m in [k for k in ["zero", "tomatometer_z", "critic_mean",
                          "topk_similar_mean", "design1", "design2", "design3"]
              if k in curves]:
        mean, sd = curves[m]
        flat = m in ("zero", "tomatometer_z", "critic_mean")
        ax.plot(xs, mean, color=C[m], linewidth=2,
                linestyle="--" if flat else "-",
                marker="" if flat else "o", markersize=4.5, label=LABEL[m])
        if not flat:
            ax.fill_between(xs, mean - sd, mean + sd, color=C[m], alpha=0.15,
                            linewidth=0)
        if m in offsets:  # direct-label solid series only; dashed flats live in the legend
            ax.annotate(LABEL[m].split("·")[0].strip(), (xs[-1], mean[-1]),
                        xytext=(8, offsets[m]), textcoords="offset points",
                        fontsize=8.5, color=C[m], va="center")
    ax.set_ylim(top=max(c[0].max() for c in curves.values()) * 1.09)
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) + 1.1)
    ax.set_xlabel("n = seen ratings sampled for each fake profile", fontsize=10,
                  color="#0b0b0b")
    ax.set_ylabel(f"RMSE on random held-out {VALUE_NAME}", fontsize=10,
                  color="#0b0b0b")
    ax.set_title("Prediction quality as sampled seen history grows",
                 fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
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
    for tag in ["design1", "tomatometer_z", "critic_mean", "topk_similar_mean"]:
        rows.append({"method": tag, "overall_rmse": rmse(pop[tag] - pop["y"]),
                     "full_history_rmse": rmse(full[tag] - full["y"])})
    for tag, frame, col in [("design2", d2, "pred_main"), ("design3", d3, "pred_nn")]:
        if frame is None:
            continue
        rows.append({"method": tag, "overall_rmse": rmse(frame[col] - frame["y"]),
                     "full_history_rmse": rmse(
                         frame[frame["n"] == -1][col] - frame[frame["n"] == -1]["y"])})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "model_summary.csv", index=False)
    return out


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
    print("Model summary (overall / full-history RMSE):")
    print(summ.round(4).to_string(index=False))
    print("\nStratified (full history):")
    print(strat_df[strat_df["n"] == -1].round(4).to_string(index=False))
    print("\nSecondary metrics:")
    print(sec.round(4).to_string(index=False))
    print("\nFigures written to", FIGURES)


if __name__ == "__main__":
    main()
