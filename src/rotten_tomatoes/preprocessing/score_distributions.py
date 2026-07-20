"""Score-distribution diagnostics for the audit.

Answers: do critic scores vary a lot, or bunch together — before and after
the per-critic z-score normalization? Produces, on the STANDARDIZED 0-5 scale
and again on the z scale:
  - a histogram of all scores
  - a histogram of per-critic mean and per-critic spread
  - a histogram of per-movie dispersion (critic disagreement)
and a table of how many movies exceed per-movie dispersion thresholds.

Called from audit.py (also runnable standalone).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rotten_tomatoes.config import FIGURES, REVIEWS_PARQUET, SCORE_LEVELS, TABLES

# dataviz reference palette (validated, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#008300"
SURFACE = "#fcfcfb"
MIN_SCORES = 5     # per-movie / per-critic minimum to be counted
MIN_CRITIC = 20    # match the z-scale pool for the per-critic panels


def _style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d8d7d3")
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5, color="#0b0b0b")
    ax.set_ylabel("count", fontsize=9.5, color="#0b0b0b")


def _triptych(all_vals, crit_std, mv_std, unit, fname, titles, bins_all,
              note_all):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    def histogram_bins(values, requested_bins):
        finite = np.asarray(values)[np.isfinite(values)]
        if len(finite) and np.ptp(finite) < 1e-9:
            center = float(finite[0])
            return np.linspace(center - 0.05, center + 0.05, requested_bins + 1)
        return requested_bins

    axes[0].hist(all_vals, bins=bins_all, color=BLUE, edgecolor=SURFACE,
                 linewidth=0.3)
    _style(axes[0], titles[0], unit)
    axes[0].annotate(note_all, (0.03, 0.95), xycoords="axes fraction",
                     va="top", fontsize=8.5, color="#52514e")

    axes[1].hist(crit_std, bins=histogram_bins(crit_std, 45), color=ORANGE, edgecolor=SURFACE,
                 linewidth=0.3)
    _style(axes[1], titles[1], f"std of one critic's own scores ({unit})")
    axes[1].annotate(f"{len(crit_std):,} critics\nmedian within-critic "
                     f"std {np.median(crit_std):.2f}",
                     (0.55, 0.95), xycoords="axes fraction", va="top",
                     fontsize=8.5, color="#52514e")

    axes[2].hist(mv_std, bins=histogram_bins(mv_std, 45), color=GREEN, edgecolor=SURFACE,
                 linewidth=0.3)
    _style(axes[2], titles[2], f"std of critics' scores on one movie ({unit})")
    med = float(np.median(mv_std))
    axes[2].axvline(med, color="#0b0b0b", linewidth=1, linestyle=":")
    axes[2].annotate(f"{len(mv_std):,} movies\nmedian dispersion "
                     f"{med:.2f} (dotted)",
                     (0.52, 0.95), xycoords="axes fraction", va="top",
                     fontsize=8.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(FIGURES / fname, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(REVIEWS_PARQUET)
    s = df[df["score_std"].notna()].copy()

    # ---- Standardized 0-5 scale ------------------------------------------
    pc = s.groupby("critic_id")["score_std"].agg(["mean", "std", "count"])
    pc = pc[pc["count"] >= MIN_CRITIC]
    pm = s.groupby("movie_id")["score_std"].agg(["mean", "std", "count"])
    pm = pm[pm["count"] >= MIN_SCORES]
    _triptych(
        s["score_std"], pc["std"], pm["std"], f"0–{SCORE_LEVELS}",
        "score_distributions.png",
        [f"All standardized scores (0–{SCORE_LEVELS})",
         f"Spread within each critic (>={MIN_CRITIC} reviews)",
         "Disagreement across critics per movie (>=5 scores)"],
        bins_all=np.arange(-0.5, SCORE_LEVELS + 1.5, 1),
        note_all=(f"n = {len(s):,} reviews\nquantized onto {SCORE_LEVELS+1} "
                  f"levels\nmedian {s['score_std'].median():.0f}, "
                  f"std {s['score_std'].std():.2f}"))

    # ---- After z-score normalization -------------------------------------
    sz = df[df["z"].notna()].copy()
    pcz = sz.groupby("critic_id")["z"].agg(["mean", "std", "count"])
    pcz = pcz[pcz["count"] >= MIN_CRITIC]
    pmz = sz.groupby("movie_id")["z"].agg(["mean", "std", "count"])
    pmz = pmz[pmz["count"] >= MIN_SCORES]
    _triptych(
        sz["z"].clip(-4, 4), pcz["std"], pmz["std"], "z",
        "zscore_distributions.png",
        ["All z-scores (per-critic normalized)",
         f"Spread within each critic in z (>={MIN_CRITIC} reviews)",
         "Disagreement across critics per movie in z (>=5 scores)"],
        bins_all=60,
        note_all=(f"n = {len(sz):,} z-scored reviews\nmean "
                  f"{sz['z'].mean():.2f}, std {sz['z'].std():.2f}\n"
                  "(clipped to +/-4 for display)"))

    # ---- Per-critic mean: harsh vs generous baselines (0-5 scale) ---------
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.hist(pc["mean"], bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    _style(ax, "Per-critic average score — harsh vs. generous baselines",
           f"a critic's mean standardized score (0–{SCORE_LEVELS})")
    ax.annotate("this spread is what per-critic z-scoring removes:\n"
                "the z panels above are centered at ~0, std ~1",
                (0.03, 0.95), xycoords="axes fraction", va="top",
                fontsize=8.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(FIGURES / "critic_mean_baselines.png", facecolor=SURFACE)
    plt.close(fig)

    # ---- Dispersion-threshold table (both scales) ------------------------
    std_rows = []
    for t in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        n = int((pm["std"] > t).sum())
        std_rows.append({"scale": f"0-{SCORE_LEVELS}", "std_gt": t,
                         "n_movies": n,
                         "pct_of_movies": round(100 * n / len(pm), 1)})
    for t in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        n = int((pmz["std"] > t).sum())
        std_rows.append({"scale": "z", "std_gt": t, "n_movies": n,
                         "pct_of_movies": round(100 * n / len(pmz), 1)})
    tbl = pd.DataFrame(std_rows)
    tbl.to_csv(TABLES / "dispersion_thresholds.csv", index=False)

    summary = {
        "standardized_scale": f"0-{SCORE_LEVELS} (heterogeneous scales quantized)",
        "n_scored_reviews": int(len(s)),
        "score_std_median": round(float(s["score_std"].median()), 2),
        "score_std_std": round(float(s["score_std"].std()), 2),
        "score_std_value_counts": {int(k): int(v) for k, v
                                   in s["score_std"].value_counts().sort_index().items()},
        "n_critics_ge20": int(len(pc)),
        "median_within_critic_std_0to5": round(float(pc["std"].median()), 3),
        "median_critic_mean_0to5": round(float(pc["mean"].median()), 2),
        "critic_mean_iqr_0to5": [round(float(pc["mean"].quantile(.25)), 2),
                                 round(float(pc["mean"].quantile(.75)), 2)],
        "n_movies_ge5": int(len(pm)),
        "median_per_movie_dispersion_0to5": round(float(pm["std"].median()), 3),
        "z_mean": round(float(sz["z"].mean()), 3),
        "z_std": round(float(sz["z"].std()), 3),
        "median_within_critic_std_z": round(float(pcz["std"].median()), 3),
        "median_per_movie_dispersion_z": round(float(pmz["std"].median()), 3),
    }
    print("Score-distribution summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nMovies by per-movie dispersion:")
    print(tbl.to_string(index=False))
    print("\nWrote score_distributions.png, zscore_distributions.png, "
          "critic_mean_baselines.png, dispersion_thresholds.csv")
    return summary


if __name__ == "__main__":
    main()
