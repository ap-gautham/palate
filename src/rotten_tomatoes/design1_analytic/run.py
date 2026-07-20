"""Design 1 driver: sweep the shrinkage k on validation, then score the shared
paired/nested test episodes for the analytic formula and the four baselines.

Run from src/:  python -m design1_analytic.run

Outputs (results/tables/):
  ksweep_validation.csv, k_star.json, nsweep_records.parquet,
  nsweep_summary.csv, value_meta.json, random_holdout_partitions.json
"""
import json
import time

import numpy as np
import pandas as pd

from rotten_tomatoes.config import MOVIES_PARQUET, REVIEWS_PARQUET, SEED, TABLES, VALUE_COL
from .analytic import predict_movies, shrink, topk_mean
from .pseudo_users import (Split, build_split, iter_paired_episodes,
                           partition_pseudo_users, rmse, sample_random_holdout,
                           similarity, target_ok_mask)

K_GRID = [3, 5, 8, 12, 20, 30]
N_GRID = [3, 5, 10, 20, 50, None]
VALIDATION_PROFILES = 4
TOPK_B4 = 10


def prediction_for_target(sp: Split, upos: int, seen_cols, seen_values,
                          target_col: int, k: int):
    """Score one unseen target: the formula prediction plus every baseline."""
    raw_similarity, overlap, mag_sim = similarity(sp, upos, seen_cols, seen_values)
    sim = shrink(raw_similarity, overlap, k)
    target = np.array([target_col])
    pred, denominator, reviewer_mean, _ = predict_movies(sp, upos, sim, mag_sim, target)
    predicted = float(pred[0] if denominator[0] > 0 else reviewer_mean[0])
    topk = topk_mean(sp, upos, sim, target, k=TOPK_B4)
    topk_prediction = float(topk[0] if np.isfinite(topk[0]) else reviewer_mean[0])
    tomatometer = sp.tm_z[target_col]
    tomatometer_prediction = float(
        tomatometer if np.isfinite(tomatometer) else reviewer_mean[0])
    return (predicted, bool(denominator[0] <= 0), float(reviewer_mean[0]),
            topk_prediction, tomatometer_prediction, float(sp.dispersion[target_col]))


def run_validation(sp: Split, users: np.ndarray,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Sweep k on critic-disjoint validation users and random holdouts."""
    rows = []
    for upos in users:
        target_ok = target_ok_mask(sp, int(upos))
        for n in N_GRID:
            for profile in range(VALIDATION_PROFILES):
                episode = sample_random_holdout(rng, sp, int(upos), n, target_ok)
                if episode is None:
                    continue
                seen_cols, seen_values, target_col, target_value = episode
                for k in K_GRID:
                    predicted, *_ = prediction_for_target(
                        sp, int(upos), seen_cols, seen_values, target_col, k)
                    rows.append({"k": k, "n": -1 if n is None else n,
                                 "profile": profile, "user": int(upos),
                                 "sse": (predicted - target_value) ** 2, "cnt": 1})
    df = pd.DataFrame(rows)
    return (df.groupby(["k", "n"])
              .agg(sse=("sse", "sum"), cnt=("cnt", "sum"))
              .assign(rmse=lambda data: np.sqrt(data["sse"] / data["cnt"]))
              .reset_index())


def run_test(sp: Split, users: np.ndarray, k_star: int) -> pd.DataFrame:
    """Score the shared paired, nested test episodes for every method."""
    records = []
    started = time.time()
    last_user, n_users = None, 0
    for (upos, target_col, target_value, n, draw,
         seen_cols, seen_values) in iter_paired_episodes(sp, users):
        (predicted, fallback, reviewer_mean, topk_prediction,
         tomatometer_prediction, dispersion) = prediction_for_target(
            sp, upos, seen_cols, seen_values, target_col, k_star)
        records.append((upos, target_col, n, draw, "pop", target_value,
                        predicted, fallback, tomatometer_prediction,
                        reviewer_mean, topk_prediction, dispersion, len(seen_cols)))
        if upos != last_user:
            last_user, n_users = upos, n_users + 1
            if n_users % 100 == 0:
                print(f"  {n_users} paired test critics, {time.time() - started:.0f}s")
    return pd.DataFrame(records, columns=[
        "user", "tcol", "n", "draw", "sampling", "y", "design1", "fallback",
        "tomatometer_z", "critic_mean", "topk_similar_mean", "dispersion", "n_seen"])


def summarize(records: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Per-n summary over the paired episodes (baselines come out flat in n)."""
    rows = []
    methods = {
        "design1": "Design 1 (movie mean + magnitude)",
        "tomatometer_z": "B2: Tomatometer->score (reviewer fallback)",
        "critic_mean": "B3: mean of all reviewers",
        "topk_similar_mean": "B4: mean of top-10 similar",
        "zero": "B1: global mean score",
    }
    pop = records[records["sampling"] == "pop"].copy()
    pop["zero"] = global_mean
    for n, group in pop.groupby("n"):
        for method, label in methods.items():
            per_draw = group.groupby("draw").apply(
                lambda draw: rmse(draw[method] - draw["y"]), include_groups=False)
            rows.append({"n": n, "method": method, "label": label,
                         "rmse": float(per_draw.mean()),
                         "rmse_std_draws": float(per_draw.std(ddof=0)),
                         "n_pred_per_draw": int(len(group) / group["draw"].nunique()),
                         "fallback_rate": (float(group["fallback"].mean())
                                           if method == "design1" else np.nan)})
    return pd.DataFrame(rows)


def write_partition_metadata(sp: Split, partitions) -> None:
    (TABLES / "random_holdout_partitions.json").write_text(json.dumps({
        "protocol": "all_time_random_holdout",
        "partitions": {name: [str(sp.critic_index[u]) for u in users]
                       for name, users in partitions.items()},
    }, indent=2))


def main() -> None:
    rng = np.random.default_rng(SEED)
    scored = pd.read_parquet(REVIEWS_PARQUET)
    scored = scored[scored[VALUE_COL].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET)

    print("Building all-time matrix ...")
    split = build_split(scored, movies)
    partitions = partition_pseudo_users(split)
    write_partition_metadata(split, partitions)
    print(f"  pool={len(split.critic_index):,}, pseudo-users={len(split.users):,}, "
          f"train/val/test={len(partitions['train']):,}/"
          f"{len(partitions['validation']):,}/{len(partitions['test']):,}")

    print("k-sweep on validation ...")
    k_sweep = run_validation(split, partitions["validation"], rng)
    k_sweep.to_csv(TABLES / "ksweep_validation.csv", index=False)
    overall = k_sweep.groupby("k")["sse"].sum() / k_sweep.groupby("k")["cnt"].sum()
    k_star = int(np.sqrt(overall).idxmin())
    print(np.sqrt(overall).round(4).to_string())
    print(f"  selected k* = {k_star}")
    (TABLES / "k_star.json").write_text(json.dumps({
        "k_star": k_star, "protocol": "all_time_random_holdout",
        "validation_rmse_by_k": {int(k): float(v) for k, v in np.sqrt(overall).items()},
    }, indent=2))

    print("Scoring paired, nested test episodes ...")
    records = run_test(split, partitions["test"], k_star)
    records.to_parquet(TABLES / "nsweep_records.parquet", index=False)
    (TABLES / "value_meta.json").write_text(json.dumps({
        "value_col": VALUE_COL, "global_mean": split.global_mean,
        "protocol": "all_time_random_holdout", "n_pseudo_users": len(split.users),
    }, indent=2))
    summary = summarize(records, split.global_mean)
    summary.to_csv(TABLES / "nsweep_summary.csv", index=False)
    print(summary.pivot(index="n", columns="method", values="rmse").round(4).to_string())


if __name__ == "__main__":
    main()
