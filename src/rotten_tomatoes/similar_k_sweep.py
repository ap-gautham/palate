"""One-off diagnostic: does knowing a target movie's "similar films" (the
K-means neighbour list used by the app's suggestion dropdown) were already
seen actually correspond to a lower prediction error? Buckets paired
(critic, n=10 seen, target) episodes -- drawn from the app's own 1,000-film
catalog -- by how many of the target's k nearest neighbours are already in
that episode's seen set, and reports Design 1's RMSE per bucket, for several
choices of k_neighbors (the neighbour-list size). The Design 1 prediction
itself does not depend on k -- only the bucketing does -- so all three k
sweeps score the SAME episodes (USERS_N x DRAWS_PER_USER), isolating exactly
the effect of how "similar" is defined.

Run from src/:  python -m rotten_tomatoes.similar_k_sweep
Outputs: results/rotten_tomatoes/figures/plotD_similar_k_sweep.png,
         results/rotten_tomatoes/tables/similar_k_sweep.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rotten_tomatoes.config import DATA_PROCESSED, FIGURES, ROOT, SEED, TABLES
from rotten_tomatoes import movie_features as MF
from .predict_analytic import critic_matches, predict

N_SEEN = 10
K_SHRINK = 8
K_VALUES = [10, 20, 30]
USERS_N = 1200
DRAWS_PER_USER = 5
BUCKET_CAP = 5   # counts >= this are pooled into a single "5+" bucket
SURFACE = "#fcfcfb"
C = {10: "#1baf7a", 20: "#2a78d6", 30: "#4a3aa7"}


def rmse(err) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def load_movies_json() -> list[dict]:
    path = ROOT / "web" / "public" / "data" / "rotten_tomatoes" / "movies.json"
    return json.loads(path.read_text())


def sample_episodes(scores: pd.DataFrame, critic_ratings: dict, id_to_pos: dict) -> list[tuple]:
    """(target_pos, seen_positions, squared_error) for Design 1, independent
    of k_neighbors -- the prediction only depends on the sampled seen set,
    never on which movies count as "similar"."""
    eligible = sorted(c for c, ratings in critic_ratings.items() if len(ratings) >= N_SEEN + 1)
    rng = np.random.default_rng(SEED)
    sample_critics = rng.choice(eligible, size=min(USERS_N, len(eligible)), replace=False)
    draw_rng = np.random.default_rng(SEED + 1)

    episodes = []
    for critic in sample_critics:
        rated_mids = list(critic_ratings[critic].keys())
        peer_scores = scores[scores["critic_id"] != critic]
        for _ in range(DRAWS_PER_USER):
            perm = draw_rng.permutation(len(rated_mids))
            target_mid = rated_mids[perm[0]]
            seen_mids = [rated_mids[i] for i in perm[1:N_SEEN + 1]]
            if target_mid not in id_to_pos:
                continue
            user_series = pd.Series({m: critic_ratings[critic][m] for m in seen_mids})
            matches = critic_matches(peer_scores, user_series, K_SHRINK)
            target_scores_df = peer_scores[peer_scores["movie_id"] == target_mid]
            if target_scores_df.empty:
                continue
            pred_df = predict(target_scores_df, matches)
            if target_mid not in pred_df.index:
                continue
            pred = float(pred_df.loc[target_mid, "prediction"])
            true = critic_ratings[critic][target_mid]
            seen_positions = frozenset(id_to_pos[m] for m in seen_mids if m in id_to_pos)
            # store the raw signed error: rmse() below squares internally, so
            # appending a squared error here would compute sqrt(mean(err^4))
            episodes.append((id_to_pos[target_mid], seen_positions, pred - true))
    return episodes


def bucket_label(count: int) -> str:
    return f"{BUCKET_CAP}+" if count >= BUCKET_CAP else str(count)


def run() -> pd.DataFrame:
    scores = pd.read_parquet(DATA_PROCESSED / "demo_scores.parquet")
    movies_json = load_movies_json()
    id_to_pos = {m["id"]: i for i, m in enumerate(movies_json)}
    consensus = {m["id"]: m["tomatometerScore"] for m in movies_json
                if m["tomatometerScore"] is not None}

    critic_ratings: dict = {}
    for row in scores.itertuples(index=False):
        critic_ratings.setdefault(row.critic_id, {})[row.movie_id] = row.score_std

    print(f"Sampling {USERS_N} critics x {DRAWS_PER_USER} draws (n={N_SEEN} seen) ...")
    episodes = sample_episodes(scores, critic_ratings, id_to_pos)
    print(f"  {len(episodes)} episodes (shared across every k)")

    rows = []
    for k in K_VALUES:
        similar = MF.top_similar(movies_json, consensus, k_neighbors=k, seed=SEED)
        similar_sets = [frozenset(s) for s in similar]
        buckets: dict[str, list[float]] = {}
        for target_pos, seen_positions, err in episodes:
            count = len(similar_sets[target_pos] & seen_positions)
            buckets.setdefault(bucket_label(count), []).append(err)
        for label, errs in buckets.items():
            rows.append({"k_neighbors": k, "similar_seen": label,
                        "n_episodes": len(errs), "rmse": rmse(errs)})

    out = pd.DataFrame(rows)
    order = [str(i) for i in range(BUCKET_CAP)] + [f"{BUCKET_CAP}+"]
    out["similar_seen"] = pd.Categorical(out["similar_seen"], categories=order, ordered=True)
    out = out.sort_values(["k_neighbors", "similar_seen"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "similar_k_sweep.csv", index=False)
    return out


def summarize_and_pick(out: pd.DataFrame) -> int:
    """Overall RMSE is identical across k (same episodes, same predictions)
    -- what differs is how cleanly the bucketing separates
    "seen similar films" from "didn't." Picks the k whose highest non-trivial
    bucket (>=3 similar films seen) shows the lowest RMSE -- the clearest
    demonstration that rating similar films actually helps -- provided that
    bucket has at least a handful of episodes."""
    best_k, best_rmse = None, None
    for k, g in out.groupby("k_neighbors"):
        high = g[g["similar_seen"].isin(["3", "4", f"{BUCKET_CAP}+"])]
        high = high[high["n_episodes"] >= 5]
        if high.empty:
            continue
        weighted = float((high["rmse"] * high["n_episodes"]).sum() / high["n_episodes"].sum())
        print(f"  k={k}: RMSE at >=3 similar seen = {weighted:.4f} "
             f"(n={int(high['n_episodes'].sum())})")
        if best_rmse is None or weighted < best_rmse:
            best_k, best_rmse = k, weighted
    return best_k


def plot(out: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    order = [str(i) for i in range(BUCKET_CAP)] + [f"{BUCKET_CAP}+"]
    xs = np.arange(len(order))
    for k, g in out.groupby("k_neighbors"):
        g = g.set_index("similar_seen").reindex(order)
        ax.plot(xs, g["rmse"], color=C[k], linewidth=2, marker="o", markersize=5,
               label=f"k_neighbors = {k}")
    ax.set_xticks(xs, order)
    ax.set_xlabel("number of the target's similar films already rated (seen set, n=10)",
                 fontsize=10, color="#0b0b0b")
    ax.set_ylabel("Design 1 RMSE", fontsize=10, color="#0b0b0b")
    ax.set_title("Does rating similar films improve the prediction?\n"
                "(same episodes; only the similarity neighbourhood size differs)",
                fontsize=12, color="#0b0b0b", loc="left", pad=12)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "plotD_similar_k_sweep.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    out = run()
    print(out.pivot(index="similar_seen", columns="k_neighbors", values="rmse").round(4).to_string())
    print("\nEpisode counts per bucket:")
    print(out.pivot(index="similar_seen", columns="k_neighbors", values="n_episodes").to_string())
    print()
    best_k = summarize_and_pick(out)
    print(f"\nBest k_neighbors: {best_k}")
    plot(out)
    print(f"Figure written to {FIGURES / 'plotD_similar_k_sweep.png'}")


if __name__ == "__main__":
    main()
