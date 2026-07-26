"""One-off diagnostic: does knowing a target film's "similar films" (the
K-means neighbour list used by the app's suggestion dropdown) were already
seen actually correspond to a lower prediction error? Buckets paired
(member, n=10 seen, target) episodes -- drawn from the app's own 1,000-film
catalog -- by how many of the target's k nearest neighbours are already in
that episode's seen set, and reports Design 1's RMSE per bucket, for several
choices of k_neighbors (the neighbour-list size). The Design 1 prediction
itself does not depend on k -- only the bucketing does -- so all three k
sweeps score the SAME episodes, isolating exactly the effect of how
"similar" is defined. Mirrors rotten_tomatoes/similar_k_sweep.py.

Run from src/:  python -m letterboxd.similar_k_sweep
Outputs: results/letterboxd/figures/plotD_similar_k_sweep.png,
         results/letterboxd/similar_k_sweep.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RATING_MAX, RATING_MIN, RATINGS_PARQUET, RESULTS, ROOT, SEED
from . import movie_features as MF
from . import features as F
from .pseudo_users import K_SHRINK, rmse

FIGURES = RESULTS / "figures"
N_SEEN = 10
K_VALUES = [10, 20, 30]
USERS_N = 1200
DRAWS_PER_USER = 5
BUCKET_CAP = 5
SURFACE = "#fcfcfb"
C = {10: "#1baf7a", 20: "#2a78d6", 30: "#4a3aa7"}
N_MOVIES = 1_000


def load_movies_json() -> list[dict]:
    path = ROOT / "web" / "public" / "data" / "letterboxd" / "movies.json"
    return json.loads(path.read_text())


def analytic_predict(target_scores: pd.DataFrame, matches: pd.DataFrame) -> float | None:
    """Movie-mean-centered, magnitude-scaled analytic prediction for one
    target film. Mirrors letterboxd/validate_against_js.py's inline formula
    (the same math as web/src/lib/letterboxd/design1.ts)."""
    if target_scores.empty:
        return None
    peer_sim = target_scores["user_id"].map(matches["sim"]).fillna(0.0).to_numpy()
    peer_mag = target_scores["user_id"].map(matches["mag_sim"]).fillna(1.0).to_numpy()
    values = target_scores["rating"].to_numpy(dtype=float)
    movie_mean = float(values.mean())
    weight = np.abs(peer_sim)
    if weight.sum() == 0:
        pred = movie_mean
    else:
        num = ((weight * movie_mean + peer_sim * (values - movie_mean)) * peer_mag).sum()
        pred = float(num / weight.sum())
    return float(np.clip(pred, RATING_MIN, RATING_MAX))


def sample_episodes(sub: pd.DataFrame, member_ratings: dict, id_to_pos: dict) -> list[tuple]:
    """(target_pos, seen_positions, squared_error) for Design 1, independent
    of k_neighbors -- the prediction only depends on the sampled seen set,
    never on which movies count as "similar". Pre-groups `sub` by movie_id
    once (2.86M rows) so each draw only ever touches the ~11 small per-movie
    slices it needs, instead of re-scanning the full catalog submatrix."""
    movie_groups = {mid: g for mid, g in sub.groupby("movie_id")}
    eligible = sorted(m for m, ratings in member_ratings.items() if len(ratings) >= N_SEEN + 1)
    rng = np.random.default_rng(SEED)
    sample_members = rng.choice(eligible, size=min(USERS_N, len(eligible)), replace=False)
    draw_rng = np.random.default_rng(SEED + 1)

    episodes = []
    for i, member in enumerate(sample_members):
        rated_mids = list(member_ratings[member].keys())
        for _ in range(DRAWS_PER_USER):
            perm = draw_rng.permutation(len(rated_mids))
            target_mid = rated_mids[perm[0]]
            seen_mids = [rated_mids[j] for j in perm[1:N_SEEN + 1]]
            if target_mid not in id_to_pos:
                continue
            user_series = pd.Series({m: member_ratings[member][m] for m in seen_mids})
            overlap = pd.concat([movie_groups[m] for m in seen_mids if m in movie_groups],
                                ignore_index=True)
            overlap = overlap[overlap["user_id"] != member]
            matches = F.app_similarity(overlap, user_series, K_SHRINK)
            target_scores_df = movie_groups.get(target_mid)
            if target_scores_df is not None:
                target_scores_df = target_scores_df[target_scores_df["user_id"] != member]
            pred = analytic_predict(target_scores_df, matches) if target_scores_df is not None else None
            if pred is None:
                continue
            true = member_ratings[member][target_mid]
            seen_positions = frozenset(id_to_pos[m] for m in seen_mids if m in id_to_pos)
            # store the raw signed error: rmse() below squares internally, so
            # appending a squared error here would compute sqrt(mean(err^4))
            episodes.append((id_to_pos[target_mid], seen_positions, pred - true))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sample_members)} members sampled")
    return episodes


def bucket_label(count: int) -> str:
    return f"{BUCKET_CAP}+" if count >= BUCKET_CAP else str(count)


def run() -> pd.DataFrame:
    ratings = pd.read_parquet(RATINGS_PARQUET)
    popularity = ratings.groupby("movie_id").size()
    top_movies = popularity.nlargest(N_MOVIES).index
    sub = ratings[ratings["movie_id"].isin(top_movies)].copy()

    movies_json = load_movies_json()
    id_to_pos = {m["id"]: i for i, m in enumerate(movies_json)}
    mean_rating = sub.groupby("movie_id")["rating"].mean().to_dict()

    member_ratings: dict = {}
    for row in sub.itertuples(index=False):
        member_ratings.setdefault(row.user_id, {})[row.movie_id] = row.rating

    print(f"Sampling {USERS_N} members x {DRAWS_PER_USER} draws (n={N_SEEN} seen) ...")
    episodes = sample_episodes(sub, member_ratings, id_to_pos)
    print(f"  {len(episodes)} episodes (shared across every k)")

    rows = []
    for k in K_VALUES:
        similar = MF.top_similar(movies_json, mean_rating, k_neighbors=k, seed=SEED)
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
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "similar_k_sweep.csv", index=False)
    return out


def summarize_and_pick(out: pd.DataFrame) -> int | None:
    """See rotten_tomatoes/similar_k_sweep.py's docstring --
    identical selection rule (lowest RMSE at >=3 similar films seen, among
    buckets with at least 5 episodes)."""
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
    ax.set_title("Letterboxd: does rating similar films improve the prediction?\n"
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
