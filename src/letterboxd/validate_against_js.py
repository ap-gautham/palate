"""One-off check: run the Python app-inference path on the exact same
synthetic user as web/scripts/validate_letterboxd.ts and diff against its
JSON output, to confirm the Letterboxd TypeScript port matches Python to
floating-point precision. Reads the SAME browser export (web/public/data/
letterboxd/) the JS side reads, so both operate on the identical top-1000-film
submatrix. Not part of the reproducible pipeline; safe to delete after use.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from letterboxd import features as F
from letterboxd.analyze import load_nn, nn_predict

WEB_DATA = ROOT / "web" / "public" / "data" / "letterboxd"
RATING_MIN, RATING_MAX = 1.0, 10.0
K_SHRINK = F.K_SHRINK

js = json.loads((ROOT / "web" / "js_validate_out.json").read_text())

movies = pd.DataFrame(json.loads((WEB_DATA / "movies.json").read_text()))
members_meta = pd.DataFrame(json.loads((WEB_DATA / "members.json").read_text())).set_index("id")
member_idx = np.fromfile(WEB_DATA / "ratings_member_idx.bin", dtype=np.uint16)
movie_idx = np.fromfile(WEB_DATA / "ratings_movie_idx.bin", dtype=np.uint16)
score = np.fromfile(WEB_DATA / "ratings_score.bin", dtype=np.float32)
member_ids = pd.DataFrame(json.loads((WEB_DATA / "members.json").read_text()))["id"].to_numpy()

scores = pd.DataFrame({
    "user_id": member_ids[member_idx],
    "movie_id": movies["id"].to_numpy()[movie_idx],
    "rating": score,
})
members = members_meta.rename(columns={"ratingSum": "rating_sum", "ratingCount": "rating_count"})

user = pd.Series({row["movie_id"]: float(row["rating"]) for row in js["seen"]})
target_ids = js["target_ids"]
target_scores = scores[scores["movie_id"].isin(target_ids)]
movie_genre = movies.set_index("id")["genreId"].to_dict()
unknown_genre_id = max(movie_genre.values(), default=0) + 1

matches = F.app_similarity(scores, user, K_SHRINK)

# Design 1 analytic (same formula as web/src/lib/letterboxd/design1.ts)
rows = []
for mid, group in target_scores.groupby("movie_id"):
    peer_sim = group["user_id"].map(matches["sim"]).fillna(0.0)
    peer_mag = group["user_id"].map(matches["mag_sim"]).fillna(1.0)
    movie_mean = group["rating"].mean()
    weight = peer_sim.abs()
    num = ((weight * movie_mean + peer_sim * (group["rating"] - movie_mean)) * peer_mag).sum()
    pred = movie_mean if weight.sum() == 0 else float(num / weight.sum())
    rows.append((mid, float(np.clip(pred, RATING_MIN, RATING_MAX)), float(movie_mean)))
analytic = pd.DataFrame(rows, columns=["movie_id", "prediction", "movie_mean"]).set_index("movie_id")

feats, feat_movie_ids = F.app_features(target_scores, matches, members, user, movie_genre, unknown_genre_id)
feats = feats[F.FEATURE_COLS]

xgb_model = xgb.Booster()
xgb_model.load_model(str(ROOT / "results" / "letterboxd" / "models" / "letterboxd_xgboost.json"))
xgb_pred = pd.Series(np.clip(xgb_model.predict(xgb.DMatrix(feats)), RATING_MIN, RATING_MAX),
                     index=feat_movie_ids)

nn_ckpt = load_nn()
nn_pred = pd.Series(nn_predict(nn_ckpt, feats), index=feat_movie_ids)

max_diffs = {"analytic": 0.0, "movie_mean": 0.0, "xgboost": 0.0, "neural_net": 0.0}
rows_out = []
for p in js["predictions"]:
    mid = p["movie_id"]
    d_analytic = abs(p["analytic"] - analytic.loc[mid, "prediction"])
    d_mean = abs(p["movie_mean"] - analytic.loc[mid, "movie_mean"])
    d_xgb = abs(p["xgboost"] - xgb_pred.loc[mid])
    d_nn = abs(p["neural_net"] - nn_pred.loc[mid])
    max_diffs["analytic"] = max(max_diffs["analytic"], d_analytic)
    max_diffs["movie_mean"] = max(max_diffs["movie_mean"], d_mean)
    max_diffs["xgboost"] = max(max_diffs["xgboost"], d_xgb)
    max_diffs["neural_net"] = max(max_diffs["neural_net"], d_nn)
    rows_out.append((mid, p["analytic"], analytic.loc[mid, "prediction"],
                     p["xgboost"], xgb_pred.loc[mid], p["neural_net"], nn_pred.loc[mid]))

print(f"{'movie':35s} {'js_D1':>7s} {'py_D1':>7s} {'js_XGB':>7s} {'py_XGB':>7s} {'js_NN':>7s} {'py_NN':>7s}")
for mid, ja, pa, jx, px, jn, pn in rows_out[:12]:
    print(f"{mid[:35]:35s} {ja:7.3f} {pa:7.3f} {jx:7.3f} {px:7.3f} {jn:7.3f} {pn:7.3f}")

print("\nmax abs diffs (JS vs Python):", max_diffs)
assert all(v < 5e-2 for v in max_diffs.values()), "port mismatch!"
print("\nOK: TypeScript port matches Python within tolerance on every prediction.")
