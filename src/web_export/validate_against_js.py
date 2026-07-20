"""One-off check: run the Python app-inference path (design1/2/3 predict.py)
on the exact same synthetic user as web/scripts/validate.ts and diff against
its JSON output, to confirm the TypeScript port matches to floating-point
precision. Not part of the reproducible pipeline; safe to delete after use.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from design1_analytic import predict as d1
from design2_xgboost import predict as d2
from design3_neural import predict as d3

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"

js = json.loads((ROOT / "web" / "js_validate_out.json").read_text())

scores = pd.read_parquet(DATA / "demo_scores.parquet")
critics = pd.read_parquet(DATA / "demo_critics.parquet").set_index("critic_id")
K_SHRINK = 8

user = pd.Series({row["movie_id"]: float(row["rating"]) for row in js["seen"]})
target_ids = js["target_ids"]
target_scores = scores[scores["movie_id"].isin(target_ids)]

matches = d1.critic_matches(scores, user, K_SHRINK)
formula = d1.predict(target_scores, matches)

xgb_model = d2.load_model(MODELS / "design2_xgboost.json")
xgb = d2.predict(scores, user, target_scores, critics, K_SHRINK, xgb_model)

nn_ckpt = d3.load_checkpoint(MODELS / "design3_mlp.pt")
nn = d3.predict(scores, user, target_scores, critics, K_SHRINK, nn_ckpt)

max_diffs = {"analytic": 0.0, "movie_mean": 0.0, "xgboost": 0.0, "neural_net": 0.0}
rows = []
for p in js["predictions"]:
    mid = p["movie_id"]
    d_analytic = abs(p["analytic"] - formula.loc[mid, "prediction"])
    d_mean = abs(p["movie_mean"] - formula.loc[mid, "movie_mean"])
    d_xgb = abs(p["xgboost"] - xgb.loc[mid])
    d_nn = abs(p["neural_net"] - nn.loc[mid])
    max_diffs["analytic"] = max(max_diffs["analytic"], d_analytic)
    max_diffs["movie_mean"] = max(max_diffs["movie_mean"], d_mean)
    max_diffs["xgboost"] = max(max_diffs["xgboost"], d_xgb)
    max_diffs["neural_net"] = max(max_diffs["neural_net"], d_nn)
    rows.append((mid, p["analytic"], formula.loc[mid, "prediction"],
                 p["xgboost"], xgb.loc[mid], p["neural_net"], nn.loc[mid]))

print(f"{'movie':40s} {'js_D1':>7s} {'py_D1':>7s} {'js_XGB':>7s} {'py_XGB':>7s} {'js_NN':>7s} {'py_NN':>7s}")
for mid, ja, pa, jx, px, jn, pn in rows[:12]:
    print(f"{mid[:40]:40s} {ja:7.3f} {pa:7.3f} {jx:7.3f} {px:7.3f} {jn:7.3f} {pn:7.3f}")

print("\nmax abs diffs (JS vs Python):", max_diffs)
assert all(v < 1e-3 for v in max_diffs.values()), "port mismatch!"
print("\nOK: TypeScript port matches Python to < 1e-3 on every prediction.")
