"""Export the app catalog and both trained models into compact static files
consumable by the browser (web/public/data/), so the website's exact
predictions can be reproduced client-side with no Python server.

Formats:
- movies.json / critics.json: small JSON catalogs (index position = id used
  in the binary ratings table).
- ratings_critic_idx.bin, ratings_movie_idx.bin, ratings_score.bin: parallel
  flat arrays (Uint16, Uint16, Float32) for the 197k (critic, movie, score)
  rows -- the same long table `predict.py` calls `scores`.
- xgb_model.json: the fields the native XGBoost tree-walk needs (base_score,
  feature order, and each tree's arrays), read straight out of the existing
  design2_xgboost.json dump.
- nn_meta.json + nn_weights_member{0,1,2}.bin: the neural-net ensemble's
  standardization stats and per-layer shapes/offsets, plus each member's
  flattened float32 weights in a fixed, documented order.
"""
import json
import re
from functools import partial
from pathlib import Path

dumps = partial(json.dumps, allow_nan=False)

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "rotten_tomatoes" / "processed"
MODELS = ROOT / "results" / "rotten_tomatoes" / "models"
OUT = ROOT / "web" / "public" / "data" / "rotten_tomatoes"


def title_from_slug(movie_id: str) -> str:
    slug = re.sub(r"^\d+-", "", movie_id)
    words = re.split(r"[_-]+", slug)
    return " ".join(w.capitalize() for w in words if w)


def export_catalog():
    scores = pd.read_parquet(DATA / "demo_scores.parquet")
    critics = pd.read_parquet(DATA / "demo_critics.parquet")

    movies = (scores.drop_duplicates("movie_id")
              [["movie_id", "title", "year", "tomatoMeter", "genre_id", "tomatometer_score"]]
              .reset_index(drop=True))
    n_scores = scores.groupby("movie_id").size().rename("n_scores")
    movies = movies.join(n_scores, on="movie_id")
    movie_index = {mid: i for i, mid in enumerate(movies["movie_id"])}

    critics = critics.reset_index(drop=True)
    critic_index = {cid: i for i, cid in enumerate(critics["critic_id"])}

    def clean(v):
        return None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))

    movies_json = [{
        "id": row.movie_id,
        "title": title_from_slug(row.movie_id) if pd.isna(row.title) else row.title,
        "year": clean(row.year), "tomatoMeter": clean(row.tomatoMeter),
        "genreId": int(row.genre_id), "tomatometerScore": clean(row.tomatometer_score),
        "nScores": int(row.n_scores),
    } for row in movies.itertuples()]

    def clean_sigma(v):
        # NaN (a single-review critic) or 0 (a perfectly constant critic)
        # can't be standardized; the browser treats null as "skip the
        # z-track prediction for this critic", mirroring the offline pipeline.
        return None if pd.isna(v) or v <= 1e-9 else float(v)

    critics_json = [{
        "id": row.critic_id, "publicationName": row.publicationName,
        "scoreCount": int(row.score_count), "scoreSum": float(row.score_sum),
        "scoreSigma": clean_sigma(row.score_std_dev),
    } for row in critics.itertuples()]

    (OUT / "movies.json").write_text(dumps(movies_json))
    (OUT / "critics.json").write_text(dumps(critics_json))

    critic_idx = scores["critic_id"].map(critic_index).to_numpy(dtype=np.uint16)
    movie_idx = scores["movie_id"].map(movie_index).to_numpy(dtype=np.uint16)
    score = scores["score_std"].to_numpy(dtype=np.float32)
    critic_idx.tofile(OUT / "ratings_critic_idx.bin")
    movie_idx.tofile(OUT / "ratings_movie_idx.bin")
    score.tofile(OUT / "ratings_score.bin")

    k_star_path = ROOT / "results" / "rotten_tomatoes" / "tables" / "k_star.json"
    k_shrink = json.loads(k_star_path.read_text())["k_star"] if k_star_path.exists() else 8
    (OUT / "k_shrink.json").write_text(dumps({"kShrink": k_shrink}))

    print(f"catalog: {len(movies_json)} movies, {len(critics_json)} critics, "
          f"{len(score)} ratings, k_shrink={k_shrink}")
    return k_shrink


def export_xgboost(model_name="design2_xgboost.json", meta_name="design2_xgboost_meta.json",
                   out_name="xgb_model.json"):
    dump = json.loads((MODELS / model_name).read_text())
    meta = json.loads((MODELS / meta_name).read_text())
    learner = dump["learner"]
    base_score = float(json.loads(learner["learner_model_param"]["base_score"])[0])
    trees = learner["gradient_booster"]["model"]["trees"]

    compact_trees = [{
        "left": t["left_children"], "right": t["right_children"],
        "splitIdx": t["split_indices"], "splitCond": t["split_conditions"],
        "defaultLeft": t["default_left"], "leafValue": t["base_weights"],
    } for t in trees]

    out = {
        "baseScore": base_score,
        "featureColumns": meta["feature_columns"],
        "genreToId": meta["genre_to_id"],
        "unknownGenreId": meta["unknown_genre_id"],
        "trees": compact_trees,
    }
    (OUT / out_name).write_text(dumps(out))
    print(f"xgboost ({out_name}): {len(compact_trees)} trees, base_score={base_score:.4f}")


LAYER_ORDER = (
    ["input_norm.weight", "input_norm.bias", "input_norm.running_mean", "input_norm.running_var",
     "embedding.weight", "proj.weight", "proj.bias"]
    + [f"blocks.{i}.net.{layer}.{field}"
       for i in range(6)
       for layer, field in [
           ("0", "weight"), ("0", "bias"),
           ("1", "weight"), ("1", "bias"), ("1", "running_mean"), ("1", "running_var"),
           ("4", "weight"), ("4", "bias"),
           ("5", "weight"), ("5", "bias"), ("5", "running_mean"), ("5", "running_var"),
       ]]
    + ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias"]
)


def export_neural_net(model_name="design3_mlp.pt", weights_prefix="nn_weights_member",
                      meta_name="nn_meta.json"):
    ckpt = torch.load(MODELS / model_name, map_location="cpu", weights_only=False)
    layers = []
    offset = 0
    for i in range(6):
        assert f"blocks.{i}.net.0.weight" in ckpt["state_dicts"][0]

    shapes = {k: tuple(v.shape) for k, v in ckpt["state_dicts"][0].items()
              if not k.endswith("num_batches_tracked")}
    for name in LAYER_ORDER:
        shape = shapes[name]
        size = int(np.prod(shape))
        layers.append({"name": name, "shape": list(shape), "offset": offset, "size": size})
        offset += size

    for m, state in enumerate(ckpt["state_dicts"]):
        buf = np.concatenate([
            state[layer["name"]].numpy().astype(np.float32).ravel()
            for layer in layers
        ])
        assert len(buf) == offset
        buf.tofile(OUT / f"{weights_prefix}{m}.bin")

    meta = {
        "numericCols": ckpt["numeric_cols"],
        "logCols": ckpt["log_cols"],
        "genreCol": ckpt["genre_col"],
        "muImpute": ckpt["mu_impute"].tolist(),
        "mu": ckpt["mu"].tolist(),
        "sd": ckpt["sd"].tolist(),
        "nGenres": int(ckpt["n_genres"]),
        "embDim": int(ckpt["emb_dim"]),
        "width": int(ckpt["width"]),
        "depth": int(ckpt["depth"]),
        "ensembleSize": len(ckpt["state_dicts"]),
        "layers": layers,
        "totalParams": offset,
    }
    (OUT / meta_name).write_text(dumps(meta))
    print(f"neural net ({meta_name}): {len(ckpt['state_dicts'])} members x {offset} params, "
          f"{len(layers)} layer blocks")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export_catalog()
    export_xgboost()
    export_neural_net()
    if (MODELS / "design2_xgboost_z.json").exists():
        export_xgboost("design2_xgboost_z.json", "design2_xgboost_z_meta.json", "xgb_z_model.json")
    if (MODELS / "design3_mlp_z.pt").exists():
        export_neural_net("design3_mlp_z.pt", "nn_z_weights_member", "nn_z_meta.json")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"\ntotal web/public/data size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
