"""Export the Letterboxd catalog and both trained models into compact static
files consumable by the browser (web/public/data/letterboxd/), so the
website's exact predictions can be reproduced client-side with no
Python server. Mirrors rotten_tomatoes.web_export.export; self-contained and
isolated from Rotten Tomatoes.

Formats:
- movies.json / members.json: small JSON catalogs (index position = id used
  in the binary ratings table). members.json carries each member's ALL-TIME
  rating_sum/rating_count (not restricted to the 1,000-film catalog), the
  same convention as the RT critics.json leave-one-out inputs.
- ratings_member_idx.bin, ratings_movie_idx.bin, ratings_score.bin: parallel
  flat arrays (Uint16, Uint16, Float32) for the top-1,000-film submatrix.
- xgb_model.json: the fields the native XGBoost tree-walk needs, read
  straight out of the existing letterboxd_xgboost.json dump.
- nn_meta.json + nn_weights_member{0,1,2}.bin: the neural-net ensemble's
  standardization stats and per-layer shapes/offsets, plus each member's
  flattened float32 weights in a fixed, documented order.
"""
import json
from functools import partial
from pathlib import Path

dumps = partial(json.dumps, allow_nan=False)

import numpy as np
import pandas as pd
import torch

from .config import MODELS, MOVIES_PARQUET, RATING_MAX, RATING_MIN, RATINGS_PARQUET, SEED
from . import features as F
from . import movie_features as MF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "data" / "letterboxd"
N_MOVIES = 1_000


def export_catalog():
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    _, genre_to_id, unknown_genre_id = F.make_genre_maps(movies)
    movie_genre = movies.drop_duplicates("movie_id").set_index("movie_id")["genres"].map(F._first_genre)
    movie_genre_id = movie_genre.map(genre_to_id).fillna(unknown_genre_id).astype(int)

    popularity = ratings.groupby("movie_id").size()
    top_movies = popularity.nlargest(N_MOVIES).index
    sub = ratings[ratings["movie_id"].isin(top_movies)].copy()

    movie_meta = (movies.drop_duplicates("movie_id").set_index("movie_id")
                  .reindex(top_movies)[["title", "year"]])
    movie_meta["n_scores"] = popularity.reindex(top_movies)
    movie_meta["genre_id"] = movie_genre_id.reindex(top_movies).fillna(unknown_genre_id).astype(int)
    movie_meta = movie_meta.reset_index()
    movie_index = {mid: i for i, mid in enumerate(movie_meta["movie_id"])}

    member_ids = pd.Index(sub["user_id"].drop_duplicates())
    # ALL-TIME stats (not restricted to the 1,000-film catalog) so the app's
    # leave-one-out peer mean -- and the z-score track's peer standardization --
    # matches the offline training convention. Never the visitor's own stats;
    # see pseudo_users.py's build_split docstring for why that distinction matters.
    all_time = ratings[ratings["user_id"].isin(member_ids)].groupby("user_id")["rating"].agg(
        ["sum", "size", "std"]).rename(
        columns={"sum": "rating_sum", "size": "rating_count", "std": "rating_sigma"})
    member_index = {mid: i for i, mid in enumerate(member_ids)}

    def clean(v):
        return None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))

    def clean_sigma(v):
        return None if pd.isna(v) or v <= 1e-9 else float(v)

    # Rich movie facets (gsimonx37 join): affinity sets as raw string arrays
    # (browser compares by string equality, mirroring the Python frozenset
    # intersection in features.py's _facet_tail), plus genre/decade multi-hot
    # ids from the SAME fixed training vocab so mh_genre_i/mh_decade_i line up
    # with the trained models' columns.
    mf = F.load_project_movie_facets(movies)

    def facets_of(mid):
        fs = mf.facet_sets.get(mid, {})
        return {f: sorted(fs.get(f, [])) for f in MF.FACETS}

    movies_json = [{
        "id": row.movie_id, "title": row.title if isinstance(row.title, str) and row.title
              else row.movie_id.replace("-", " ").title(),
        "year": clean(row.year), "genreId": int(row.genre_id), "nScores": int(row.n_scores),
        "facets": facets_of(row.movie_id),
        "genreMh": mf.genre_multihot.get(row.movie_id, []),
        "decadeMh": mf.decade_multihot.get(row.movie_id, []),
        "runtimeLog": clean(mf.runtime_log.get(row.movie_id)),
        "gsRating": clean(mf.gs_rating.get(row.movie_id)),
        "nThemesLog": float(np.log1p(mf.n_themes.get(row.movie_id, 0))),
        "nLanguagesLog": float(np.log1p(mf.n_languages.get(row.movie_id, 0))),
        "nCountriesLog": float(np.log1p(mf.n_countries.get(row.movie_id, 0))),
    } for row in movie_meta.itertuples()]

    members_json = [{
        "id": mid, "ratingSum": float(all_time.loc[mid, "rating_sum"]),
        "ratingCount": int(all_time.loc[mid, "rating_count"]),
        "ratingSigma": clean_sigma(all_time.loc[mid, "rating_sigma"]),
    } for mid in member_ids]

    (OUT / "movies.json").write_text(dumps(movies_json))
    (OUT / "members.json").write_text(dumps(members_json))

    member_idx = sub["user_id"].map(member_index).to_numpy(dtype=np.uint16)
    movie_idx = sub["movie_id"].map(movie_index).to_numpy(dtype=np.uint16)
    score = sub["rating"].to_numpy(dtype=np.float32)
    member_idx.tofile(OUT / "ratings_member_idx.bin")
    movie_idx.tofile(OUT / "ratings_movie_idx.bin")
    score.tofile(OUT / "ratings_score.bin")

    (OUT / "meta.json").write_text(dumps({
        "ratingMin": RATING_MIN, "ratingMax": RATING_MAX, "kShrink": F.K_SHRINK,
        "genreToId": genre_to_id, "unknownGenreId": unknown_genre_id,
        "membersWritten": len(member_ids), "ratingsWritten": len(sub),
    }))
    print(f"catalog: {len(movies_json)} movies, {len(members_json)} members, "
          f"{len(score)} ratings, k_shrink={F.K_SHRINK}")


def export_xgboost(model_name="letterboxd_xgboost.json", meta_name="letterboxd_xgboost_meta.json",
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


# Identical layer order convention to rotten_tomatoes.web_export.export --
# both networks share the same architecture (depth 6 residual blocks).
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


def export_neural_net(model_name="letterboxd_neural.pt", weights_prefix="nn_weights_member",
                      meta_name="nn_meta.json"):
    ckpt = torch.load(MODELS / model_name, map_location="cpu", weights_only=False)
    layers = []
    offset = 0
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
        "ratingMin": ckpt.get("rating_min", RATING_MIN),
        "ratingMax": ckpt.get("rating_max", RATING_MAX),
    }
    (OUT / meta_name).write_text(dumps(meta))
    print(f"neural net ({meta_name}): {len(ckpt['state_dicts'])} members x {offset} params, "
          f"{len(layers)} layer blocks")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export_catalog()
    export_xgboost()
    export_neural_net()
    if (MODELS / "letterboxd_xgboost_z.json").exists():
        export_xgboost("letterboxd_xgboost_z.json", "letterboxd_xgboost_z_meta.json", "xgb_z_model.json")
    if (MODELS / "letterboxd_neural_z.pt").exists():
        export_neural_net("letterboxd_neural_z.pt", "nn_z_weights_member", "nn_z_meta.json")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"\ntotal web/public/data/letterboxd size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
