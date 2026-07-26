"""Build the cached episode-row pool shared by Design 2 (XGBoost) and
Design 3 (neural net). Letterboxd counterpart of rotten_tomatoes.build_rows;
self-contained and isolated from Rotten Tomatoes.

Members are partitioned into train/validation/test FIRST (deterministic
seeded shuffle, see pseudo_users.partition_members); pseudo-profiles are then
amplified WITHIN each partition -- every sampled profile derived from one
member stays in that member's split, so no rating of a test member ever
reaches training. Each member draws from its own RNG stream
(``default_rng([seed, member_pos])``), so the pool is byte-identical
regardless of worker count or processing order.

Uses ALL train members (previously the trainers capped at 1500 of ~5.2k),
so the training pool grows to ~125k profiles per track.

Run from src/:  python -m letterboxd.build_rows   (or `make lb-rows`)
Outputs: data/letterboxd/processed/rows/{train,val,test}{,_z}.parquet,
         aux.npz (test mu/sigma convert-back arrays + per-movie means),
         rows_meta.json (seed + feature-contract hash for staleness checks)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .config import MOVIES_PARQUET, PROCESSED, RATINGS_PARQUET, SEED
from . import features as F
from . import pseudo_users as PU

ROWS_DIR = PROCESSED / "rows"
META_COLS = ["member", "tcol", "n", "draw", "n_seen"]
N_GRID = [3, 5, 10, 20, 50, None]
TRAIN_PROFILES_PER_N = 4
VALIDATION_PROFILES_PER_N = 2
EVAL_TARGETS_PER_MEMBER = 8
EVAL_DRAWS = 3
N_MAX_FINITE = 50


def _cols_hash() -> str:
    return hashlib.sha1(",".join(F.FEATURE_COLS).encode()).hexdigest()[:12]


def _write(name: str, frame) -> int:
    x, y, meta = frame
    df = x.copy()
    df["y"] = y
    for c in meta.columns:
        df[c] = meta[c].to_numpy()
    df.to_parquet(ROWS_DIR / f"{name}.parquet", index=False)
    return len(df)


def _read(name: str):
    df = pd.read_parquet(ROWS_DIR / f"{name}.parquet")
    return (df[F.FEATURE_COLS], df["y"].to_numpy(np.float32),
            df[[c for c in META_COLS if c in df.columns]])


def load_rows() -> dict:
    """Load the cached pool for a trainer. Raises with a pointer to
    `make lb-rows` if the cache is missing or was built for a different
    feature contract/seed."""
    meta_path = ROWS_DIR / "rows_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("no cached rows -- run `make lb-rows` first")
    meta = json.loads(meta_path.read_text())
    if meta["feature_cols_hash"] != _cols_hash() or meta["seed"] != SEED:
        raise RuntimeError("cached rows are stale (feature contract or seed "
                           "changed) -- rebuild with `make lb-rows`")
    aux = np.load(ROWS_DIR / "aux.npz")
    return {"train": _read("train"), "train_z": _read("train_z"),
            "val": _read("val"), "val_z": _read("val_z"),
            "test": _read("test"), "test_z": _read("test_z"),
            "te_mu": aux["te_mu"], "te_sigma": aux["te_sigma"],
            "movie_mean": aux["movie_mean"],
            "global_mean": float(meta["global_mean"]), "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="parallel row-generation workers")
    parser.add_argument("--profiles-per-n", type=int, default=TRAIN_PROFILES_PER_N)
    parser.add_argument("--val-profiles", type=int, default=VALIDATION_PROFILES_PER_N)
    parser.add_argument("--max-members", type=int, default=None,
                        help="cap each partition (smoke tests only)")
    args = parser.parse_args()
    if not RATINGS_PARQUET.exists():
        raise FileNotFoundError("Run python -m letterboxd.preprocess first.")

    started = time.time()
    ratings = pd.read_parquet(RATINGS_PARQUET)
    movies = pd.read_parquet(MOVIES_PARQUET)
    data = PU.build_data(ratings, movies)
    data_z = PU.build_data(ratings, movies, value="z")
    parts = PU.partition_members(data)
    if args.max_members:
        parts = {k: v[:args.max_members] for k, v in parts.items()}
    print(f"built matrix {data.n_members}x{data.n_movies}; train/val/test = "
          f"{len(parts['train'])}/{len(parts['validation'])}/{len(parts['test'])} "
          f"members ({time.time()-started:.0f}s)")

    fc = F.build_facet_context(movies, data.movies, data.global_std)
    rng = np.random.default_rng(SEED + 1)

    t0 = time.time()
    tr, tr_z, _, _ = F.generate_rows(data, parts["train"], rng, N_GRID,
                                     args.profiles_per_n, fc, data_z=data_z,
                                     jobs=args.jobs)
    print(f"  {len(tr[1]):,} train rows ({time.time()-t0:.0f}s)")
    t0 = time.time()
    va, va_z, _, _ = F.generate_rows(data, parts["validation"], rng, N_GRID,
                                     args.val_profiles, fc, data_z=data_z,
                                     jobs=args.jobs)
    print(f"  {len(va[1]):,} validation rows ({time.time()-t0:.0f}s)")
    t0 = time.time()
    te, te_z, te_mu, te_sigma = F.generate_paired_rows(
        data, parts["test"], N_GRID, EVAL_TARGETS_PER_MEMBER, EVAL_DRAWS,
        N_MAX_FINITE, fc, data_z=data_z, jobs=args.jobs)
    print(f"  {len(te[1]):,} test rows ({time.time()-t0:.0f}s)")

    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"train": _write("train", tr), "train_z": _write("train_z", tr_z),
              "val": _write("val", va), "val_z": _write("val_z", va_z),
              "test": _write("test", te), "test_z": _write("test_z", te_z)}
    np.savez(ROWS_DIR / "aux.npz", te_mu=te_mu, te_sigma=te_sigma,
             movie_mean=data.movie_mean)
    (ROWS_DIR / "rows_meta.json").write_text(json.dumps({
        "seed": SEED, "feature_cols_hash": _cols_hash(),
        "n_grid": [n if n is not None else -1 for n in N_GRID],
        "train_profiles_per_n": args.profiles_per_n,
        "val_profiles_per_n": args.val_profiles,
        "eval_targets_per_member": EVAL_TARGETS_PER_MEMBER, "eval_draws": EVAL_DRAWS,
        "n_max_finite": N_MAX_FINITE,
        "partition_sizes": {k: len(v) for k, v in parts.items()},
        "rows": counts, "global_mean": data.global_mean,
    }, indent=2))
    print(f"Wrote {ROWS_DIR} ({sum(counts.values()):,} rows total, "
          f"{time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
