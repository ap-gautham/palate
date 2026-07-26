"""Build the cached episode-row pool shared by Design 2 (XGBoost) and
Design 3 (neural net).

Critics are partitioned into train/validation/test FIRST (deterministic seeded
shuffle, see pseudo_users.partition_pseudo_users); pseudo-profiles are then
amplified WITHIN each partition -- every sampled profile derived from one
critic stays in that critic's split, so no rating of a test critic ever
reaches training. Each critic draws from its own RNG stream
(``default_rng([seed, critic_pos])``), so the pool is byte-identical
regardless of worker count or processing order.

Generating rows once and caching them (instead of per-trainer regeneration)
guarantees both designs train on byte-identical rows and makes a trainer
re-run seconds of loading instead of minutes of feature building.

Run from src/:  python -m rotten_tomatoes.build_rows   (or `make rt-rows`)
Outputs: data/rotten_tomatoes/processed/rows/{train,val,test}{,_z}.parquet,
         aux.npz (test mu/sigma convert-back arrays + per-movie means),
         rows_meta.json (seed + feature-contract hash for staleness checks)
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from rotten_tomatoes.config import (DATA_PROCESSED, MOVIES_PARQUET, REVIEWS_PARQUET,
                                    SEED, VALUE_COL)
from rotten_tomatoes import features as F
from rotten_tomatoes.pseudo_users import build_split, partition_pseudo_users

ROWS_DIR = DATA_PROCESSED / "rows"
META_COLS = ["user", "tcol", "n", "draw", "n_seen"]
N_GRID_TRAIN = [3, 5, 10, 20, 50, None]
TRAIN_PROFILES_PER_N = 32
VALIDATION_PROFILES_PER_N = 8


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
    `make rt-rows` if the cache is missing or was built for a different
    feature contract/seed."""
    meta_path = ROWS_DIR / "rows_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("no cached rows -- run `make rt-rows` first")
    meta = json.loads(meta_path.read_text())
    if meta["feature_cols_hash"] != _cols_hash() or meta["seed"] != SEED:
        raise RuntimeError("cached rows are stale (feature contract or seed "
                           "changed) -- rebuild with `make rt-rows`")
    aux = np.load(ROWS_DIR / "aux.npz")
    return {"train": _read("train"), "train_z": _read("train_z"),
            "val": _read("val"), "val_z": _read("val_z"),
            "test": _read("test"), "test_z": _read("test_z"),
            "te_mu": aux["te_mu"], "te_sigma": aux["te_sigma"],
            "movie_mean": aux["movie_mean"],
            "global_mean": float(meta["global_mean"]), "meta": meta}


def _movie_mean(sp) -> np.ndarray:
    """Full (not leave-one-out) all-time mean rating per target-movie position
    -- the B3-like baseline the trainers' scratch plots use; analyze.py owns
    the authoritative leave-one-out figure."""
    sums = np.asarray(sp.TT.sum(axis=1)).ravel()
    counts = np.asarray(sp.TTmask.sum(axis=1)).ravel()
    return np.divide(sums, counts, out=np.full_like(sums, sp.global_mean),
                     where=counts > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="parallel row-generation workers")
    parser.add_argument("--train-profiles", type=int, default=TRAIN_PROFILES_PER_N)
    parser.add_argument("--val-profiles", type=int, default=VALIDATION_PROFILES_PER_N)
    parser.add_argument("--max-users", type=int, default=None,
                        help="cap each partition (smoke tests only)")
    args = parser.parse_args()

    started = time.time()
    scored = pd.read_parquet(REVIEWS_PARQUET)
    # Both value columns must be present so the raw and z-space Splits share
    # an identical critic/movie index (see build_split's docstring).
    scored = scored[scored[VALUE_COL].notna() & scored["z"].notna()]
    movies = pd.read_parquet(MOVIES_PARQUET)

    print("Building all-time matrix (raw + z) ...")
    split = build_split(scored, movies)
    split_z = build_split(scored, movies, value_col="z")
    parts = partition_pseudo_users(split)
    if args.max_users:
        parts = {k: v[:args.max_users] for k, v in parts.items()}
    print(f"  train/val/test = {len(parts['train'])}/{len(parts['validation'])}"
          f"/{len(parts['test'])} critics")

    fc = F.build_facet_context(movies, split.tgt_movie_index, split.global_std)
    rng = np.random.default_rng(SEED + 1)

    t0 = time.time()
    tr, tr_z, _, _ = F.generate_rows(split, parts["train"], rng, N_GRID_TRAIN,
                                     args.train_profiles, fc, sp_z=split_z,
                                     jobs=args.jobs)
    print(f"  {len(tr[1]):,} train rows ({time.time() - t0:.0f}s)")
    t0 = time.time()
    va, va_z, _, _ = F.generate_rows(split, parts["validation"], rng, N_GRID_TRAIN,
                                     args.val_profiles, fc, sp_z=split_z,
                                     jobs=args.jobs)
    print(f"  {len(va[1]):,} validation rows ({time.time() - t0:.0f}s)")
    t0 = time.time()
    te, te_z, te_mu, te_sigma = F.generate_paired_rows(split, parts["test"], fc,
                                                       sp_z=split_z, jobs=args.jobs)
    print(f"  {len(te[1]):,} test rows ({time.time() - t0:.0f}s)")

    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"train": _write("train", tr), "train_z": _write("train_z", tr_z),
              "val": _write("val", va), "val_z": _write("val_z", va_z),
              "test": _write("test", te), "test_z": _write("test_z", te_z)}
    np.savez(ROWS_DIR / "aux.npz", te_mu=te_mu, te_sigma=te_sigma,
             movie_mean=_movie_mean(split))
    (ROWS_DIR / "rows_meta.json").write_text(json.dumps({
        "seed": SEED, "feature_cols_hash": _cols_hash(),
        "n_grid": [n if n is not None else -1 for n in N_GRID_TRAIN],
        "train_profiles_per_n": args.train_profiles,
        "val_profiles_per_n": args.val_profiles,
        "partition_sizes": {k: len(v) for k, v in parts.items()},
        "rows": counts, "global_mean": split.global_mean,
        "value_col": VALUE_COL,
    }, indent=2))
    print(f"Wrote {ROWS_DIR} ({sum(counts.values()):,} rows total, "
          f"{time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
