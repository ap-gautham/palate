"""Shared RMSE-vs-seen-count plotting, used both by `analyze.py` (the
canonical, full cross-design figures) and by each trainer's optional per-run
scratch plot (`--plot-file`, see train_xgboost.py/train_neural.py). Kept
minimal and separate from analyze.py's own richer plotting (which has
per-design baked-in labels/colors for the canonical figures) so a manual
`python -m rotten_tomatoes.train_xgboost` run gets an immediate visual
before/after without needing the full `make rt-analyze` pipeline.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_ORDER = [3, 5, 10, 20, 50, -1]
N_TICK = ["3", "5", "10", "20", "50", "all"]
SURFACE = "#fcfcfb"

# A superset of design/baseline keys either trainer's scratch plot might draw;
# analyze.py's canonical figures define their own palette independently.
PALETTE = {
    "design2": "#008300", "design2_z": "#7fcf7f",
    "design3": "#4a3aa7", "design3_z": "#a89bdb",
    "zero": "#e87ba4", "movie_mean": "#eda100",
}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d8d7d3")
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_rmse_by_n(curves: dict, path, title: str, ylabel: str):
    """``curves``: {label: {n: rmse}}, n using -1 for "all". Missing n's are
    skipped (NaN) rather than erroring, so a still-untrained design's curve
    can simply be omitted."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    xs = np.arange(len(N_ORDER))
    for label, per_n in curves.items():
        color = PALETTE.get(label)
        ys = [per_n.get(n, np.nan) for n in N_ORDER]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5, label=label)
    ax.set_xticks(xs, N_TICK)
    ax.set_xlim(-0.3, len(xs) - 0.4)
    ax.set_xlabel("n = seen ratings sampled", fontsize=10, color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=10, color="#0b0b0b")
    ax.set_title(title, fontsize=12, color="#0b0b0b", loc="left", pad=12)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def rmse_by_n(pred: np.ndarray, y: np.ndarray, n_col: np.ndarray) -> dict:
    """{n: rmse} from parallel arrays, n using -1 for "all" (matches te_meta's
    convention throughout this codebase)."""
    out = {}
    for n in N_ORDER:
        mask = n_col == n
        if mask.any():
            err = pred[mask] - y[mask]
            out[n] = float(np.sqrt(np.mean(err ** 2)))
    return out


def _score_nn(path, te_x, feature_cols, lo, hi):
    import torch
    from .network import TabularResNet
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["numeric_cols"] != feature_cols:
        return None  # checkpoint predates the current feature contract
    numeric = te_x[ckpt["numeric_cols"]].to_numpy(np.float32).copy()
    log_idx = np.array([ckpt["numeric_cols"].index(c) for c in ckpt["log_cols"]])
    numeric[:, log_idx] = np.log1p(np.clip(numeric[:, log_idx], 0, None))
    nan = np.isnan(numeric)
    numeric[nan] = np.take(ckpt["mu_impute"], np.where(nan)[1])
    numeric = (numeric - ckpt["mu"]) / ckpt["sd"]
    num_t = torch.from_numpy(numeric)
    preds = np.zeros(len(numeric), dtype=np.float64)
    for state in ckpt["state_dicts"]:
        model = TabularResNet(len(ckpt["numeric_cols"]), ckpt["width"],
                              ckpt["depth"], ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds += model(num_t).numpy()
    return np.clip(preds / len(ckpt["state_dicts"]), lo, hi)


def _score_xgb(path, te_x, feature_cols, lo, hi):
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(str(path))
    return np.clip(model.predict(te_x[feature_cols]), lo, hi)


def score_other_design(kind: str, path, te_x, feature_cols, lo, hi):
    """Score the sibling design's saved checkpoint on the test rows, in a
    CLEAN subprocess that imports only the one ML library it needs: torch and
    xgboost each bundle their own OpenMP runtime, and any process that loads
    both segfaults (or deadlocks in a kmp join barrier) the moment either does
    real parallel work. A multiprocessing pool doesn't help -- its spawned
    child re-imports the trainer module and inherits both libraries -- so this
    shells out to `python -m <package>.plots`, whose module imports neither.
    Returns None if the checkpoint was trained on a different feature
    contract."""
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    pkg = __package__  # "rotten_tomatoes" or "letterboxd"
    with tempfile.TemporaryDirectory() as td:
        te_path = Path(td) / "te.parquet"
        out_path = Path(td) / "preds.npy"
        te_x[list(feature_cols)].to_parquet(te_path)
        r = subprocess.run([sys.executable, "-m", f"{pkg}.plots", kind, str(path),
                            str(te_path), str(out_path), str(lo), str(hi)],
                           capture_output=True, text=True)
        if r.returncode == 3:
            return None  # feature-contract mismatch sentinel from _cli
        if r.returncode != 0:
            raise RuntimeError(f"sibling scorer failed: {r.stderr.strip()[-300:]}")
        return np.load(out_path)


def _cli():
    """`python -m <package>.plots <nn|xgb> <ckpt> <te.parquet> <out.npy> <lo> <hi>`
    -- the clean-subprocess entry point for score_other_design."""
    import sys
    import pandas as pd
    kind, ckpt, te_path, out_path, lo, hi = sys.argv[1:7]
    te_x = pd.read_parquet(te_path)
    fn = _score_nn if kind == "nn" else _score_xgb
    out = fn(ckpt, te_x, list(te_x.columns), float(lo), float(hi))
    if out is None:
        sys.exit(3)
    np.save(out_path, out)


if __name__ == "__main__":
    _cli()
