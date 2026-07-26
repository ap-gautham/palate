"""Design 1 model: the movie-mean-centered, magnitude-scaled analytic formula,
plus the top-|sim| variant. Mirrors rotten_tomatoes/analytic.py; this file
holds the reusable per-target-movie formula, factored out of analyze.py's
inline evaluation loop so the app-time predict_analytic.py can share it.
"""
import numpy as np


def predict_movie(sim: np.ndarray, mag_sim: np.ndarray, values: np.ndarray,
                  rating_min: float, rating_max: float) -> float:
    """Movie-mean-centered, magnitude-scaled prediction for one target movie,
    given the already-sliced peer arrays (sim, mag_sim, values) over raters
    who rated it (self excluded upstream, and the caller must ensure at least
    one other rater exists). Clipped to [rating_min, rating_max]."""
    consensus = float(values.mean())
    weight = np.abs(sim)
    num = ((weight * consensus + sim * (values - consensus)) * mag_sim).sum()
    pred = consensus if weight.sum() == 0 else float(num / weight.sum())
    return float(np.clip(pred, rating_min, rating_max))


def predict_movie_topk_abs(sim: np.ndarray, mag_sim: np.ndarray, values: np.ndarray,
                          k: int = 10) -> float:
    """A variation of the formula above: restrict the neighbourhood to the
    ``k`` raters with the largest |sim| -- both strongly aligned and strongly
    anti-aligned -- then apply the identical formula. UNCLIPPED -- the
    raw-track caller clips to [rating_min, rating_max]; the z-track caller
    must convert back to the raw scale first.
    """
    if len(values) == 0:
        return 0.0
    order = np.argsort(-np.abs(sim))[:k]
    s, m, v = sim[order], mag_sim[order], values[order]
    mean_topk = float(v.mean())
    weight = np.abs(s)
    den = float(weight.sum())
    if den <= 0:
        return mean_topk
    return float(((weight * mean_topk + s * (v - mean_topk)) * m).sum() / den)
