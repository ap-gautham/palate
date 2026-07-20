"""Design 1 model: the movie-mean-centered, magnitude-scaled analytic formula,
plus the shrinkage and the top-k baseline it is compared against.
"""
import numpy as np

from .pseudo_users import Split


def shrink(r: np.ndarray, cnt: np.ndarray, k: float) -> np.ndarray:
    """Damp low-overlap correlations toward zero: sim' = r * min(overlap, k)/k."""
    return r * np.minimum(cnt, k) / k


def predict_movies(sp: Split, upos: int, sim: np.ndarray, mag_sim: np.ndarray,
                   tcols: np.ndarray):
    """Movie-mean-centered neighbourhood prediction:

        pred[m] = Σ_c (|sim_c| * mean_m + sim_c * (r[c,m] - mean_m))
                        * mag_sim_c  /  Σ_c |sim_c|

    over other critics c who rated m. ``mean_m`` is the unweighted mean rating
    for target movie m from those critics; ``sim_c`` is the shrunk Pearson
    alignment; ``mag_sim_c`` is the overlap-derived rating-scale match. The
    denominator holds only ``|sim_c|``. Predictions are clipped to [0, 5].

    Returns (pred, den, mean_of_reviewers, n_other_reviewers).
    """
    s = sim.copy()
    s[upos] = 0.0
    mag = np.asarray(mag_sim, dtype=float).copy()
    mag[~np.isfinite(mag)] = 1.0
    mag[upos] = 1.0
    TTs = sp.TT[tcols]        # movie x critic raw scores
    TTm = sp.TTmask[tcols]
    own = np.asarray(sp.T[upos, tcols].todense()).ravel()
    own_m = np.asarray(sp.Tmask[upos, tcols].todense()).ravel()
    sums = np.asarray(TTs.sum(axis=1)).ravel() - own
    cnts = np.asarray(TTm.sum(axis=1)).ravel() - own_m
    den = np.asarray(TTm @ np.abs(s)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rev = sums / cnts
        signed_weight = s * mag
        base_weight = np.abs(s) * mag
        deviation = (np.asarray(TTs @ signed_weight).ravel()
                     - mean_rev * np.asarray(TTm @ signed_weight).ravel())
        base = mean_rev * np.asarray(TTm @ base_weight).ravel()
        pred = np.clip((base + deviation) / den, 0.0, 5.0)
    return pred, den, mean_rev, cnts


def topk_mean(sp: Split, upos: int, sim: np.ndarray, tcols: np.ndarray,
              k: int = 10) -> np.ndarray:
    """Baseline 4: unweighted mean raw score of the k most similar (positively
    aligned) critics who reviewed each movie (selection without weighting)."""
    out = np.full(len(tcols), np.nan)
    TT = sp.TT
    for i, mc in enumerate(tcols):
        lo, hi = TT.indptr[mc], TT.indptr[mc + 1]
        crit = TT.indices[lo:hi]
        vals = TT.data[lo:hi]
        keep = crit != upos
        crit, vals = crit[keep], vals[keep]
        if len(crit) == 0:
            continue
        s = sim[crit]
        top = np.argsort(-s)[:k]
        top = top[s[top] > 0]
        if len(top) == 0:
            continue
        out[i] = vals[top].mean()
    return out
