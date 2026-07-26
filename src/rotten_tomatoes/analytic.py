"""Design 1 model: the movie-mean-centered, magnitude-scaled analytic formula,
plus the shrinkage and the top-k baseline it is compared against.
"""
import numpy as np

from rotten_tomatoes.pseudo_users import Split


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


def predict_movie_topk_abs(sp: Split, upos: int, sim: np.ndarray, mag_sim: np.ndarray,
                          target_col: int, k: int = 10):
    """A new variation of Design 1 (the full-neighbourhood formula above is
    unchanged): restrict the neighbourhood to the ``k`` reviewers with the
    largest |sim| -- both strongly aligned (sim > 0) and strongly
    anti-aligned (sim < 0) critics, not just positively-aligned ones like
    `topk_mean`'s baseline -- then apply the identical movie-mean-centered,
    magnitude-scaled formula from `predict_movies`, restricted to that
    smaller peer set. One target movie at a time (offline test scoring
    already calls `predict_movies` per single target).

    Returns (pred, den, mean_of_topk, n_topk). ``pred`` is UNCLIPPED -- the
    raw-track caller clips to [0, 5]; the z-track caller must convert back to
    the raw scale (mu + sigma * pred) before clipping, exactly like
    `predict_movies`/`prediction_for_target_z` already do.
    """
    TT = sp.TT
    lo, hi = TT.indptr[target_col], TT.indptr[target_col + 1]
    crit = TT.indices[lo:hi]
    vals = TT.data[lo:hi]
    keep = crit != upos
    crit, vals = crit[keep], vals[keep]
    if len(crit) == 0:
        return 0.0, 0.0, 0.0, 0
    s_all = sim[crit]
    mag_all = np.asarray(mag_sim, dtype=float)[crit].copy()
    mag_all[~np.isfinite(mag_all)] = 1.0
    order = np.argsort(-np.abs(s_all))[:k]
    s, mag, v = s_all[order], mag_all[order], vals[order]
    mean_topk = float(v.mean())
    weight = np.abs(s)
    den = float(weight.sum())
    if den <= 0:
        return mean_topk, 0.0, mean_topk, len(order)
    num = float(((weight * mean_topk + s * (v - mean_topk)) * mag).sum())
    return float(num / den), den, mean_topk, len(order)


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
