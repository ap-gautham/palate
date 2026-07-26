"""Design 3 app inference: load the trained neural-network ensemble and score
chosen target films from a user's star ratings. Self-contained.
Mirrors rotten_tomatoes/predict_neural.py.
"""
import numpy as np
import pandas as pd

from . import features as F


def load_checkpoint(path):
    import torch
    torch.set_num_threads(1)
    return torch.load(path, map_location="cpu", weights_only=False)


def predict(scores: pd.DataFrame, user: pd.Series, target_scores: pd.DataFrame,
           members: pd.DataFrame, k_shrink: int, ckpt, mf, theme_sim,
           global_std: float, rating_min: float, rating_max: float) -> pd.Series:
    """Return the ensemble-averaged neural-net prediction per target movie_id.
    ``mf`` is a `movie_features.MovieFacets``, ``theme_sim`` a
    `movie_features.ThemeSimilarity` (see `features.load_project_movie_facets`/
    `load_project_theme_similarity`)."""
    import torch
    from .network import TabularResNet

    matches = F.app_similarity(scores, user, k_shrink)
    feats, movie_ids = F.app_features(target_scores, matches, members, user,
                                      mf, theme_sim, global_std)

    numeric = feats[ckpt["numeric_cols"]].to_numpy(dtype=np.float32).copy()
    log_idx = np.array([ckpt["numeric_cols"].index(c) for c in ckpt["log_cols"]])
    numeric[:, log_idx] = np.log1p(np.clip(numeric[:, log_idx], 0, None))
    nan = np.isnan(numeric)
    numeric[nan] = np.take(ckpt["mu_impute"], np.where(nan)[1])
    numeric = (numeric - ckpt["mu"]) / ckpt["sd"]

    num_t = torch.from_numpy(numeric.copy())
    preds = np.zeros(len(feats), dtype=np.float64)
    for state in ckpt["state_dicts"]:
        model = TabularResNet(len(ckpt["numeric_cols"]), ckpt["width"],
                              ckpt["depth"], ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds += model(num_t).numpy()
    return pd.Series(np.clip(preds / len(ckpt["state_dicts"]), rating_min, rating_max),
                     index=movie_ids, name="neural_net")
