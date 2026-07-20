"""Design 3 app inference: load the trained neural-network ensemble and score
chosen target films from a user's star ratings. Self-contained.
"""
import numpy as np
import pandas as pd

from . import features as F


def load_checkpoint(path):
    import torch
    torch.set_num_threads(1)
    return torch.load(path, map_location="cpu", weights_only=False)


def predict(scores: pd.DataFrame, user: pd.Series, target_scores: pd.DataFrame,
            critics: pd.DataFrame, k_shrink: int, ckpt) -> pd.Series:
    """Return the ensemble-averaged neural-net prediction per target movie_id."""
    import torch
    from .network import TabularResNet

    matches = F.app_similarity(scores, user, k_shrink)
    feats, movie_ids = F.app_features(target_scores, matches, critics, user)

    numeric = feats[ckpt["numeric_cols"]].to_numpy(dtype=np.float32).copy()
    genre = feats["genre_id"].to_numpy(dtype=np.int64)
    log_idx = np.array([ckpt["numeric_cols"].index(c) for c in ckpt["log_cols"]])
    numeric[:, log_idx] = np.log1p(np.clip(numeric[:, log_idx], 0, None))
    nan = np.isnan(numeric)
    numeric[nan] = np.take(ckpt["mu_impute"], np.where(nan)[1])
    numeric = (numeric - ckpt["mu"]) / ckpt["sd"]

    num_t = torch.from_numpy(numeric.copy())
    gen_t = torch.from_numpy(genre.copy())
    preds = np.zeros(len(feats), dtype=np.float64)
    for state in ckpt["state_dicts"]:
        model = TabularResNet(len(ckpt["numeric_cols"]), ckpt["n_genres"],
                              ckpt["emb_dim"], ckpt["width"], ckpt["depth"],
                              ckpt["dropout"])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds += model(num_t, gen_t).numpy()
    return pd.Series(np.clip(preds / len(ckpt["state_dicts"]), 0.0, 5.0),
                     index=movie_ids, name="neural_net")
