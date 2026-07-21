"""Design 2 app inference: load the trained XGBoost model and score chosen
target films from a user's star ratings. Self-contained within the package.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from rotten_tomatoes import features as F


def load_model(path):
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def predict(scores: pd.DataFrame, user: pd.Series, target_scores: pd.DataFrame,
            critics: pd.DataFrame, k_shrink: int, model, mf) -> pd.Series:
    """Return an XGBoost prediction per target movie_id (clipped to [0, 5]).
    ``mf`` is a `movie_features.MovieFacets` (see `features.load_project_movie_facets`)."""
    matches = F.app_similarity(scores, user, k_shrink)
    feats, movie_ids = F.app_features(target_scores, matches, critics, user, mf)
    preds = np.clip(model.predict(xgb.DMatrix(feats)), 0.0, 5.0)
    return pd.Series(preds, index=movie_ids, name="xgboost")
