"""Design 2 app inference: load the trained XGBoost model and score chosen
target films from a user's star ratings. Self-contained within the package.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from rotten_tomatoes import features as F


def load_model(path):
    # scikit-learn estimator wrapper around the saved booster: fit/predict API,
    # feature frames go in directly (no DMatrix).
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def predict(scores: pd.DataFrame, user: pd.Series, target_scores: pd.DataFrame,
            critics: pd.DataFrame, k_shrink: int, model, mf, theme_sim,
            global_std: float) -> pd.Series:
    """Return an XGBoost prediction per target movie_id (clipped to [0, 5]).
    ``mf`` is a `movie_features.MovieFacets`, ``theme_sim`` a
    `movie_features.ThemeSimilarity` (see `features.load_project_movie_facets`/
    `load_project_theme_similarity`)."""
    matches = F.app_similarity(scores, user, k_shrink)
    feats, movie_ids = F.app_features(target_scores, matches, critics, user, mf,
                                      theme_sim, global_std)
    preds = np.clip(model.predict(feats), 0.0, 5.0)
    return pd.Series(preds, index=movie_ids, name="xgboost")
