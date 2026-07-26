"""Design 2 app inference: load the trained XGBoost model and score chosen
target films from a user's star ratings. Self-contained within the package.
Mirrors rotten_tomatoes/predict_xgboost.py.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import features as F


def load_model(path):
    # scikit-learn estimator wrapper around the saved booster: fit/predict API,
    # feature frames go in directly (no DMatrix).
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def predict(scores: pd.DataFrame, user: pd.Series, target_scores: pd.DataFrame,
           members: pd.DataFrame, k_shrink: int, model, mf, theme_sim,
           global_std: float, rating_min: float, rating_max: float) -> pd.Series:
    """Return an XGBoost prediction per target movie_id (clipped to
    [rating_min, rating_max]). ``mf`` is a `movie_features.MovieFacets``,
    ``theme_sim`` a `movie_features.ThemeSimilarity`."""
    matches = F.app_similarity(scores, user, k_shrink)
    feats, movie_ids = F.app_features(target_scores, matches, members, user,
                                      mf, theme_sim, global_std)
    preds = np.clip(model.predict(feats[F.FEATURE_COLS]), rating_min, rating_max)
    return pd.Series(preds, index=movie_ids, name="xgboost")
