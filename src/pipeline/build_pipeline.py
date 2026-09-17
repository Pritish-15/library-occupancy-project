"""Assemble preprocessing + estimator into a serializable sklearn Pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.transformers import FEATURE_SETS
from src.models.regressors import make_xgboost


def build_regressor_pipeline(feature_set: str = "D", model=None) -> Pipeline:
    cols = FEATURE_SETS[feature_set]
    estimator = model if model is not None else make_xgboost()
    return Pipeline(
        [
            ("select", ColumnTransformer([("keep", "passthrough", cols)], remainder="drop")),
            ("model", estimator),
        ]
    )


def build_scaled_pipeline(feature_set: str = "D", model=None) -> Pipeline:
    cols = FEATURE_SETS[feature_set]
    estimator = model if model is not None else make_xgboost()
    return Pipeline(
        [
            ("select", ColumnTransformer([("keep", StandardScaler(), cols)], remainder="drop")),
            ("model", estimator),
        ]
    )
