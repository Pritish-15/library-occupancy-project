"""Regression models for occupancy forecasting."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def make_linear(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_random_forest(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=random_state,
    )


def make_xgboost(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )
