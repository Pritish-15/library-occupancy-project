"""sklearn-compatible transformers for the occupancy feature stack."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import load_config
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features
from src.features.temporal import add_temporal_features


class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_temporal_features(pd.DataFrame(X))


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict | None = None):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_calendar_features(pd.DataFrame(X), cfg=self.config or load_config())


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags: list[int] | None = None, value_col: str = "occupied_seats"):
        self.lags = lags
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_lag_features(pd.DataFrame(X), lags=self.lags, value_col=self.value_col)


FEATURE_SETS = {
    "A": ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity"],
    "B": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
    ],
    "C": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
        "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
    ],
    "D": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
        "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
        "computer_usage", "study_room_usage", "book_demand", "charging_station_usage",
        "library_code", "zone_code",
    ],
}


def encode_locations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["library_code"] = pd.Categorical(out["library_id"]).codes
    out["zone_code"] = pd.Categorical(out["zone_id"]).codes
    return out


def occupancy_level(utilization: pd.Series, thresholds: dict | None = None) -> pd.Series:
    thresholds = thresholds or load_config()["utilization_thresholds"]
    bins = [-1e-9, thresholds["low"], thresholds["moderate"], thresholds["high"], 10]
    labels = ["Low", "Moderate", "High", "Critical"]
    return pd.cut(utilization, bins=bins, labels=labels).astype(str)


def prepare_frame(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = add_temporal_features(df)
    out = add_calendar_features(out, cfg=cfg)
    out = add_lag_features(out, lags=cfg["lags"])
    out = encode_locations(out)
    out["utilization"] = out["occupied_seats"] / out["total_capacity"]
    out["occupancy_level"] = occupancy_level(out["utilization"], cfg["utilization_thresholds"])
    lag_cols = [f"lag_{k}" for k in cfg["lags"]]
    out = out.dropna(subset=lag_cols)
    return out.reset_index(drop=True)
