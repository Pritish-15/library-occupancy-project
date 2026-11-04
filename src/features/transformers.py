"""sklearn-compatible transformers for the occupancy feature stack."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import load_config
from src.features.calendar import add_calendar_features
from src.features.lags import RESOURCE_LAG_COLUMNS, add_lag_features, add_resource_lags
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
        # Resource usage is measured in the same hour as the target, so only
        # its previous-hour value is a legitimate predictor.
        "computer_usage_lag_1", "study_room_usage_lag_1", "book_demand_lag_1", "charging_station_usage_lag_1",
    ],
}

RESOURCE_COLS = ("computer_usage", "study_room_usage", "book_demand", "charging_station_usage")


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
    for col in RESOURCE_COLS:
        if col in out.columns:
            out[f"{col}_lag_1"] = out.groupby(["library_id", "zone_id"], sort=False)[col].shift(1)
    out = add_resource_lags(out)
    out = encode_locations(out)
    out["utilization"] = out["occupied_seats"] / out["total_capacity"]
    out["occupancy_level"] = occupancy_level(out["utilization"], cfg["utilization_thresholds"])
    lag_cols = [f"lag_{k}" for k in cfg["lags"]]
    lag_cols += [c for c in RESOURCE_LAG_COLUMNS if c in out.columns]
    out = out.dropna(subset=lag_cols)
    return out.reset_index(drop=True)
