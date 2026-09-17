"""Naive occupancy baselines every ML model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_mean(train: pd.DataFrame, test: pd.DataFrame, target: str = "occupied_seats") -> np.ndarray:
    means = train.groupby(["library_id", "zone_id", "hour"])[target].mean()
    global_mean = train[target].mean()
    keys = list(zip(test["library_id"], test["zone_id"], test["hour"]))
    return np.array([means.get(k, global_mean) for k in keys], dtype=float)


def previous_interval(test: pd.DataFrame) -> np.ndarray:
    return test["lag_1"].to_numpy(dtype=float)


def same_time_previous_day(test: pd.DataFrame) -> np.ndarray:
    return test["lag_24"].to_numpy(dtype=float)


def same_time_previous_week(test: pd.DataFrame) -> np.ndarray:
    return test["lag_168"].to_numpy(dtype=float)


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str = "occupied_seats") -> dict[str, np.ndarray]:
    return {
        "historical_mean": historical_mean(train, test, target),
        "previous_interval": previous_interval(test),
        "same_time_prev_day": same_time_previous_day(test),
        "same_time_prev_week": same_time_previous_week(test),
    }
