"""Per library/zone lag features without future leakage."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    value_col: str = "occupied_seats",
    group_cols: tuple[str, ...] = ("library_id", "zone_id"),
) -> pd.DataFrame:
    cfg_lags = lags or load_config()["lags"]
    out = df.sort_values([*group_cols, "timestamp"]).copy()
    grouped = out.groupby(list(group_cols), sort=False)[value_col]
    for lag in cfg_lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out
