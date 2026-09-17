"""Hour / day / weekday / weekend / month features."""

from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["day"] = ts.dt.day
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    return out
