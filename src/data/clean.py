"""Missing-value handling and outlier treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import validate_occupancy


NUMERIC_FILL = [
    "occupied_seats",
    "available_seats",
    "computer_usage",
    "study_room_usage",
    "book_demand",
    "charging_station_usage",
]


def clean_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_occupancy(df, strict=False).copy()
    df = df.sort_values(["library_id", "zone_id", "timestamp"])

    group_cols = ["library_id", "zone_id"]
    for col in NUMERIC_FILL:
        if col not in df.columns:
            continue
        df[col] = df.groupby(group_cols)[col].transform(lambda s: s.interpolate(limit_direction="both"))
        df[col] = df.groupby(group_cols)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(0)

    util = df["occupied_seats"] / df["total_capacity"].replace(0, np.nan)
    z = df.groupby(group_cols)["occupied_seats"].transform(
        lambda s: (s - s.median()) / (s.std(ddof=0) + 1e-6)
    )
    spike = (z.abs() > 4) & (util > 1.0)
    df.loc[spike, "occupied_seats"] = df.loc[spike, "total_capacity"]

    over = df["occupied_seats"] > df["total_capacity"]
    df.loc[over, "occupied_seats"] = df.loc[over, "total_capacity"]
    df.loc[df["occupied_seats"] < 0, "occupied_seats"] = 0
    df["available_seats"] = df["total_capacity"] - df["occupied_seats"]

    for col in NUMERIC_FILL:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)
