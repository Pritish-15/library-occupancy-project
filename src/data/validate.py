"""Schema and range validation for occupancy records."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "library_id",
    "zone_id",
    "total_capacity",
    "occupied_seats",
    "available_seats",
]

OPTIONAL_CONTEXT = [
    "computer_usage",
    "study_room_usage",
    "book_demand",
    "charging_station_usage",
    "library_name",
    "zone_name",
]


class ValidationError(ValueError):
    pass


def validate_occupancy(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValidationError("timestamp contains unparseable values")

    cap = df["total_capacity"]
    if (cap <= 0).any():
        raise ValidationError("total_capacity must be positive")

    occ = df["occupied_seats"]
    avail = df["available_seats"]
    occ_ok = occ.dropna()
    if (occ_ok < 0).any():
        raise ValidationError("occupied_seats cannot be negative")

    paired = df.dropna(subset=["occupied_seats", "total_capacity"])
    if (paired["occupied_seats"] > paired["total_capacity"]).any():
        if strict:
            raise ValidationError("occupied_seats exceeds total_capacity")

    if strict:
        filled = df.dropna(subset=["occupied_seats", "available_seats", "total_capacity"])
        mismatch = filled["occupied_seats"] + filled["available_seats"] != filled["total_capacity"]
        if mismatch.any():
            raise ValidationError("occupied_seats + available_seats must equal total_capacity")

    return df
