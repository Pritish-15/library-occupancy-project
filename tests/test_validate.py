import pandas as pd
import pytest

from src.data.clean import clean_occupancy
from src.data.validate import ValidationError, validate_occupancy


def _frame(**kwargs):
    base = dict(
        timestamp=["2025-01-01 10:00:00", "2025-01-01 11:00:00"],
        library_id=["lib_central", "lib_central"],
        zone_id=["z_quiet", "z_quiet"],
        total_capacity=[100, 100],
        occupied_seats=[40, 55],
        available_seats=[60, 45],
    )
    base.update(kwargs)
    return pd.DataFrame(base)


def test_validate_accepts_well_formed_frame():
    out = validate_occupancy(_frame())
    assert "timestamp" in out.columns


def test_validate_rejects_missing_columns():
    with pytest.raises(ValidationError):
        validate_occupancy(pd.DataFrame({"timestamp": ["2025-01-01"]}))


def test_validate_rejects_non_positive_capacity():
    with pytest.raises(ValidationError):
        validate_occupancy(_frame(total_capacity=[0, 100]))


def test_clean_fills_missing_and_clips_capacity():
    df = _frame(occupied_seats=[40, None], available_seats=[60, None])
    df.loc[len(df)] = {
        "timestamp": "2025-01-01 12:00:00",
        "library_id": "lib_central",
        "zone_id": "z_quiet",
        "total_capacity": 100,
        "occupied_seats": 250,
        "available_seats": -150,
    }
    cleaned = clean_occupancy(df)
    assert cleaned["occupied_seats"].isna().sum() == 0
    assert (cleaned["occupied_seats"] <= cleaned["total_capacity"]).all()
    assert (cleaned["available_seats"] == cleaned["total_capacity"] - cleaned["occupied_seats"]).all()


def test_clean_fill_does_not_use_future_values():
    # A gap must be filled from the past (40), not interpolated toward the future (80).
    df = pd.DataFrame(
        dict(
            timestamp=[f"2025-01-01 {h}:00:00" for h in (10, 11, 12)],
            library_id=["lib_central"] * 3,
            zone_id=["z_quiet"] * 3,
            total_capacity=[100] * 3,
            occupied_seats=[40.0, None, 80.0],
            available_seats=[60.0, None, 20.0],
        )
    )
    cleaned = clean_occupancy(df).sort_values("timestamp")
    assert cleaned["occupied_seats"].tolist() == [40, 40, 80]
