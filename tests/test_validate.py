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
