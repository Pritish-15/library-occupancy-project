from pathlib import Path

import pandas as pd

from src.calendar.store import load_calendar, save_calendar, semester_week, validate_calendar
from src.features.calendar import add_calendar_features


def test_seed_calendar_has_autumn_2026_ete():
    cal = load_calendar()
    ete = [w for w in cal["exam_windows"] if w["start"] == "2026-12-14"]
    assert ete
    assert ete[0]["end"] == "2026-12-30"
    autumn = [t for t in cal["terms"] if t["id"] == "2026-27-autumn"][0]
    assert autumn["classes_start"] == "2026-07-27"
    assert autumn["classes_end"] == "2026-12-11"


def test_semester_week_uses_term_start_not_hardcoded_month():
    cal = load_calendar()
    week1 = semester_week(pd.Timestamp("2026-07-27"), calendar=cal)
    week2 = semester_week(pd.Timestamp("2026-08-03"), calendar=cal)
    assert week1 == 1
    assert week2 == 2


def test_calendar_features_flag_ete_and_winter_break():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-10-05 10:00:00",
                "2026-12-20 10:00:00",
                "2027-01-05 10:00:00",
            ]
        }
    )
    out = add_calendar_features(df)
    assert out.loc[0, "is_academic_event"] == 1  # MTT
    assert out.loc[1, "is_exam_period"] == 1
    assert out.loc[2, "is_holiday"] == 1
    assert out.loc[2, "is_in_semester"] == 0


def test_save_calendar_roundtrip(tmp_path: Path):
    path = tmp_path / "cal.json"
    payload = {
        "terms": [
            {
                "id": "t1",
                "name": "Test term",
                "classes_start": "2027-01-11",
                "classes_end": "2027-05-01",
            }
        ],
        "exam_windows": [{"start": "2027-05-02", "end": "2027-05-15", "name": "ETE"}],
        "academic_events": [],
        "closures": [{"start": "2027-03-01", "end": "2027-03-02", "name": "Break"}],
    }
    saved = save_calendar(payload, path=path, updated_by="test")
    assert saved["updated_by"] == "test"
    loaded = load_calendar(path)
    assert loaded["terms"][0]["id"] == "t1"
    assert not validate_calendar(loaded)
