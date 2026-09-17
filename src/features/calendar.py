"""Academic calendar features from the editable JSON calendar store."""

from __future__ import annotations

import pandas as pd

from src.calendar.store import in_semester, load_calendar, semester_week
from src.config import load_config


def _flag_windows(ts: pd.Series, windows: list[dict]) -> pd.Series:
    flag = pd.Series(False, index=ts.index)
    if not windows:
        return flag.astype(int)
    days = ts.dt.normalize()
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w.get("end") or w["start"])
        flag |= (days >= start) & (days <= end)
    return flag.astype(int)


def add_calendar_features(df: pd.DataFrame, cfg: dict | None = None, ts_col: str = "timestamp") -> pd.DataFrame:
    cfg = cfg or load_config()
    calendar = load_calendar()
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["semester_week"] = ts.map(lambda t: semester_week(t, calendar=calendar, cfg=cfg))
    out["is_exam_period"] = _flag_windows(ts, calendar.get("exam_windows") or [])
    out["is_academic_event"] = _flag_windows(ts, calendar.get("academic_events") or [])
    out["is_holiday"] = _flag_windows(ts, calendar.get("closures") or [])
    out["is_in_semester"] = ts.map(lambda t: int(in_semester(t, calendar=calendar)))
    return out
