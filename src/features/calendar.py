"""Academic calendar features: semester week, exam, holiday, events."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def _flag_windows(ts: pd.Series, windows: list[dict]) -> pd.Series:
    flag = pd.Series(False, index=ts.index)
    days = ts.dt.normalize()
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        flag |= (days >= start) & (days <= end)
    return flag.astype(int)


def add_calendar_features(df: pd.DataFrame, cfg: dict | None = None, ts_col: str = "timestamp") -> pd.DataFrame:
    cfg = cfg or load_config()
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    sem = cfg["semester"]

    def semester_week(t: pd.Timestamp) -> int:
        year = t.year
        if t.month >= sem["fall_start_month"]:
            start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
        elif t.month <= 6:
            start = pd.Timestamp(year=year, month=sem["spring_start_month"], day=sem["spring_start_day"])
        else:
            start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
        week = int((t.normalize() - start).days // 7) + 1
        return max(0, min(week, sem["weeks"] + 4))

    out["semester_week"] = ts.map(semester_week)
    out["is_exam_period"] = _flag_windows(ts, cfg["exam_windows"])
    out["is_academic_event"] = _flag_windows(ts, cfg["academic_events"])
    holidays = set(pd.to_datetime(cfg["holidays"]).normalize())
    out["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    out["is_in_semester"] = ((out["semester_week"] >= 1) & (out["semester_week"] <= sem["weeks"])).astype(int)
    return out
