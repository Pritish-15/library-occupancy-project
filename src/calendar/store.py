"""Editable academic calendar stored as JSON (not hardcoded in config)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, project_path

REQUIRED_TOP = ("terms", "exam_windows", "academic_events", "closures")


def calendar_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    rel = cfg.get("paths", {}).get("academic_calendar", "data/calendar/academic_calendar.json")
    return project_path(rel)


def calendar_mtime(path: Path | None = None) -> float:
    p = path or calendar_path()
    return p.stat().st_mtime if p.exists() else 0.0


def empty_calendar() -> dict[str, Any]:
    return {
        "timezone": "Asia/Kolkata",
        "updated_at": None,
        "updated_by": None,
        "notes": "",
        "terms": [],
        "exam_windows": [],
        "academic_events": [],
        "closures": [],
    }


def _as_windows(items: list) -> list[dict]:
    out = []
    for item in items or []:
        if isinstance(item, str):
            out.append({"start": item, "end": item, "name": "Holiday"})
            continue
        start = item.get("start")
        if not start:
            continue
        end = item.get("end") or start
        row = {
            "start": str(start)[:10],
            "end": str(end)[:10],
            "name": item.get("name") or "",
        }
        if item.get("term_id"):
            row["term_id"] = item["term_id"]
        if item.get("notes"):
            row["notes"] = item["notes"]
        out.append(row)
    return out


def normalize_calendar(raw: dict[str, Any]) -> dict[str, Any]:
    data = empty_calendar()
    data.update({k: v for k, v in (raw or {}).items() if k in data or k not in REQUIRED_TOP})
    data["timezone"] = raw.get("timezone") or "Asia/Kolkata"
    data["notes"] = raw.get("notes") or ""
    data["updated_at"] = raw.get("updated_at")
    data["updated_by"] = raw.get("updated_by")
    terms = []
    for t in raw.get("terms") or []:
        if not t.get("id") and not t.get("name"):
            continue
        terms.append(
            {
                "id": t.get("id") or "",
                "session": t.get("session") or "",
                "name": t.get("name") or "",
                "programmes": t.get("programmes") or "",
                "classes_start": str(t["classes_start"])[:10] if t.get("classes_start") else None,
                "classes_end": str(t["classes_end"])[:10] if t.get("classes_end") else None,
                "next_term_start": str(t["next_term_start"])[:10] if t.get("next_term_start") else None,
                "notes": t.get("notes") or "",
            }
        )
    data["terms"] = terms
    data["exam_windows"] = _as_windows(raw.get("exam_windows") or [])
    data["academic_events"] = _as_windows(raw.get("academic_events") or [])
    closures = raw.get("closures")
    if closures is None:
        closures = raw.get("holidays") or []
    data["closures"] = _as_windows(closures)
    return data


def validate_calendar(data: dict[str, Any]) -> list[str]:
    errors = []
    for key in REQUIRED_TOP:
        if key not in data or not isinstance(data[key], list):
            errors.append(f"Missing list: {key}")
    for label, rows in (
        ("exam_windows", data.get("exam_windows") or []),
        ("academic_events", data.get("academic_events") or []),
        ("closures", data.get("closures") or []),
    ):
        for i, row in enumerate(rows):
            try:
                start = pd.Timestamp(row["start"])
                end = pd.Timestamp(row["end"])
            except Exception:
                errors.append(f"{label}[{i}] has an invalid date")
                continue
            if end < start:
                errors.append(f"{label}[{i}] ends before it starts")
    for i, term in enumerate(data.get("terms") or []):
        start, end = term.get("classes_start"), term.get("classes_end")
        if start and end and pd.Timestamp(end) < pd.Timestamp(start):
            errors.append(f"terms[{i}] classes_end is before classes_start")
    return errors


def load_calendar(path: Path | None = None) -> dict[str, Any]:
    p = path or calendar_path()
    if not p.exists():
        return _from_config_fallback()
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return normalize_calendar(raw)


def save_calendar(data: dict[str, Any], path: Path | None = None, updated_by: str = "admin") -> dict[str, Any]:
    normalized = normalize_calendar(data)
    errors = validate_calendar(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    normalized["updated_at"] = datetime.now(timezone.utc).isoformat()
    normalized["updated_by"] = updated_by
    p = path or calendar_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(normalized, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return normalized


def _from_config_fallback() -> dict[str, Any]:
    cfg = load_config()
    data = empty_calendar()
    sem = cfg.get("semester") or {}
    data["exam_windows"] = _as_windows(cfg.get("exam_windows") or [])
    data["academic_events"] = _as_windows(cfg.get("academic_events") or [])
    data["closures"] = _as_windows(cfg.get("holidays") or [])
    if sem:
        data["terms"] = [
            {
                "id": "config-fallback",
                "session": "",
                "name": "Config fallback term",
                "programmes": "",
                "classes_start": None,
                "classes_end": None,
                "next_term_start": None,
                "notes": "Derived from config.yaml because no calendar file was found.",
            }
        ]
    return data


def admin_token(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return os.environ.get("LIBRARY_ADMIN_TOKEN") or str(cfg.get("admin", {}).get("token") or "")


def active_term_for(ts: pd.Timestamp, terms: list[dict] | None = None) -> dict | None:
    day = pd.Timestamp(ts).normalize()
    matched = None
    for term in terms or []:
        start = term.get("classes_start")
        if not start:
            continue
        start_ts = pd.Timestamp(start)
        end_raw = term.get("classes_end")
        end_ts = pd.Timestamp(end_raw) if end_raw else start_ts + pd.Timedelta(days=120)
        if start_ts <= day <= end_ts:
            matched = term
    return matched


def semester_week(ts: pd.Timestamp, calendar: dict | None = None, cfg: dict | None = None) -> int:
    calendar = calendar if calendar is not None else load_calendar()
    term = active_term_for(ts, calendar.get("terms") or [])
    if term and term.get("classes_start"):
        start = pd.Timestamp(term["classes_start"])
        end = pd.Timestamp(term["classes_end"]) if term.get("classes_end") else None
        week = int((pd.Timestamp(ts).normalize() - start).days // 7) + 1
        if end is not None:
            span = max(1, int((end - start).days // 7) + 1)
            return int(max(0, min(week, span + 4)))
        return int(max(0, week))
    cfg = cfg or load_config()
    sem = cfg.get("semester") or {}
    if not sem:
        return 0
    year = pd.Timestamp(ts).year
    month = pd.Timestamp(ts).month
    if month >= sem.get("fall_start_month", 8):
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    elif month <= 6:
        start = pd.Timestamp(year=year, month=sem["spring_start_month"], day=sem["spring_start_day"])
    else:
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    week = int((pd.Timestamp(ts).normalize() - start).days // 7) + 1
    return int(max(0, min(week, sem.get("weeks", 16) + 4)))


def in_semester(ts: pd.Timestamp, calendar: dict | None = None) -> bool:
    calendar = calendar if calendar is not None else load_calendar()
    return active_term_for(ts, calendar.get("terms") or []) is not None


def snapshot() -> dict[str, Any]:
    return deepcopy(load_calendar())
