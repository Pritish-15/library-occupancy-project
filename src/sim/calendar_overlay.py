"""In-memory academic calendar overlays and runtime scenario windows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import uuid

import numpy as np
import pandas as pd

from src.calendar.store import load_calendar


class CalendarOverlay:
    """Keep scenario windows isolated from the persisted academic calendar."""

    def __init__(self, calendar: dict | None = None):
        self.base = deepcopy(calendar if calendar is not None else load_calendar())
        self.scenarios: list[dict[str, Any]] = []
        self._calendar_cache: dict | None = None

    def add(
        self,
        kind: str,
        now: pd.Timestamp,
        *,
        library_id: str | None = None,
        zone_id: str | None = None,
        hours: int | None = None,
        days: int | None = None,
        magnitude: float | None = None,
        start: pd.Timestamp | None = None,
        zones: list[tuple[str, str]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        timestamp = pd.Timestamp(start if start is not None else now)
        params: dict[str, Any] = {}
        if kind in {"exam_surge", "holiday", "event_spike"}:
            duration_days = int(days if days is not None else (5 if kind == "exam_surge" else 2 if kind == "event_spike" else 1))
            if duration_days < 1:
                raise ValueError("days must be at least 1")
            end = timestamp + pd.Timedelta(days=duration_days)
            params["days"] = duration_days
        elif kind == "zone_closure":
            if not library_id or not zone_id:
                raise ValueError("zone_closure requires library_id and zone_id")
            duration_hours = int(hours if hours is not None else 24)
            if duration_hours < 1:
                raise ValueError("hours must be at least 1")
            end = timestamp + pd.Timedelta(hours=duration_hours)
            params.update(library_id=library_id, zone_id=zone_id, hours=duration_hours)
        elif kind == "random_spike":
            generator = rng if rng is not None else np.random.default_rng()
            candidates = zones or []
            if zone_id:
                candidates = [pair for pair in candidates if pair[1] == zone_id]
            if library_id:
                candidates = [pair for pair in candidates if pair[0] == library_id]
            if candidates and (not library_id or not zone_id):
                library_id, zone_id = candidates[int(generator.integers(0, len(candidates)))]
            if not library_id or not zone_id:
                raise ValueError("random_spike requires a valid zone or configured zones")
            spike_magnitude = float(magnitude if magnitude is not None else generator.uniform(1.3, 1.8))
            duration_hours = int(hours if hours is not None else generator.integers(3, 7))
            if not 1.3 <= spike_magnitude <= 1.8:
                raise ValueError("magnitude must be between 1.3 and 1.8")
            if not 3 <= duration_hours <= 6:
                raise ValueError("random_spike hours must be between 3 and 6")
            end = timestamp + pd.Timedelta(hours=duration_hours)
            params.update(library_id=library_id, zone_id=zone_id, magnitude=spike_magnitude, hours=duration_hours)
        else:
            raise ValueError(f"Unknown scenario kind: {kind}")

        scenario = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "start": timestamp,
            "end": end,
            "params": params,
        }
        self.scenarios.append(scenario)
        self._calendar_cache = None
        return scenario

    def cancel(self, scenario_id: str) -> bool:
        before = len(self.scenarios)
        self.scenarios = [item for item in self.scenarios if item["id"] != scenario_id]
        if len(self.scenarios) != before:
            self._calendar_cache = None
        return len(self.scenarios) != before

    def active(self, ts: pd.Timestamp, kind: str | None = None) -> list[dict[str, Any]]:
        timestamp = pd.Timestamp(ts)
        return [
            item
            for item in self.scenarios
            if item["start"] <= timestamp < item["end"] and (kind is None or item["kind"] == kind)
        ]

    def calendar(self) -> dict:
        if self._calendar_cache is not None:
            return self._calendar_cache
        overlay = deepcopy(self.base)
        for item in self.scenarios:
            start = item["start"].strftime("%Y-%m-%d")
            end = item["end"].strftime("%Y-%m-%d")
            window = {"start": start, "end": end, "name": f"Simulation {item['kind']}"}
            if item["kind"] == "exam_surge":
                overlay.setdefault("exam_windows", []).append(window)
            elif item["kind"] == "holiday":
                overlay.setdefault("closures", []).append(window)
            elif item["kind"] == "event_spike":
                overlay.setdefault("academic_events", []).append(window)
        self._calendar_cache = overlay
        return overlay

    def list(self, now: pd.Timestamp | None = None) -> list[dict[str, Any]]:
        timestamp = pd.Timestamp(now) if now is not None else None
        return [
            {
                "id": item["id"],
                "kind": item["kind"],
                "start": item["start"].isoformat(sep=" "),
                "end": item["end"].isoformat(sep=" "),
                "params": dict(item["params"]),
                "active": timestamp is None or item["start"] <= timestamp < item["end"],
            }
            for item in self.scenarios
        ]