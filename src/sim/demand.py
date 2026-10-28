"""Generator-matched latent occupancy demand for simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar.store import active_term_for, semester_week
from src.config import load_config
from src.data.generate import HOUR_PROFILE, WEEKDAY_MULT, _in_windows


def latent_utilization(
    ts: pd.Timestamp,
    zone: dict,
    lib_bias: float,
    zone_bias: float,
    calendar_overlay: dict,
    cfg: dict,
    rng: np.random.Generator | None = None,
    noise: bool = True,
) -> float:
    """Return unclipped synthetic utilization demand, capped at a sane maximum."""
    cfg = cfg or load_config()
    timestamp = pd.Timestamp(ts)
    zone_id = str(zone["id"])
    base = float(HOUR_PROFILE[timestamp.hour] * WEEKDAY_MULT[timestamp.weekday()])
    if timestamp.hour >= 20 and ("late" in zone_id or "computing" in zone_id):
        base *= 1.25

    sem_week = semester_week(timestamp, calendar=calendar_overlay, cfg=cfg)
    term = active_term_for(timestamp, calendar_overlay.get("terms") or [])
    if term:
        ramp = 0.75 + 0.25 * min(sem_week / 6, 1.0)
        if term.get("classes_end") and term.get("classes_start"):
            span = max(
                1,
                int((pd.Timestamp(term["classes_end"]) - pd.Timestamp(term["classes_start"])).days // 7) + 1,
            )
            if sem_week >= span - 1:
                ramp *= 0.55
    else:
        ramp = 0.35

    multiplier = float(lib_bias) * float(zone_bias) * ramp
    if _in_windows(timestamp, calendar_overlay.get("exam_windows") or []):
        multiplier *= 1.55
    if _in_windows(timestamp, calendar_overlay.get("academic_events") or []):
        multiplier *= 1.18
    if _in_windows(timestamp, calendar_overlay.get("closures") or []):
        multiplier *= 0.18

    utilization = base * multiplier
    if noise:
        generator = rng if rng is not None else np.random.default_rng()
        utilization += float(generator.normal(0, cfg["generation"]["noise_scale"]))
    return float(np.clip(utilization, 0.0, 1.6))