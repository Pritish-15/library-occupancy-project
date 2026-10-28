"""Headless with/without recommender comparison runs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config
from src.sim.engine import SimulationEngine


def _summary(engine: SimulationEngine) -> dict[str, Any]:
    return {
        "total_overflow": float(engine.total_overflow),
        "total_redirected": float(engine.total_redirected),
        "hours_any_critical": int(engine.hours_any_critical),
        "peak_utilization_by_zone": dict(engine.peak_utilization),
    }


def run_comparison(
    days: int = 14,
    seed: int | None = None,
    preset: str | None = None,
    compliance: float = 0.4,
) -> dict[str, Any]:
    """Run identical scenario timing with recommender enabled and disabled."""
    if days < 1:
        raise ValueError("days must be at least 1")
    if not 0 <= compliance <= 1:
        raise ValueError("compliance must be between 0 and 1")
    cfg = load_config()
    start = cfg.get("simulation_presets", {}).get(preset) if preset else cfg.get("simulation", {}).get("start")
    if preset and start is None:
        raise ValueError(f"Unknown simulation preset: {preset}")
    run_seed = int(seed if seed is not None else cfg.get("simulation", {}).get("seed", 7))
    engines = [
        SimulationEngine(cfg, start=start, seed=run_seed, recommender_enabled=True, track_accuracy=False),
        SimulationEngine(cfg, start=start, seed=run_seed, recommender_enabled=False, track_accuracy=False),
    ]
    exam_start = engines[0].clock + pd.Timedelta(days=3)
    closure_start = engines[0].clock + pd.Timedelta(days=8)
    for engine in engines:
        engine.compliance = float(compliance)
        engine.add_scenario("exam_surge", days=5, start=exam_start)
        engine.add_scenario(
            "random_spike",
            library_id="lib_central",
            zone_id="z_quiet",
            magnitude=1.8,
            hours=6,
            start=exam_start,
        )
        engine.add_scenario(
            "zone_closure", library_id="lib_central", zone_id="z_quiet", hours=24, start=closure_start
        )
    for _ in range(days * 24):
        for engine in engines:
            engine.step(1, return_state=False)
    with_recommender = _summary(engines[0])
    without_recommender = _summary(engines[1])
    baseline = without_recommender["total_overflow"]
    reduction = (baseline - with_recommender["total_overflow"]) / baseline * 100 if baseline else 0.0
    return {
        "days": days,
        "seed": run_seed,
        "preset": preset,
        "compliance": compliance,
        "with_recommender": with_recommender,
        "without_recommender": without_recommender,
        "overflow_reduction_percent": float(reduction),
    }