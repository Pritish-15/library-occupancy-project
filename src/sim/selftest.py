"""Quick seven-day simulation health and performance check."""

from __future__ import annotations

import time

import pandas as pd

from src.config import load_config
from src.sim.engine import SimulationEngine


def run_selftest(days: int = 7) -> dict:
    cfg = load_config()
    start = cfg["simulation"]["start"]
    seed = int(cfg["simulation"]["seed"])
    engines = [
        SimulationEngine(cfg, start=start, seed=seed, recommender_enabled=True),
        SimulationEngine(cfg, start=start, seed=seed, recommender_enabled=False),
    ]
    exam_start = engines[0].clock + pd.Timedelta(days=2)
    closure_start = engines[0].clock + pd.Timedelta(days=5)
    for engine in engines:
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
            "zone_closure", library_id="lib_central", zone_id="z_quiet", hours=12, start=closure_start
        )

    step_times = []
    for _ in range(days * 24):
        for index, engine in enumerate(engines):
            started = time.perf_counter()
            engine.step(1, return_state=False)
            elapsed = (time.perf_counter() - started) * 1000
            if index == 0:
                step_times.append(elapsed)

    started = time.perf_counter()
    engines[0].forecast(3)
    forecast_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    engines[0].state()
    state_ms = (time.perf_counter() - started) * 1000
    result = {
        "days": days,
        "overflow_with_recommender": engines[0].total_overflow,
        "overflow_without_recommender": engines[1].total_overflow,
        "live_mae_overall": engines[0]._mae()["overall"],
        "timings_ms": {
            "step_mean": sum(step_times) / max(1, len(step_times)),
            "step_p95": float(pd.Series(step_times).quantile(0.95)),
            "state": state_ms,
            "forecast_3h": forecast_ms,
        },
    }
    return result


def main() -> None:
    result = run_selftest()
    print(f"7-day overflow (recommender on): {result['overflow_with_recommender']:.1f}")
    print(f"7-day overflow (recommender off): {result['overflow_without_recommender']:.1f}")
    print(f"Live 1h-ahead MAE (overall): {result['live_mae_overall']:.3f}")
    print("Timing table (milliseconds)")
    print("operation       mean/p95")
    print(f"step mean       {result['timings_ms']['step_mean']:.2f}")
    print(f"step p95        {result['timings_ms']['step_p95']:.2f}")
    print(f"state           {result['timings_ms']['state']:.2f}")
    print(f"forecast (3h)   {result['timings_ms']['forecast_3h']:.2f}")


if __name__ == "__main__":
    main()