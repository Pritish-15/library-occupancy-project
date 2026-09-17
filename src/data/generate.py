"""Synthetic occupancy generator matching the ideal dataset schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config, data_path


HOUR_PROFILE = np.array(
    [
        0.08, 0.05, 0.04, 0.03, 0.03, 0.05, 0.12, 0.28,
        0.48, 0.62, 0.72, 0.78, 0.82, 0.80, 0.76, 0.74,
        0.78, 0.84, 0.80, 0.68, 0.50, 0.36, 0.22, 0.14,
    ]
)

WEEKDAY_MULT = np.array([0.92, 1.05, 1.08, 1.10, 1.02, 0.55, 0.42])  # Mon..Sun


def _in_windows(ts: pd.Timestamp, windows: list[dict]) -> bool:
    d = ts.normalize()
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        if start <= d <= end:
            return True
    return False


def _semester_week(ts: pd.Timestamp, cfg: dict) -> int:
    year = ts.year
    month = ts.month
    sem = cfg["semester"]
    if month >= sem["fall_start_month"]:
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    elif month <= 6:
        start = pd.Timestamp(year=year, month=sem["spring_start_month"], day=sem["spring_start_day"])
    else:
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    week = int((ts.normalize() - start).days // 7) + 1
    return int(np.clip(week, 0, sem["weeks"] + 4))


def generate_occupancy(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg["project"]["seed"])
    gen = cfg["generation"]
    index = pd.date_range(gen["start"], gen["end"], freq=gen["freq"])
    holidays = {pd.Timestamp(h) for h in cfg["holidays"]}

    rows: list[dict] = []
    for lib in cfg["libraries"]:
        lib_bias = rng.normal(1.0, 0.06)
        for zone in lib["zones"]:
            zone_bias = rng.normal(1.0, 0.08)
            capacity = int(zone["capacity"])
            late_boost = 1.25 if "late" in zone["id"] or "computing" in zone["id"] else 1.0
            for ts in index:
                hour = ts.hour
                base = HOUR_PROFILE[hour] * WEEKDAY_MULT[ts.weekday()]
                if hour >= 20:
                    base *= late_boost
                sem_week = _semester_week(ts, cfg)
                if 1 <= sem_week <= cfg["semester"]["weeks"]:
                    ramp = 0.75 + 0.25 * min(sem_week / 6, 1.0)
                    if sem_week >= cfg["semester"]["weeks"] - 1:
                        ramp *= 0.55
                else:
                    ramp = 0.35
                exam = _in_windows(ts, cfg["exam_windows"])
                event = _in_windows(ts, cfg["academic_events"])
                holiday = ts.normalize() in holidays
                mult = lib_bias * zone_bias * ramp
                if exam:
                    mult *= 1.55
                if event:
                    mult *= 1.18
                if holiday:
                    mult *= 0.18
                noise = rng.normal(0, gen["noise_scale"])
                util = float(np.clip(base * mult + noise, 0.0, 1.02))
                occupied = int(np.clip(round(util * capacity), 0, capacity))
                computer = int(np.clip(round(occupied * (0.25 + 0.2 * ("computing" in zone["id"] or "media" in zone["id"])) + rng.normal(0, 2)), 0, capacity))
                rooms = int(np.clip(round(occupied * (0.12 + 0.25 * ("rooms" in zone["id"] or "group" in zone["id"] or "collab" in zone["id"])) + rng.normal(0, 1)), 0, max(8, capacity // 4)))
                books = int(np.clip(round(occupied * 0.35 + rng.normal(0, 4)), 0, capacity * 2))
                charging = int(np.clip(round(occupied * 0.22 + rng.normal(0, 2)), 0, capacity))
                rows.append(
                    {
                        "timestamp": ts,
                        "library_id": lib["id"],
                        "library_name": lib["name"],
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "total_capacity": capacity,
                        "occupied_seats": occupied,
                        "available_seats": capacity - occupied,
                        "computer_usage": computer,
                        "study_room_usage": rooms,
                        "book_demand": books,
                        "charging_station_usage": charging,
                    }
                )

    df = pd.DataFrame(rows)

    n = len(df)
    n_out = max(1, int(n * gen["outlier_rate"]))
    outlier_idx = rng.choice(n, size=n_out, replace=False)
    df.loc[outlier_idx, "occupied_seats"] = np.clip(
        df.loc[outlier_idx, "occupied_seats"] + rng.integers(15, 45, size=n_out),
        0,
        df.loc[outlier_idx, "total_capacity"],
    )
    df.loc[outlier_idx, "available_seats"] = (
        df.loc[outlier_idx, "total_capacity"] - df.loc[outlier_idx, "occupied_seats"]
    )

    n_gaps = max(1, int(n * gen["missing_gap_rate"]))
    gap_starts = rng.choice(n - 6, size=n_gaps, replace=False)
    miss_cols = ["occupied_seats", "available_seats", "computer_usage", "study_room_usage"]
    for start in gap_starts:
        length = int(rng.integers(1, 5))
        df.loc[start : start + length, miss_cols] = np.nan

    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)


def write_raw(df: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    df = df if df is not None else generate_occupancy()
    path = data_path("raw_data")
    df.to_csv(path, index=False)
    return df


def main() -> None:
    df = write_raw()
    print(f"Wrote {len(df):,} rows to {data_path('raw_data')}")


if __name__ == "__main__":
    main()
