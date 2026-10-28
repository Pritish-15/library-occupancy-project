"""Thread-safe, seeded hourly occupancy simulation engine."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.calendar.store import load_calendar
from src.config import load_config, project_path
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features
from src.features.temporal import add_temporal_features
from src.features.transformers import occupancy_level
from src.pipeline.predict import artifact_dir
from src.pipeline.recommend import recommend_alternatives
from src.sim.calendar_overlay import CalendarOverlay
from src.sim.demand import latent_utilization


RAW_COLUMNS = [
    "timestamp", "library_id", "library_name", "zone_id", "zone_name", "total_capacity",
    "occupied_seats", "available_seats", "computer_usage", "study_room_usage", "book_demand",
    "charging_station_usage",
]
RESOURCE_COLUMNS = ("computer_usage", "study_room_usage", "book_demand", "charging_station_usage")


class SimulationEngine:
    """Maintain simulated occupancy history and model forecasts for all zones."""

    def __init__(
        self,
        cfg: dict | None = None,
        start: str | pd.Timestamp | None = None,
        seed: int | None = None,
        recommender_enabled: bool = True,
        track_accuracy: bool = True,
    ):
        self.cfg = cfg or load_config()
        self._lock = threading.RLock()
        self.track_accuracy = bool(track_accuracy)
        self._initialize(start=start, seed=seed, recommender_enabled=recommender_enabled)

    def _initialize(
        self,
        start: str | pd.Timestamp | None,
        seed: int | None,
        recommender_enabled: bool,
    ) -> None:
        settings = self.cfg.get("simulation", {})
        self.seed = int(seed if seed is not None else settings.get("seed", self.cfg["project"]["seed"]))
        self.rng = np.random.default_rng(self.seed)
        start_value = start if start is not None else settings.get("start", "2026-11-30 06:00")
        self.clock = pd.Timestamp(start_value)
        self.buffer_hours = int(settings.get("buffer_hours", 240))
        self.warmup_hours = int(settings.get("warmup_hours", 192))
        self.near_capacity = float(settings.get("near_capacity", self.cfg["recommendation"]["near_capacity"]))
        self.compliance = float(settings.get("recommender_compliance", 0.4))
        self.recommender_enabled = bool(recommender_enabled)
        self.running = True
        self.speed = 1
        self.calendar_overlay = CalendarOverlay(load_calendar())
        self.locations: dict[tuple[str, str], dict[str, Any]] = {}
        libraries = sorted({str(lib["id"]) for lib in self.cfg["libraries"]})
        zones = sorted({str(zone["id"]) for lib in self.cfg["libraries"] for zone in lib["zones"]})
        self.library_codes = {value: index for index, value in enumerate(libraries)}
        self.zone_codes = {value: index for index, value in enumerate(zones)}
        self.lib_bias: dict[str, float] = {}
        self.zone_bias: dict[tuple[str, str], float] = {}
        for lib in self.cfg["libraries"]:
            library_id = str(lib["id"])
            self.lib_bias[library_id] = float(self.rng.normal(1.0, 0.06))
            for zone in lib["zones"]:
                key = (library_id, str(zone["id"]))
                self.zone_bias[key] = float(self.rng.normal(1.0, 0.08))
                self.locations[key] = {
                    "library_id": library_id,
                    "library_name": str(lib["name"]),
                    "zone_id": str(zone["id"]),
                    "zone_name": str(zone["name"]),
                    "total_capacity": int(zone["capacity"]),
                    "zone": zone,
                }

        models = artifact_dir(self.cfg)
        meta_path = models / "meta.joblib"
        model_path = models / "regressor.joblib"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError("Occupancy model artifacts missing. Run python -m src.pipeline.train")
        self.meta = joblib.load(meta_path)
        self.feature_cols = list(self.meta["feature_cols"])
        self.model = joblib.load(model_path)
        self._resource_models: dict[str, Any] | None = None
        self._resource_meta: dict[str, Any] | None = None

        self.history: dict[tuple[str, str], pd.DataFrame] = {key: pd.DataFrame() for key in self.locations}
        self._last_overflow: dict[tuple[str, str], float] = defaultdict(float)
        self._closed: set[tuple[str, str]] = set()
        self._pending_redirects: dict[tuple[str, str], tuple[tuple[str, str], float]] = {}
        self.total_overflow = 0.0
        self.total_redirected = 0.0
        self.hours_critical: dict[str, int] = defaultdict(int)
        self.peak_utilization: dict[str, float] = {f"{key[0]}/{key[1]}": 0.0 for key in self.locations}
        self._event_log: deque[dict[str, Any]] = deque(maxlen=50)
        self._last_levels: dict[tuple[str, str], str] = {}
        self._errors: dict[tuple[str, str], deque[float]] = {key: deque(maxlen=24) for key in self.locations}
        self._all_errors: dict[tuple[str, str], list[float]] = {key: [] for key in self.locations}
        self._pending_accuracy: dict[tuple[str, str], float] = {}
        self._forecast_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._recommendations: list[dict[str, Any]] = []
        self.hours_any_critical = 0

        warmup_rows: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in self.locations}
        first = self.clock - pd.Timedelta(hours=self.warmup_hours - 1)
        for ts in pd.date_range(first, self.clock, freq="h"):
            for row in self._generate_hour(ts, warmup=True):
                key = (row["library_id"], row["zone_id"])
                warmup_rows[key].append(row)
                self._last_levels[key] = self._level(row["occupied_seats"], row["total_capacity"])
        self.history = {key: pd.DataFrame(rows) for key, rows in warmup_rows.items()}
        initial = self.forecast(1) if self.track_accuracy else []
        self._pending_accuracy = {
            (row["library_id"], row["zone_id"]): row["predicted_occupied"] for row in initial
        }
        self._plan_recommendations()

    def _level(self, occupied: float, capacity: int) -> str:
        return str(occupancy_level(pd.Series([occupied / capacity]), self.cfg["utilization_thresholds"]).iloc[0])

    def _expected_resources(self, occupied: float, zone: dict) -> dict[str, int]:
        capacity = int(zone["capacity"])
        zone_id = str(zone["id"])
        computer = round(occupied * (0.25 + 0.2 * ("computing" in zone_id or "media" in zone_id)))
        rooms = round(occupied * (0.12 + 0.25 * ("rooms" in zone_id or "group" in zone_id or "collab" in zone_id)))
        books = round(occupied * 0.35)
        charging = round(occupied * 0.22)
        return {
            "computer_usage": int(np.clip(computer, 0, capacity)),
            "study_room_usage": int(np.clip(rooms, 0, max(8, capacity // 4))),
            "book_demand": int(np.clip(books, 0, capacity * 2)),
            "charging_station_usage": int(np.clip(charging, 0, capacity)),
        }

    def _resource_row(self, occupied: int, zone: dict) -> dict[str, int]:
        capacity = int(zone["capacity"])
        zone_id = str(zone["id"])
        computer = round(occupied * (0.25 + 0.2 * ("computing" in zone_id or "media" in zone_id)) + self.rng.normal(0, 2))
        rooms = round(occupied * (0.12 + 0.25 * ("rooms" in zone_id or "group" in zone_id or "collab" in zone_id)) + self.rng.normal(0, 1))
        books = round(occupied * 0.35 + self.rng.normal(0, 4))
        charging = round(occupied * 0.22 + self.rng.normal(0, 2))
        return {
            "computer_usage": int(np.clip(computer, 0, capacity)),
            "study_room_usage": int(np.clip(rooms, 0, max(8, capacity // 4))),
            "book_demand": int(np.clip(books, 0, capacity * 2)),
            "charging_station_usage": int(np.clip(charging, 0, capacity)),
        }

    def _base_demands(self, ts: pd.Timestamp) -> dict[tuple[str, str], float]:
        calendar = self.calendar_overlay.calendar()
        demands = {}
        for key, location in self.locations.items():
            utilization = latent_utilization(
                ts,
                location["zone"],
                self.lib_bias[key[0]],
                self.zone_bias[key],
                calendar,
                self.cfg,
                self.rng,
            )
            demands[key] = utilization * location["total_capacity"]
        for scenario in self.calendar_overlay.active(ts, "random_spike"):
            key = (scenario["params"]["library_id"], scenario["params"]["zone_id"])
            if key in demands:
                demands[key] *= float(scenario["params"]["magnitude"])
        return demands

    def _allocate(self, demands: dict[tuple[str, str], float], keys: list[tuple[str, str]], amount: float) -> float:
        remaining = max(0.0, amount)
        candidates = list(keys)
        while remaining > 1e-9 and candidates:
            free = {
                key: max(0.0, self.locations[key]["total_capacity"] - demands[key])
                for key in candidates
            }
            total_free = sum(free.values())
            if total_free <= 1e-9:
                break
            allocated = 0.0
            for key in candidates:
                share = min(free[key], remaining * free[key] / total_free)
                demands[key] += share
                allocated += share
            remaining -= allocated
            candidates = [key for key in candidates if self.locations[key]["total_capacity"] - demands[key] > 1e-9]
            if allocated <= 1e-9:
                break
        return remaining

    def _apply_closures(self, ts: pd.Timestamp, demands: dict[tuple[str, str], float]) -> float:
        closed = set()
        stranded = 0.0
        for scenario in self.calendar_overlay.active(ts, "zone_closure"):
            key = (scenario["params"]["library_id"], scenario["params"]["zone_id"])
            if key not in demands:
                continue
            closed.add(key)
            pool = demands[key]
            demands[key] = 0.0
            same_library = [candidate for candidate in demands if candidate[0] == key[0] and candidate not in closed]
            other_libraries = [candidate for candidate in demands if candidate[0] != key[0] and candidate not in closed]
            same_library_left = self._allocate(demands, same_library, pool * 0.7)
            same_library_allocated = pool * 0.7 - same_library_left
            stranded += self._allocate(demands, other_libraries, pool - same_library_allocated)
        self._closed = closed
        return stranded

    def _apply_pending_redirects(self, demands: dict[tuple[str, str], float], closed: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
        redirected: dict[tuple[str, str], float] = defaultdict(float)
        if not self.recommender_enabled:
            return redirected
        for source, (target, compliance) in self._pending_redirects.items():
            if source not in demands or target not in demands or source in closed or target in closed:
                continue
            source_cap = self.locations[source]["total_capacity"]
            target_cap = self.locations[target]["total_capacity"]
            overflow = max(0.0, demands[source] - source_cap)
            headroom = max(0.0, target_cap - demands[target])
            amount = min(overflow * compliance, headroom)
            if amount > 0:
                demands[source] -= amount
                demands[target] += amount
                redirected[source] += amount
                self._event("redirect", f"Recommender redirected {amount:.1f} students from {source[1]} to {target[1]}.")
        return redirected

    def _generate_hour(self, ts: pd.Timestamp, warmup: bool = False) -> list[dict[str, Any]]:
        demands = self._base_demands(ts)
        stranded = self._apply_closures(ts, demands) if not warmup else 0.0
        closed = set(self._closed) if not warmup else set()
        redirected = self._apply_pending_redirects(demands, closed) if not warmup else {}
        rows = []
        overflow_by_zone = {}
        for key, location in self.locations.items():
            capacity = location["total_capacity"]
            demand = 0.0 if key in closed else demands[key]
            overflow = max(0.0, demand - capacity)
            occupied = 0 if key in closed else int(min(round(demand), capacity))
            row = {
                "timestamp": ts,
                "library_id": key[0],
                "library_name": location["library_name"],
                "zone_id": key[1],
                "zone_name": location["zone_name"],
                "total_capacity": capacity,
                "occupied_seats": occupied,
                "available_seats": capacity - occupied,
                **self._resource_row(occupied, location["zone"]),
            }
            rows.append(row)
            overflow_by_zone[key] = overflow
            if not warmup:
                self._last_overflow[key] = overflow
                self._errors[key]
        if not warmup:
            self._current_stranded = stranded
            self._current_redirected = redirected
            self._current_overflow = overflow_by_zone
        return rows

    def _trim_history(self) -> None:
        for key, frame in self.history.items():
            if len(frame) > self.buffer_hours:
                self.history[key] = frame.tail(self.buffer_hours).reset_index(drop=True)

    def features(self, df_buffer: pd.DataFrame) -> pd.DataFrame:
        """Build model features from only the supplied rolling history."""
        out = add_temporal_features(df_buffer)
        out = add_calendar_features(out, cfg=self.cfg, calendar=self.calendar_overlay.calendar())
        out = add_lag_features(out, lags=self.cfg["lags"])
        out["library_code"] = out["library_id"].map(self.library_codes).astype(int)
        out["zone_code"] = out["zone_id"].map(self.zone_codes).astype(int)
        out["utilization"] = out["occupied_seats"] / out["total_capacity"]
        out["occupancy_level"] = occupancy_level(out["utilization"], self.cfg["utilization_thresholds"])
        return out

    def _strict_model_input(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = self.features(pd.DataFrame(rows))
        missing = [column for column in self.feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Simulation model features missing required columns: {missing}")
        return frame.loc[:, self.feature_cols]

    def _future_features(self, key: tuple[str, str], ts: pd.Timestamp, history: pd.DataFrame) -> dict[str, Any]:
        location = self.locations[key]
        last_occupied = float(history["occupied_seats"].iloc[-1])
        candidate = {
            "timestamp": ts,
            "library_id": key[0],
            "library_name": location["library_name"],
            "zone_id": key[1],
            "zone_name": location["zone_name"],
            "total_capacity": location["total_capacity"],
            "occupied_seats": last_occupied,
            "available_seats": int(location["total_capacity"] - last_occupied),
            **self._expected_resources(last_occupied, location["zone"]),
        }
        return pd.concat([history, pd.DataFrame([candidate])], ignore_index=True).to_dict("records")[-1]

    def _forecast_uncached(self, horizon: int) -> list[dict[str, Any]]:
        working = {key: frame.copy() for key, frame in self.history.items()}
        output = []
        for step in range(1, horizon + 1):
            ts = self.clock + pd.Timedelta(hours=step)
            candidates = {}
            for key, frame in working.items():
                location = self.locations[key]
                last_occupied = float(frame["occupied_seats"].iloc[-1])
                candidate = {
                    "timestamp": ts,
                    "library_id": key[0],
                    "library_name": location["library_name"],
                    "zone_id": key[1],
                    "zone_name": location["zone_name"],
                    "total_capacity": location["total_capacity"],
                    "occupied_seats": last_occupied,
                    "available_seats": int(location["total_capacity"] - last_occupied),
                    **self._expected_resources(last_occupied, location["zone"]),
                }
                candidates[key] = candidate
            candidate_frame = add_temporal_features(pd.DataFrame(list(candidates.values())))
            candidate_frame = add_calendar_features(
                candidate_frame, cfg=self.cfg, calendar=self.calendar_overlay.calendar()
            )
            rows = candidate_frame.copy()
            for index, key in enumerate(candidates):
                history = working[key]["occupied_seats"]
                for lag in self.cfg["lags"]:
                    rows.loc[index, f"lag_{lag}"] = (
                        float(history.iloc[-lag]) if len(history) >= lag else float(history.iloc[-1])
                    )
            rows["library_code"] = rows["library_id"].map(self.library_codes).astype(int)
            rows["zone_code"] = rows["zone_id"].map(self.zone_codes).astype(int)
            rows["utilization"] = rows["occupied_seats"] / rows["total_capacity"]
            rows["occupancy_level"] = occupancy_level(rows["utilization"], self.cfg["utilization_thresholds"])
            missing = [column for column in self.feature_cols if column not in rows.columns]
            if missing:
                raise ValueError(f"Simulation model features missing required columns: {missing}")
            model_input = rows.loc[:, self.feature_cols]
            predicted = np.asarray(self.model.predict(model_input), dtype=float).reshape(-1)
            next_candidates = {}
            for index, (key, candidate) in enumerate(candidates.items()):
                capacity = self.locations[key]["total_capacity"]
                occupied = float(np.clip(predicted[index], 0, capacity))
                for scenario in self.calendar_overlay.active(ts, "random_spike"):
                    if (scenario["params"]["library_id"], scenario["params"]["zone_id"]) == key:
                        occupied = float(np.clip(occupied * scenario["params"]["magnitude"], 0, capacity))
                for scenario in self.calendar_overlay.active(ts, "zone_closure"):
                    if (scenario["params"]["library_id"], scenario["params"]["zone_id"]) == key:
                        occupied = 0.0
                location = self.locations[key]
                result = {
                    "timestamp": ts.isoformat(sep=" "),
                    "horizon_h": step,
                    "hour": int(rows.iloc[index]["hour"]),
                    "day": int(rows.iloc[index]["day"]),
                    "weekday": int(rows.iloc[index]["weekday"]),
                    "is_weekend": int(rows.iloc[index]["is_weekend"]),
                    "month": int(rows.iloc[index]["month"]),
                    "semester_week": int(rows.iloc[index]["semester_week"]),
                    "is_exam_period": int(rows.iloc[index]["is_exam_period"]),
                    "is_holiday": int(rows.iloc[index]["is_holiday"]),
                    "is_academic_event": int(rows.iloc[index]["is_academic_event"]),
                    "is_in_semester": int(rows.iloc[index]["is_in_semester"]),
                    "library_id": key[0],
                    "zone_id": key[1],
                    "library_name": location["library_name"],
                    "zone_name": location["zone_name"],
                    "predicted_occupied": occupied,
                    "predicted_utilization": occupied / capacity,
                    "predicted_level": self._level(occupied, capacity),
                    "total_capacity": capacity,
                }
                output.append(result)
                next_candidates[key] = {
                    **candidate,
                    "occupied_seats": occupied,
                    "available_seats": int(capacity - occupied),
                    **self._expected_resources(occupied, location["zone"]),
                }
            for key, candidate in next_candidates.items():
                working[key] = pd.concat([working[key], pd.DataFrame([candidate])], ignore_index=True).tail(self.buffer_hours)
        return output

    def forecast(self, horizon: int = 1) -> list[dict[str, Any]]:
        if horizon not in {1, 3, 6, 12, 24}:
            raise ValueError("horizon must be 1, 3, 6, 12, or 24")
        with self._lock:
            cache_key = (str(self.clock), horizon)
            if cache_key not in self._forecast_cache:
                self._forecast_cache[cache_key] = self._forecast_uncached(horizon)
            return [dict(row) for row in self._forecast_cache[cache_key]]

    def forecast_resources(self, horizon: int = 3) -> list[dict[str, Any]]:
        """Recursively forecast resource counts using dedicated target models."""
        if horizon not in {1, 3, 6, 12, 24}:
            raise ValueError("horizon must be 1, 3, 6, 12, or 24")
        with self._lock:
            models_dir = artifact_dir(self.cfg)
            meta_path = models_dir / "resource_meta.joblib"
            if not meta_path.exists():
                raise FileNotFoundError("Resource models missing. Run python -m src.pipeline.train_resources")
            if self._resource_models is None:
                self._resource_meta = joblib.load(meta_path)
                self._resource_models = {
                    resource: joblib.load(models_dir / f"resource_{resource}.joblib")
                    for resource in RESOURCE_COLUMNS
                }

            occupancy = {
                (row["library_id"], row["zone_id"], row["horizon_h"]): row
                for row in self.forecast(horizon)
            }
            working = {key: frame.copy() for key, frame in self.history.items()}
            results: dict[tuple[str, str, int], dict[str, Any]] = {}
            for step in range(1, horizon + 1):
                candidates = []
                candidate_keys = []
                timestamp = self.clock + pd.Timedelta(hours=step)
                closed_zones = {
                    key: any(
                        (item["params"]["library_id"], item["params"]["zone_id"]) == key
                        for item in self.calendar_overlay.active(timestamp, "zone_closure")
                    )
                    for key in working
                }
                for key, history in working.items():
                    occ = occupancy[(key[0], key[1], step)]
                    resource_row: dict[str, Any] = {
                        "library_id": key[0],
                        "zone_id": key[1],
                        "library_code": self.library_codes[key[0]],
                        "zone_code": self.zone_codes[key[1]],
                        "total_capacity": self.locations[key]["total_capacity"],
                        **{name: occ[name] for name in (
                            "hour", "day", "weekday", "is_weekend", "month", "semester_week",
                            "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
                        )},
                        "closed": closed_zones[key],
                    }
                    for lag in (1, 2, 3, 6, 24, 168):
                        resource_row[f"occupied_lag_{lag}"] = float(history["occupied_seats"].iloc[-lag])
                    for resource in RESOURCE_COLUMNS:
                        for lag in (1, 2, 3, 6, 24, 168):
                            resource_row[f"{resource}_lag_{lag}"] = float(history[resource].iloc[-lag])
                    candidates.append(resource_row)
                    candidate_keys.append(key)
                predictions: dict[str, np.ndarray] = {}
                for resource in RESOURCE_COLUMNS:
                    feature_cols = self._resource_meta["resources"][resource]["feature_cols"]
                    frame = pd.DataFrame(candidates)
                    missing = [column for column in feature_cols if column not in frame.columns]
                    if missing:
                        raise ValueError(f"Resource model features missing for {resource}: {missing}")
                    predictions[resource] = np.maximum(
                        0.0,
                        np.asarray(self._resource_models[resource].predict(frame.loc[:, feature_cols]), dtype=float),
                    )
                for index, key in enumerate(candidate_keys):
                    occ = occupancy[(key[0], key[1], step)]
                    result = {
                        "timestamp": occ["timestamp"],
                        "horizon_h": step,
                        "library_id": key[0],
                        "zone_id": key[1],
                        "library_name": self.locations[key]["library_name"],
                        "zone_name": self.locations[key]["zone_name"],
                        "predicted_occupied": occ["predicted_occupied"],
                        "predicted_utilization": occ["predicted_utilization"],
                        "total_capacity": occ["total_capacity"],
                        "closed": closed_zones[key],
                    }
                    for resource in RESOURCE_COLUMNS:
                        value = float(predictions[resource][index])
                        resource_capacity = float(
                            self._resource_meta["resource_capacity"][key[0]][key[1]][resource]
                        )
                        utilization = value / resource_capacity if resource_capacity > 0 else 0.0
                        result[resource] = value
                        result[f"{resource}_capacity"] = resource_capacity
                        result[f"{resource}_utilization"] = utilization
                        result[f"{resource}_level"] = self._level(utilization, 1)
                    results[(key[0], key[1], step)] = result
                    next_row = {
                        **working[key].iloc[-1].to_dict(),
                        "timestamp": pd.Timestamp(occ["timestamp"]),
                        "occupied_seats": occ["predicted_occupied"],
                        **{resource: predictions[resource][index] for resource in RESOURCE_COLUMNS},
                    }
                    working[key] = pd.concat([working[key], pd.DataFrame([next_row])], ignore_index=True).tail(self.buffer_hours)
            return [results[(key[0], key[1], step)] for step in range(1, horizon + 1) for key in self.locations]

    def _event(self, event_type: str, message: str) -> None:
        self._event_log.append({"timestamp": self.clock.isoformat(sep=" "), "type": event_type, "message": message})

    def _plan_recommendations(self) -> None:
        self._pending_redirects = {}
        self._recommendations = []
        if not self.recommender_enabled:
            return
        if not self.track_accuracy and not any(
            float(self.history[key]["occupied_seats"].iloc[-1]) / location["total_capacity"] >= 0.7
            for key, location in self.locations.items()
        ):
            return
        next_time = self.clock + pd.Timedelta(hours=1)
        future_closed = {
            (item["params"]["library_id"], item["params"]["zone_id"])
            for item in self.calendar_overlay.active(next_time, "zone_closure")
        }
        forecast_rows = [
            row for row in self.forecast(1)
            if (row["library_id"], row["zone_id"]) not in self._closed | future_closed
        ]
        for row in forecast_rows:
            key = (row["library_id"], row["zone_id"])
            row["occupied_seats"] = int(self.history[key]["occupied_seats"].iloc[-1])
        forecast_frame = pd.DataFrame(forecast_rows)
        for row in forecast_rows:
            key = (row["library_id"], row["zone_id"])
            if row["predicted_utilization"] < self.near_capacity:
                continue
            recs = recommend_alternatives(forecast_frame, key[0], key[1], self.cfg)
            if not recs or not recs[0].get("needed"):
                continue
            recommendation = {"library_id": key[0], "zone_id": key[1], **recs[0]}
            self._recommendations.append(recommendation)
            alternatives = recs[0].get("alternatives", [])
            if alternatives:
                target = (alternatives[0]["library_id"], alternatives[0]["zone_id"])
                self._pending_redirects[key] = (target, self.compliance)

    def step(self, hours: int = 1, *, return_state: bool = True) -> dict[str, Any] | None:
        if not 1 <= int(hours) <= 48:
            raise ValueError("hours must be between 1 and 48")
        with self._lock:
            for _ in range(int(hours)):
                ts = self.clock + pd.Timedelta(hours=1)
                predicted_before = self._pending_accuracy
                rows = self._generate_hour(ts)
                self.clock = ts
                hour_has_critical = False
                for row in rows:
                    key = (row["library_id"], row["zone_id"])
                    peak_key = f"{key[0]}/{key[1]}"
                    self.peak_utilization[peak_key] = max(
                        self.peak_utilization[peak_key], row["occupied_seats"] / row["total_capacity"]
                    )
                    if key in predicted_before:
                        error = abs(float(row["occupied_seats"]) - float(predicted_before[key]))
                        self._errors[key].append(error)
                        self._all_errors[key].append(error)
                    self.history[key] = pd.concat([self.history[key], pd.DataFrame([row])], ignore_index=True)
                    level = self._level(row["occupied_seats"], row["total_capacity"])
                    if level == "Critical" and self._last_levels.get(key) != "Critical":
                        self._event("critical", f"Zone {row['zone_name']} crossed Critical.")
                    self._last_levels[key] = level
                    if level == "Critical":
                        self.hours_critical[row["zone_id"]] += 1
                        hour_has_critical = True
                self.hours_any_critical += int(hour_has_critical)
                self.total_overflow = max(
                    0.0,
                    self.total_overflow + sum(self._current_overflow.values()) + self._current_stranded,
                )
                self.total_redirected += sum(self._current_redirected.values())
                for scenario in self.calendar_overlay.scenarios:
                    if scenario["start"] == self.clock:
                        self._event("scenario", f"Scenario started: {scenario['kind']}.")
                self._trim_history()
                self._forecast_cache.clear()
                self._plan_recommendations()
                self._pending_accuracy = {
                    (row["library_id"], row["zone_id"]): row["predicted_occupied"]
                    for row in self.forecast(1)
                } if self.track_accuracy else {}
            return self.state() if return_state else None

    def add_scenario(self, kind: str, **params) -> dict[str, Any]:
        with self._lock:
            scenario = self.calendar_overlay.add(
                kind,
                self.clock,
                zones=list(self.locations),
                rng=self.rng,
                **params,
            )
            self._forecast_cache.clear()
            self._event("scenario", f"Scenario started: {kind}.")
            self._plan_recommendations()
            return self.calendar_overlay.list(self.clock)[-1]

    def cancel_scenario(self, scenario_id: str) -> bool:
        with self._lock:
            canceled = self.calendar_overlay.cancel(scenario_id)
            if canceled:
                self._forecast_cache.clear()
                self._event("scenario", "Scenario canceled.")
            return canceled

    def set_recommender(self, enabled: bool, compliance: float | None = None) -> None:
        if compliance is not None and not 0 <= compliance <= 1:
            raise ValueError("compliance must be between 0 and 1")
        with self._lock:
            self.recommender_enabled = bool(enabled)
            if compliance is not None:
                self.compliance = float(compliance)
            self._pending_redirects.clear()
            self._plan_recommendations()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            records = []
            for key, location in self.locations.items():
                row = self.history[key].iloc[-1]
                occupied = int(row["occupied_seats"])
                utilization = occupied / location["total_capacity"]
                active_closures = self.calendar_overlay.active(self.clock, "zone_closure")
                is_closed = key in self._closed or any(
                    (item["params"]["library_id"], item["params"]["zone_id"]) == key
                    for item in active_closures
                )
                records.append({
                    "timestamp": self.clock.isoformat(sep=" "),
                    "library_id": key[0],
                    "library_name": location["library_name"],
                    "zone_id": key[1],
                    "zone_name": location["zone_name"],
                    "total_capacity": location["total_capacity"],
                    "occupied": occupied,
                    "available": location["total_capacity"] - occupied,
                    "utilization": utilization,
                    "level": self._level(occupied, location["total_capacity"]),
                    "closed": is_closed,
                    "overflow_this_hour": float(self._last_overflow.get(key, 0.0)),
                })
            return records

    def _mae(self) -> dict[str, Any]:
        zone_metrics = {}
        errors = []
        for key, values in self._all_errors.items():
            recent = list(self._errors[key])
            all_values = values
            zone_metrics[f"{key[0]}/{key[1]}"] = {
                "rolling_24h": float(np.mean(recent)) if recent else None,
                "overall": float(np.mean(all_values)) if all_values else None,
                "samples": len(all_values),
            }
            errors.extend(all_values)
        return {
            "zones": zone_metrics,
            "overall_rolling_24h": float(np.mean([value for q in self._errors.values() for value in q])) if any(self._errors.values()) else None,
            "overall": float(np.mean(errors)) if errors else None,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            forecast_rows = self.forecast(3)
            first_step = [row for row in forecast_rows if row["horizon_h"] == 1]
            recommendations = list(self._recommendations)
            if not recommendations and self.recommender_enabled:
                recommendations = []
            series = {}
            for key, frame in self.history.items():
                series[f"{key[0]}/{key[1]}"] = [
                    {"timestamp": row.timestamp.isoformat(sep=" "), "utilization": float(row.occupied_seats / row.total_capacity), "occupied": int(row.occupied_seats)}
                    for row in frame.tail(24).itertuples()
                ]
            calendar = self.calendar_overlay.calendar()
            current_calendar = add_calendar_features(
                pd.DataFrame({"timestamp": [self.clock]}), cfg=self.cfg, calendar=calendar
            ).iloc[0]
            if int(current_calendar["is_exam_period"]):
                academic_status = "exam"
            elif int(current_calendar["is_holiday"]):
                academic_status = "holiday"
            elif int(current_calendar["is_in_semester"]):
                academic_status = f"term week {int(current_calendar['semester_week'])}"
            else:
                academic_status = "outside term"
            ete_days = min(
                (pd.Timestamp(window["start"]) - self.clock.normalize()).days
                for window in calendar.get("exam_windows", [])
                if pd.Timestamp(window["start"]) >= self.clock.normalize()
            ) if any(pd.Timestamp(window["start"]) >= self.clock.normalize() for window in calendar.get("exam_windows", [])) else None
            return {
                "clock": self.clock.isoformat(sep=" "),
                "running": self.running,
                "speed": self.speed,
                "recommender_enabled": self.recommender_enabled,
                "compliance": self.compliance,
                "academic_status": academic_status,
                "ete_in_days": ete_days,
                "snapshot": self.snapshot(),
                "forecast_3h": forecast_rows,
                "recommendations": recommendations,
                "active_scenarios": [item for item in self.calendar_overlay.list(self.clock) if item["active"]],
                "series_24h": series,
                "live_mae": self._mae(),
                "counters": {
                    "total_overflow": float(self.total_overflow),
                    "total_redirected": float(self.total_redirected),
                    "hours_critical": dict(self.hours_critical),
                    "hours_any_critical": self.hours_any_critical,
                },
                "event_log": list(reversed(self._event_log)),
            }

    def reset(
        self,
        start: str | pd.Timestamp | None = None,
        seed: int | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        presets = self.cfg.get("simulation_presets", {})
        if preset is not None:
            if preset not in presets:
                raise ValueError(f"Unknown simulation preset: {preset}")
            start = presets[preset]
        with self._lock:
            self._initialize(start=start, seed=seed, recommender_enabled=True)
            return self.state()

    def export_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        if not destination.is_absolute():
            destination = project_path(str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            frame = pd.concat(self.history.values(), ignore_index=True)
            frame = frame.loc[:, RAW_COLUMNS].sort_values(["timestamp", "library_id", "zone_id"])
            frame.to_csv(destination, index=False)
        return destination