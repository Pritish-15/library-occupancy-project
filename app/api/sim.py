"""API routes for the isolated virtual simulation engine."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import load_config, project_path
from src.pipeline.recommend import recommend_alternatives
from src.pipeline.recommend_resources import recommend_resource_alternatives
from src.sim.compare import run_comparison
from src.sim.engine import SimulationEngine


router = APIRouter(prefix="/sim", tags=["simulation"])
_LOCK = threading.RLock()
_ENGINE: SimulationEngine | None = None
_COMPARISONS: dict[tuple[int, int, str | None, float], dict] = {}


class StartPayload(BaseModel):
    preset: str | None = None
    start: str | None = None
    seed: int | None = None


class StepPayload(BaseModel):
    hours: int = Field(default=1, ge=1, le=48)


class ScenarioPayload(BaseModel):
    kind: Literal["exam_surge", "zone_closure", "holiday", "event_spike", "random_spike"]
    library_id: str | None = None
    zone_id: str | None = None
    hours: int | None = Field(default=None, ge=1, le=24 * 31)
    days: int | None = Field(default=None, ge=1, le=90)
    magnitude: float | None = Field(default=None, ge=1.3, le=1.8)


class RecommenderPayload(BaseModel):
    enabled: bool
    compliance: float | None = Field(default=None, ge=0, le=1)


def _engine() -> SimulationEngine:
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            try:
                _ENGINE = SimulationEngine()
            except FileNotFoundError as exc:
                raise HTTPException(503, str(exc)) from exc
        return _ENGINE


def _valid_location(library_id: str | None, zone_id: str | None) -> None:
    if not library_id and not zone_id:
        return
    cfg = load_config()
    for library in cfg["libraries"]:
        if library_id and library["id"] != library_id:
            continue
        for zone in library["zones"]:
            if zone_id and zone["id"] != zone_id:
                continue
            return
    raise HTTPException(404, "Unknown library or zone")


@router.get("/state")
def state():
    return _engine().state()


@router.post("/start")
def start(payload: StartPayload = StartPayload()):
    global _ENGINE
    cfg = load_config()
    if payload.preset is not None and payload.preset not in cfg.get("simulation_presets", {}):
        raise HTTPException(400, f"Unknown simulation preset: {payload.preset}")
    if payload.preset and payload.start:
        raise HTTPException(400, "Provide either preset or start, not both")
    with _LOCK:
        try:
            _ENGINE = SimulationEngine(
                cfg=cfg,
                start=payload.start or cfg.get("simulation_presets", {}).get(payload.preset),
                seed=payload.seed,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        return _ENGINE.state()


@router.post("/step")
def step(payload: StepPayload):
    try:
        return _engine().step(payload.hours)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/reset")
def reset():
    try:
        return _engine().reset()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/scenario")
def add_scenario(payload: ScenarioPayload):
    engine = _engine()
    _valid_location(payload.library_id, payload.zone_id)
    params = payload.model_dump(exclude={"kind"}, exclude_none=True)
    try:
        return {"scenario": engine.add_scenario(payload.kind, **params), "state": engine.state()}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/scenario/{scenario_id}")
def cancel_scenario(scenario_id: str):
    engine = _engine()
    if not engine.cancel_scenario(scenario_id):
        raise HTTPException(404, "Unknown scenario")
    return engine.state()


@router.post("/recommender")
def recommender(payload: RecommenderPayload):
    try:
        engine = _engine()
        engine.set_recommender(payload.enabled, payload.compliance)
        return engine.state()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/forecast")
def forecast(horizon: int = Query(default=3)):
    try:
        return _engine().forecast(horizon)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/recommend")
def recommend(library_id: str, zone_id: str, horizon: int = Query(default=1)):
    _valid_location(library_id, zone_id)
    engine = _engine()
    try:
        forecasts = engine.forecast(horizon)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    predicted = {
        (row["library_id"], row["zone_id"]): row
        for row in forecasts
        if row["horizon_h"] == horizon
    }
    snapshot = []
    for current in engine.snapshot():
        key = (current["library_id"], current["zone_id"])
        row = predicted.get(key)
        if row is None or current["closed"]:
            continue
        snapshot.append({
            **current,
            "occupied_seats": current["occupied"],
            "predicted_occupied": row["predicted_occupied"],
            "predicted_utilization": row["predicted_utilization"],
            "total_capacity": row["total_capacity"],
        })
    recs = recommend_alternatives(pd.DataFrame(snapshot), library_id, zone_id, engine.cfg)
    if not recs:
        raise HTTPException(404, "Location not available in the simulation snapshot")
    return {"library_id": library_id, "zone_id": zone_id, "horizon": horizon, "recommendations": recs}


@router.get("/resources")
def resources(horizon: int = Query(default=3)):
    engine = _engine()
    try:
        forecasts = engine.forecast_resources(horizon)
        rows = []
        frame = pd.DataFrame(forecasts)
        for row in forecasts:
            item = dict(row)
            for resource in ("computer_usage", "study_room_usage", "book_demand", "charging_station_usage"):
                item[f"{resource}_recommendations"] = recommend_resource_alternatives(
                    frame[frame["horizon_h"] == row["horizon_h"]],
                    row["library_id"],
                    row["zone_id"],
                    resource,
                    engine.cfg,
                )
            rows.append(item)
        return rows
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/compare")
def compare(
    days: int = Query(default=14, ge=1, le=60),
    compliance: float = Query(default=0.4, ge=0, le=1),
    seed: int = Query(default=7),
    preset: str | None = None,
):
    key = (days, seed, preset, compliance)
    with _LOCK:
        if key not in _COMPARISONS:
            try:
                _COMPARISONS[key] = run_comparison(days, seed, preset, compliance)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return _COMPARISONS[key]


@router.get("/export.csv")
def export_csv():
    path = project_path("data", "uploads", "simulation_export.csv")
    _engine().export_csv(path)
    return FileResponse(path, media_type="text/csv", filename="simulation_history.csv")


@router.get("/resources/metrics")
def resource_metrics():
    path = project_path("models", "resource_metrics.json")
    if not path.exists():
        raise HTTPException(503, "Resource metrics missing. Run python -m src.pipeline.train_resources")
    import json

    return json.loads(path.read_text(encoding="utf-8"))