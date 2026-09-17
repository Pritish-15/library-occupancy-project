from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.calendar.store import admin_token, load_calendar, save_calendar
from src.config import load_config, project_path
from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.features.transformers import FEATURE_SETS, occupancy_level, prepare_frame
from src.pipeline.explain import shap_instance
from src.pipeline.predict import load_artifacts
from src.pipeline.recommend import recommend_alternatives

app = FastAPI(title="Smart Library Intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    return prepare_frame(clean_occupancy(load_raw()))


def _latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame["timestamp"].max()
    return frame[frame["timestamp"] == ts].copy()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def current_status():
    frame = _frame()
    snap = _latest_snapshot(frame)
    artifacts = load_artifacts()
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    if "regressor" in artifacts:
        snap["predicted_occupied"] = artifacts["regressor"].predict(snap[cols])
        snap["predicted_utilization"] = snap["predicted_occupied"] / snap["total_capacity"]
    else:
        snap["predicted_occupied"] = snap["occupied_seats"]
        snap["predicted_utilization"] = snap["occupied_seats"] / snap["total_capacity"]
    records = []
    for _, row in snap.iterrows():
        util = float(row["predicted_utilization"])
        records.append(
            {
                "timestamp": str(row["timestamp"]),
                "library_id": row["library_id"],
                "library_name": row.get("library_name", row["library_id"]),
                "zone_id": row["zone_id"],
                "zone_name": row.get("zone_name", row["zone_id"]),
                "total_capacity": int(row["total_capacity"]),
                "occupied_seats": int(row["occupied_seats"]),
                "available_seats": int(row["available_seats"]),
                "predicted_occupied": float(row["predicted_occupied"]),
                "predicted_utilization": util,
                "occupancy_level": occupancy_level(pd.Series([util])).iloc[0],
            }
        )
    return {"as_of": str(snap["timestamp"].iloc[0]), "locations": records}


def _recursive_forecast(horizon: int, library_id: str | None, zone_id: str | None) -> list[dict]:
    artifacts = load_artifacts()
    if "regressor" not in artifacts:
        raise HTTPException(503, "Model artifacts missing. Run python -m src.pipeline.train")
    frame = _frame()
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    model = artifacts["regressor"]
    cfg = load_config()
    lags = cfg["lags"]

    subset = frame
    if library_id:
        subset = subset[subset["library_id"] == library_id]
    if zone_id:
        subset = subset[subset["zone_id"] == zone_id]
    if subset.empty:
        raise HTTPException(404, "Unknown library/zone")

    out = []
    for (lib, zone), grp in subset.groupby(["library_id", "zone_id"]):
        grp = grp.sort_values("timestamp")
        last = grp.iloc[-1].copy()
        history = grp["occupied_seats"].tolist()
        for step in range(1, horizon + 1):
            feat = last[cols].copy()
            for lag in lags:
                feat[f"lag_{lag}"] = history[-lag] if len(history) >= lag else history[-1]
            pred = float(model.predict(pd.DataFrame([feat]))[0])
            pred = max(0.0, min(pred, float(last["total_capacity"])))
            history.append(pred)
            ts = pd.Timestamp(last["timestamp"]) + pd.Timedelta(hours=step)
            out.append(
                {
                    "timestamp": str(ts),
                    "horizon_h": step,
                    "library_id": lib,
                    "zone_id": zone,
                    "library_name": last.get("library_name", lib),
                    "zone_name": last.get("zone_name", zone),
                    "predicted_occupied": pred,
                    "predicted_utilization": pred / float(last["total_capacity"]),
                    "total_capacity": int(last["total_capacity"]),
                }
            )
    return out


@app.get("/forecast")
def forecast(
    horizon: int = Query(1, description="Hours ahead: 1, 3, or 6"),
    library_id: str | None = None,
    zone_id: str | None = None,
):
    if horizon not in (1, 3, 6, 12, 24):
        raise HTTPException(400, "horizon must be 1, 3, 6, 12, or 24")
    return {"horizon": horizon, "forecasts": _recursive_forecast(horizon, library_id, zone_id)}


@app.get("/explain")
def explain(library_id: str, zone_id: str):
    artifacts = load_artifacts()
    if "regressor" not in artifacts:
        raise HTTPException(503, "Model artifacts missing")
    frame = _frame()
    snap = _latest_snapshot(frame)
    row = snap[(snap["library_id"] == library_id) & (snap["zone_id"] == zone_id)]
    if row.empty:
        raise HTTPException(404, "Location not found in latest snapshot")
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    model = artifacts["regressor"]
    return shap_instance(model, row[cols])


@app.get("/recommend")
def recommend(library_id: str, zone_id: str, horizon: int = 1):
    forecasts = _recursive_forecast(horizon, None, None)
    fc = pd.DataFrame(forecasts)
    fc = fc[fc["horizon_h"] == horizon]
    status = current_status()
    snap = pd.DataFrame(status["locations"])
    merged = snap.merge(
        fc[["library_id", "zone_id", "predicted_occupied", "predicted_utilization"]],
        on=["library_id", "zone_id"],
        how="left",
        suffixes=("", "_fc"),
    )
    merged["predicted_utilization"] = merged["predicted_utilization_fc"].fillna(merged["predicted_utilization"])
    recs = recommend_alternatives(merged, library_id, zone_id)
    return {"library_id": library_id, "zone_id": zone_id, "horizon": horizon, "recommendations": recs}


def _require_admin(x_admin_token: str | None) -> None:
    expected = admin_token()
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Admin token required")


@app.get("/calendar")
def get_calendar():
    return load_calendar()


@app.put("/calendar")
def put_calendar(payload: dict, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    try:
        saved = save_calendar(payload, updated_by="api")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _frame.cache_clear()
    return saved


@app.get("/metrics")
def metrics():
    cfg = load_config()
    path = project_path(cfg["paths"]["metrics_path"])
    if not path.exists():
        raise HTTPException(503, "Train first to produce metrics")
    import json

    return json.loads(path.read_text(encoding="utf-8"))
