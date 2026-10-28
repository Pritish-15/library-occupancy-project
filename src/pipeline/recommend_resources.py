"""Rule-based resource alternatives using predicted resource headroom."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


RESOURCE_COLUMNS = {"computer_usage", "study_room_usage", "book_demand", "charging_station_usage"}
BOOK_ADVISORY = "High book demand expected: consider reserving high-demand titles / promoting digital copies"


def recommend_resource_alternatives(
    resource_forecast_df: pd.DataFrame,
    library_id: str,
    zone_id: str,
    resource: str,
    cfg: dict | None = None,
) -> list[dict]:
    """Recommend resource headroom without inventing inventory/catalog facts."""
    if resource not in RESOURCE_COLUMNS:
        raise ValueError(f"Unsupported resource: {resource}")
    cfg = cfg or load_config()
    threshold = float(cfg["recommendation"]["near_capacity"])
    current = resource_forecast_df[
        (resource_forecast_df["library_id"] == library_id)
        & (resource_forecast_df["zone_id"] == zone_id)
    ]
    if current.empty:
        return []
    row = current.iloc[0]
    value = float(row[resource])
    capacity = float(row.get(f"{resource}_capacity", 0.0))
    utilization = value / capacity if capacity > 0 else 0.0
    if resource == "book_demand":
        if capacity > 0 and value >= capacity * threshold:
            return [{
                "needed": True,
                "resource": resource,
                "utilization": utilization,
                "advisory": BOOK_ADVISORY,
                "alternatives": [],
            }]
        return [{"needed": False, "resource": resource, "utilization": utilization, "advisory": None, "alternatives": []}]
    if utilization < threshold:
        return [{"needed": False, "resource": resource, "utilization": utilization, "alternatives": []}]

    others = resource_forecast_df[
        ~((resource_forecast_df["library_id"] == library_id) & (resource_forecast_df["zone_id"] == zone_id))
    ].copy()
    if "closed" in others:
        others = others[~others["closed"].astype(bool)]
    capacity_col = f"{resource}_capacity"
    others["resource_utilization"] = others[resource] / others[capacity_col].replace(0, float("nan"))
    others["resource_headroom"] = (others[capacity_col] - others[resource]).clip(lower=0)
    others["same_library"] = (others["library_id"] == library_id).astype(int)
    others = others.sort_values(["same_library", "resource_headroom"], ascending=[False, False])
    alternatives = [
        {
            "library_id": item.library_id,
            "zone_id": item.zone_id,
            "library_name": item.library_name,
            "zone_name": item.zone_name,
            "resource_headroom": float(item.resource_headroom),
            "resource_utilization": float(item.resource_utilization) if pd.notna(item.resource_utilization) else 0.0,
            "same_library": bool(item.same_library),
        }
        for item in others.head(int(cfg["recommendation"]["top_k"])).itertuples()
    ]
    return [{
        "needed": True,
        "resource": resource,
        "utilization": utilization,
        "alternatives": alternatives,
    }]