"""Constraint-based resource recommendations from predicted occupancy."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def recommend_alternatives(
    snapshot: pd.DataFrame,
    library_id: str,
    zone_id: str,
    cfg: dict | None = None,
) -> list[dict]:
    cfg = cfg or load_config()
    rules = cfg["recommendation"]
    current = snapshot[(snapshot["library_id"] == library_id) & (snapshot["zone_id"] == zone_id)]
    if current.empty:
        return []
    row = current.iloc[0]
    util = float(row.get("predicted_utilization", row["occupied_seats"] / row["total_capacity"]))
    if util < rules["near_capacity"]:
        return [
            {
                "needed": False,
                "reason": "Location is below the near-capacity threshold.",
                "utilization": util,
                "alternatives": [],
            }
        ]

    others = snapshot[~((snapshot["library_id"] == library_id) & (snapshot["zone_id"] == zone_id))].copy()
    others["predicted_utilization"] = others.get(
        "predicted_utilization",
        others["occupied_seats"] / others["total_capacity"],
    )
    others["remaining"] = others["total_capacity"] * (1 - others["predicted_utilization"])
    others["same_library"] = (others["library_id"] == library_id).astype(int)
    others = others.sort_values(["same_library", "remaining"], ascending=[False, False])
    top = others.head(int(rules["top_k"]))
    alts = []
    for _, alt in top.iterrows():
        alts.append(
            {
                "library_id": alt["library_id"],
                "library_name": alt.get("library_name", alt["library_id"]),
                "zone_id": alt["zone_id"],
                "zone_name": alt.get("zone_name", alt["zone_id"]),
                "predicted_utilization": float(alt["predicted_utilization"]),
                "remaining_seats": float(alt["remaining"]),
                "same_library": bool(alt["same_library"]),
            }
        )
    return [
        {
            "needed": True,
            "reason": "Predicted utilization is near or above capacity; alternatives ranked by remaining seats.",
            "utilization": util,
            "alternatives": alts,
        }
    ]
