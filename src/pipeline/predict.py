"""Load serialized artifacts and run inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, project_path
from src.features.transformers import FEATURE_SETS, prepare_frame


def artifact_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return project_path(cfg["paths"]["models_dir"])


def load_artifacts(cfg: dict | None = None) -> dict:
    models = artifact_dir(cfg)
    bundle = {}
    for name in ("regressor", "classifier"):
        path = models / f"{name}.joblib"
        if path.exists():
            bundle[name] = joblib.load(path)
    meta_path = models / "meta.joblib"
    if meta_path.exists():
        bundle["meta"] = joblib.load(meta_path)
    return bundle


def predict_occupancy(df: pd.DataFrame, artifacts: dict | None = None, feature_set: str = "D") -> pd.DataFrame:
    artifacts = artifacts or load_artifacts()
    if "regressor" not in artifacts:
        raise FileNotFoundError("Train the pipeline first: python -m src.pipeline.train")
    prepared = prepare_frame(df)
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS[feature_set])
    use = [c for c in cols if c in prepared.columns]
    prepared["predicted_occupied"] = artifacts["regressor"].predict(prepared[use])
    prepared["predicted_utilization"] = prepared["predicted_occupied"] / prepared["total_capacity"]
    if "classifier" in artifacts:
        mapping = artifacts.get("meta", {}).get(
            "level_mapping", {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}
        )
        mapping = {int(k): v for k, v in mapping.items()}
        pred = artifacts["classifier"].predict(prepared[use])
        prepared["predicted_level"] = [mapping.get(int(i), str(i)) for i in pred]
    return prepared
