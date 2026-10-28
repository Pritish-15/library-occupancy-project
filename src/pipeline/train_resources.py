"""Train synthetic resource-demand regressors without concurrent-hour leakage."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config, project_path
from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.evaluation.metrics import regression_metrics
from src.evaluation.splits import chronological_split
from src.features.calendar import add_calendar_features
from src.features.temporal import add_temporal_features
from src.features.transformers import encode_locations
from src.models.regressors import make_xgboost


RESOURCES = ("book_demand", "computer_usage", "study_room_usage", "charging_station_usage")
LAGS = (1, 2, 3, 6, 24, 168)
CONTEXT_COLUMNS = (
    "total_capacity", "hour", "day", "weekday", "is_weekend", "month", "semester_week",
    "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester", "library_code", "zone_code",
)


def _feature_frame(cfg: dict) -> pd.DataFrame:
    frame = clean_occupancy(load_raw())
    frame = add_temporal_features(frame)
    frame = add_calendar_features(frame, cfg=cfg)
    frame = encode_locations(frame)
    frame = frame.sort_values(["library_id", "zone_id", "timestamp"]).copy()
    for lag in LAGS:
        frame[f"occupied_lag_{lag}"] = frame.groupby(["library_id", "zone_id"], sort=False)["occupied_seats"].shift(lag)
        for resource in RESOURCES:
            frame[f"{resource}_lag_{lag}"] = frame.groupby(["library_id", "zone_id"], sort=False)[resource].shift(lag)
    return frame.dropna(subset=[f"occupied_lag_{lag}" for lag in LAGS] + [f"{r}_lag_{lag}" for r in RESOURCES for lag in LAGS])


def _resource_features(resource: str) -> list[str]:
    return [
        *[f"occupied_lag_{lag}" for lag in LAGS],
        *[f"{resource}_lag_{lag}" for lag in LAGS],
        *CONTEXT_COLUMNS,
    ]


def _historical_mean(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    means = train.groupby(["library_id", "zone_id", "hour"])[target].mean()
    global_mean = float(train[target].mean())
    return np.asarray(
        [means.get((row.library_id, row.zone_id, row.hour), global_mean) for row in test.itertuples()],
        dtype=float,
    )


def train_resources(cfg: dict | None = None) -> dict:
    """Train one resource model per target and save model/metric metadata."""
    cfg = cfg or load_config()
    ensure_dirs()
    frame = _feature_frame(cfg)
    sample_rows = cfg.get("resource_training", {}).get("sample_rows")
    if sample_rows:
        frame = frame.sort_values("timestamp").tail(int(sample_rows)).reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(frame, cfg)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    model_settings = cfg["model"]
    resource_settings = cfg.get("resource_training", {})
    models_dir = project_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}
    resource_meta: dict[str, dict] = {"resources": {}, "resource_capacity": {}}
    for (library_id, zone_id), group in train_df.groupby(["library_id", "zone_id"]):
        resource_meta["resource_capacity"].setdefault(str(library_id), {})[str(zone_id)] = {
            resource: float(group[resource].quantile(0.99)) for resource in RESOURCES
        }

    for resource in RESOURCES:
        feature_cols = _resource_features(resource)
        model = make_xgboost(
            n_estimators=int(resource_settings.get("n_estimators", model_settings["n_estimators"])),
            max_depth=int(resource_settings.get("max_depth", model_settings["max_depth"])),
            random_state=int(model_settings["random_state"]),
        )
        model.fit(fit_df.loc[:, feature_cols], fit_df[resource].to_numpy(dtype=float))
        prediction = model.predict(test_df.loc[:, feature_cols])
        baselines = {
            "historical_mean": _historical_mean(fit_df, test_df, resource),
            "lag_1": test_df[f"{resource}_lag_1"].to_numpy(dtype=float),
            "lag_24": test_df[f"{resource}_lag_24"].to_numpy(dtype=float),
            "lag_168": test_df[f"{resource}_lag_168"].to_numpy(dtype=float),
        }
        metrics[resource] = {
            "model": regression_metrics(test_df[resource], prediction),
            "baselines": {name: regression_metrics(test_df[resource], values) for name, values in baselines.items()},
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        }
        joblib.dump(model, models_dir / f"resource_{resource}.joblib")
        resource_meta["resources"][resource] = {"feature_cols": feature_cols}

    joblib.dump(resource_meta, models_dir / "resource_meta.joblib")
    metrics_path = models_dir / "resource_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (models_dir / "resource_capacity.json").write_text(
        json.dumps(resource_meta["resource_capacity"], indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    result = train_resources()
    for resource, values in result.items():
        print(f"{resource}: model MAE={values['model']['mae']:.3f}")


if __name__ == "__main__":
    main()