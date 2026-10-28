from copy import deepcopy
import shutil

import pandas as pd

from src.config import load_config, project_path
from src.pipeline.recommend_resources import recommend_resource_alternatives
from src.pipeline.train_resources import RESOURCES, train_resources
from src.sim.engine import SimulationEngine


def test_resource_recommendations_rank_headroom_and_advise_books():
    forecast = pd.DataFrame([
        {"library_id": "lib_central", "zone_id": "source", "library_name": "Central", "zone_name": "Source", "computer_usage": 95, "computer_usage_capacity": 100, "book_demand": 90, "book_demand_capacity": 100, "closed": False},
        {"library_id": "lib_central", "zone_id": "closed", "library_name": "Central", "zone_name": "Closed", "computer_usage": 0, "computer_usage_capacity": 100, "book_demand": 0, "book_demand_capacity": 100, "closed": True},
        {"library_id": "lib_central", "zone_id": "same", "library_name": "Central", "zone_name": "Same", "computer_usage": 20, "computer_usage_capacity": 100, "book_demand": 10, "book_demand_capacity": 100, "closed": False},
        {"library_id": "lib_east", "zone_id": "other", "library_name": "East", "zone_name": "Other", "computer_usage": 5, "computer_usage_capacity": 100, "book_demand": 5, "book_demand_capacity": 100, "closed": False},
    ])
    alternatives = recommend_resource_alternatives(forecast, "lib_central", "source", "computer_usage")
    assert alternatives[0]["needed"] is True
    assert alternatives[0]["alternatives"][0]["zone_id"] == "same"
    assert "closed" not in [item["zone_id"] for item in alternatives[0]["alternatives"]]
    books = recommend_resource_alternatives(forecast, "lib_central", "source", "book_demand")
    assert books[0]["advisory"].startswith("High book demand expected:")
    assert "alternatives" in books[0] and books[0]["alternatives"] == []


def test_resource_training_saves_metrics_and_forecasts_nonnegative(tmp_path):
    cfg = deepcopy(load_config())
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["resource_training"] = {"sample_rows": 3600, "n_estimators": 5, "max_depth": 2}
    target_models = project_path(cfg["paths"]["models_dir"])
    target_models.mkdir(parents=True, exist_ok=True)
    source_models = project_path(load_config()["paths"]["models_dir"])
    shutil.copy2(source_models / "meta.joblib", target_models / "meta.joblib")
    shutil.copy2(source_models / "regressor.joblib", target_models / "regressor.joblib")

    metrics = train_resources(cfg)
    assert set(metrics) == set(RESOURCES)
    for resource in RESOURCES:
        assert {"mae", "rmse", "r2", "mape"} <= set(metrics[resource]["model"])
        assert {"historical_mean", "lag_1", "lag_24", "lag_168"} == set(metrics[resource]["baselines"])
        assert (target_models / f"resource_{resource}.joblib").exists()
    assert (target_models / "resource_metrics.json").exists()
    assert (target_models / "resource_meta.joblib").exists()
    assert (target_models / "resource_capacity.json").exists()

    engine = SimulationEngine(cfg=cfg, seed=3, recommender_enabled=False)
    resource_meta = engine._resource_meta or __import__("joblib").load(target_models / "resource_meta.joblib")
    for resource, values in resource_meta["resources"].items():
        columns = values["feature_cols"]
        assert "occupied_seats" not in columns
        assert resource not in columns
    engine.add_scenario("zone_closure", library_id="lib_central", zone_id="z_quiet", hours=2)
    forecast = engine.forecast_resources(3)
    assert len(forecast) == 27
    assert all(row[resource] >= 0 for row in forecast for resource in RESOURCES)
    closed_zone = next(row for row in forecast if row["library_id"] == "lib_central" and row["zone_id"] == "z_quiet")
    assert closed_zone["closed"] is True
    reopened_zone = next(
        row for row in forecast
        if row["library_id"] == "lib_central" and row["zone_id"] == "z_quiet" and row["horizon_h"] == 3
    )
    assert reopened_zone["closed"] is False