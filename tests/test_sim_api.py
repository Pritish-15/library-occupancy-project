from fastapi.testclient import TestClient

from app.api.main import app
from app.api import sim
from src.sim.compare import run_comparison
from src.sim.engine import SimulationEngine


def test_comparison_includes_overflow_and_measurable_redirects():
    result = run_comparison(days=14, seed=7)
    with_recommender = result["with_recommender"]
    without_recommender = result["without_recommender"]
    assert with_recommender["total_overflow"] > 0
    assert with_recommender["total_redirected"] > 0
    assert with_recommender["total_overflow"] < without_recommender["total_overflow"]


def test_simulation_api_routes_and_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(sim, "_ENGINE", None)
    monkeypatch.setattr(sim, "_COMPARISONS", {})
    monkeypatch.setattr(sim, "project_path", lambda *parts: tmp_path.joinpath(*parts))
    def fake_resource_forecast(self, horizon=3):
        rows = []
        for step in range(1, horizon + 1):
            for location in self.snapshot():
                row = {
                    "timestamp": self.clock.isoformat(sep=" "),
                    "horizon_h": step,
                    "library_id": location["library_id"],
                    "zone_id": location["zone_id"],
                    "library_name": location["library_name"],
                    "zone_name": location["zone_name"],
                    "closed": False,
                }
                for resource in ("computer_usage", "study_room_usage", "book_demand", "charging_station_usage"):
                    row[resource] = 1.0
                    row[f"{resource}_capacity"] = 100.0
                rows.append(row)
        return rows

    monkeypatch.setattr(SimulationEngine, "forecast_resources", fake_resource_forecast)
    client = TestClient(app)

    started = client.post("/sim/start", json={"preset": "normal_week", "seed": 19})
    assert started.status_code == 200
    assert started.json()["clock"].startswith("2026-09-14")
    assert client.get("/sim/state").status_code == 200
    stepped = client.post("/sim/step", json={"hours": 1})
    assert stepped.status_code == 200
    assert stepped.json()["clock"].endswith("07:00:00")

    scenario = client.post("/sim/scenario", json={"kind": "event_spike", "days": 2})
    assert scenario.status_code == 200
    scenario_id = scenario.json()["scenario"]["id"]
    assert client.delete(f"/sim/scenario/{scenario_id}").status_code == 200
    assert client.post("/sim/recommender", json={"enabled": False}).status_code == 200
    assert len(client.get("/sim/forecast?horizon=3").json()) == 27
    assert client.get("/sim/recommend?library_id=lib_central&zone_id=z_quiet").status_code == 200
    resources = client.get("/sim/resources?horizon=3")
    assert resources.status_code == 200
    assert len(resources.json()) == 27
    assert client.get("/sim/compare?days=1&seed=19").status_code == 200
    exported = client.get("/sim/export.csv")
    assert exported.status_code == 200
    assert exported.text.startswith("timestamp,library_id,library_name,zone_id")

    assert client.post("/sim/step", json={"hours": 49}).status_code == 422
    assert client.post("/sim/start", json={"preset": "not-a-preset"}).status_code == 400
    assert client.get("/sim/forecast?horizon=2").status_code == 400
    assert client.get("/sim/recommend?library_id=missing&zone_id=z_quiet").status_code == 404
    assert client.post("/sim/scenario", json={"kind": "zone_closure", "library_id": "lib_central", "zone_id": "bad"}).status_code == 404
    assert client.delete("/sim/scenario/unknown").status_code == 404
    assert client.post("/sim/reset").status_code == 200