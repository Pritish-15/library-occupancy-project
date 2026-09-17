import pandas as pd

from src.pipeline.recommend import recommend_alternatives


def _snapshot():
    return pd.DataFrame(
        [
            {
                "library_id": "lib_central",
                "library_name": "Central",
                "zone_id": "z_quiet",
                "zone_name": "Quiet",
                "total_capacity": 100,
                "occupied_seats": 95,
                "predicted_utilization": 0.95,
            },
            {
                "library_id": "lib_central",
                "library_name": "Central",
                "zone_id": "z_collab",
                "zone_name": "Collab",
                "total_capacity": 80,
                "occupied_seats": 20,
                "predicted_utilization": 0.25,
            },
            {
                "library_id": "lib_east",
                "library_name": "East",
                "zone_id": "z_main",
                "zone_name": "Main",
                "total_capacity": 150,
                "occupied_seats": 40,
                "predicted_utilization": 0.27,
            },
        ]
    )


def test_recommend_when_near_capacity_prefers_same_library():
    recs = recommend_alternatives(_snapshot(), "lib_central", "z_quiet")
    assert recs[0]["needed"] is True
    alts = recs[0]["alternatives"]
    assert alts
    assert alts[0]["library_id"] == "lib_central"
    assert alts[0]["zone_id"] == "z_collab"


def test_recommend_not_needed_when_under_threshold():
    recs = recommend_alternatives(_snapshot(), "lib_east", "z_main")
    assert recs[0]["needed"] is False
    assert recs[0]["alternatives"] == []
