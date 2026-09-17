import pandas as pd

from src.features.lags import add_lag_features


def test_lags_use_only_past_values_within_group():
    rows = []
    for lib, zone in [("a", "z1"), ("b", "z2")]:
        for i in range(10):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                    "library_id": lib,
                    "zone_id": zone,
                    "occupied_seats": i * 10 + (0 if lib == "a" else 1000),
                }
            )
    df = pd.DataFrame(rows)
    out = add_lag_features(df, lags=[1, 2, 3])
    a = out[(out["library_id"] == "a")].sort_values("timestamp").reset_index(drop=True)
    assert pd.isna(a.loc[0, "lag_1"])
    assert a.loc[3, "lag_1"] == a.loc[2, "occupied_seats"]
    assert a.loc[3, "lag_2"] == a.loc[1, "occupied_seats"]
    assert a.loc[3, "lag_3"] == a.loc[0, "occupied_seats"]
    b = out[(out["library_id"] == "b")].sort_values("timestamp").reset_index(drop=True)
    assert b.loc[1, "lag_1"] == 1000
    assert b.loc[1, "lag_1"] != a.loc[1, "occupied_seats"]


def test_lags_do_not_peek_at_future():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "library_id": ["x"] * 5,
            "zone_id": ["z"] * 5,
            "occupied_seats": [1, 2, 3, 4, 5],
        }
    )
    out = add_lag_features(df, lags=[1])
    for i in range(1, 5):
        assert out.loc[i, "lag_1"] == out.loc[i - 1, "occupied_seats"]
        assert out.loc[i, "lag_1"] != out.loc[i, "occupied_seats"] or i == 0
