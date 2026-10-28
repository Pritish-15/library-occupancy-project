import pandas as pd
from pandas.api.types import is_integer_dtype

from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.features.transformers import prepare_frame
from src.sim.engine import SimulationEngine


def test_warmup_features_and_location_codes_match_training_frame():
    engine = SimulationEngine(recommender_enabled=False)
    assert min(len(frame) for frame in engine.history.values()) >= 169
    prepared = engine.features(engine.history[("lib_central", "z_quiet")])
    assert list(prepared.loc[:, engine.feature_cols].columns) == engine.feature_cols

    training = prepare_frame(clean_occupancy(load_raw()))
    libraries = training[["library_id", "library_code"]].drop_duplicates()
    zones = training[["zone_id", "zone_code"]].drop_duplicates()
    assert engine.library_codes == dict(zip(libraries["library_id"], libraries["library_code"]))
    assert engine.zone_codes == dict(zip(zones["zone_id"], zones["zone_code"]))


def test_same_seed_repeats_history_and_different_seed_changes_it():
    first = SimulationEngine(seed=31, recommender_enabled=False)
    repeat = SimulationEngine(seed=31, recommender_enabled=False)
    different = SimulationEngine(seed=32, recommender_enabled=False)
    for key in first.history:
        pd.testing.assert_frame_equal(first.history[key], repeat.history[key])
    assert not first.history[("lib_central", "z_quiet")].equals(
        different.history[("lib_central", "z_quiet")]
    )


def test_exam_surge_increases_weekday_daytime_mean():
    start = "2027-01-04 06:00"
    baseline = SimulationEngine(start=start, seed=13, recommender_enabled=False)
    exam = SimulationEngine(start=start, seed=13, recommender_enabled=False)
    exam.add_scenario("exam_surge", days=5)
    baseline.step(48, return_state=False)
    exam.step(48, return_state=False)
    base_rows = pd.concat(baseline.history.values(), ignore_index=True)
    exam_rows = pd.concat(exam.history.values(), ignore_index=True)
    base_mask = (base_rows["timestamp"] > pd.Timestamp(start)) & (base_rows["timestamp"].dt.day_of_week < 5)
    base_mask &= base_rows["timestamp"].dt.hour.between(9, 17)
    exam_mask = (exam_rows["timestamp"] > pd.Timestamp(start)) & (exam_rows["timestamp"].dt.day_of_week < 5)
    exam_mask &= exam_rows["timestamp"].dt.hour.between(9, 17)
    base_rows = base_rows.loc[base_mask]
    exam_rows = exam_rows.loc[exam_mask]
    exam_utilization = (exam_rows["occupied_seats"] / exam_rows["total_capacity"]).mean()
    base_utilization = (base_rows["occupied_seats"] / base_rows["total_capacity"]).mean()
    assert exam_utilization / base_utilization > 1.3


def test_zone_closure_redistributes_same_library_demand():
    start = "2026-11-30 10:00"
    baseline = SimulationEngine(start=start, seed=17, recommender_enabled=False)
    closed = SimulationEngine(start=start, seed=17, recommender_enabled=False)
    closed.add_scenario("zone_closure", library_id="lib_central", zone_id="z_quiet", hours=6)
    baseline.step(1, return_state=False)
    closed.step(1, return_state=False)
    close_row = closed.history[("lib_central", "z_quiet")].iloc[-1]
    assert close_row["occupied_seats"] == 0
    base_other = sum(
        baseline.history[("lib_central", zone)].iloc[-1]["occupied_seats"]
        for zone in ("z_collab", "z_computing")
    )
    closed_other = sum(
        closed.history[("lib_central", zone)].iloc[-1]["occupied_seats"]
        for zone in ("z_collab", "z_computing")
    )
    assert closed_other > base_other


def test_occupancy_capacity_availability_and_integer_schema():
    engine = SimulationEngine(recommender_enabled=False)
    engine.step(6, return_state=False)
    integer_columns = [
        "occupied_seats", "available_seats", "computer_usage", "study_room_usage",
        "book_demand", "charging_station_usage",
    ]
    for frame in engine.history.values():
        assert (frame["occupied_seats"] <= frame["total_capacity"]).all()
        assert (frame["available_seats"] == frame["total_capacity"] - frame["occupied_seats"]).all()
        assert all(is_integer_dtype(frame[column]) for column in integer_columns)


def test_forecast_recurses_all_zones_with_advancing_time_context():
    engine = SimulationEngine(start="2026-11-30 22:00", recommender_enabled=False)
    forecasts = engine.forecast(3)
    assert len(forecasts) == 27
    assert all(0 <= row["predicted_occupied"] <= row["total_capacity"] for row in forecasts)
    quiet = [row for row in forecasts if row["zone_id"] == "z_quiet"]
    assert [row["hour"] for row in quiet] == [23, 0, 1]


def test_recommender_reduces_overflow_in_exam_spike():
    engines = [
        SimulationEngine(start="2026-12-10 06:00", seed=7, recommender_enabled=enabled)
        for enabled in (True, False)
    ]
    for engine in engines:
        engine.add_scenario("exam_surge", days=5)
        engine.add_scenario(
            "random_spike", library_id="lib_central", zone_id="z_quiet", magnitude=1.8, hours=6
        )
        engine.step(24, return_state=False)
    assert engines[0].total_overflow <= engines[1].total_overflow
    assert engines[0].total_redirected >= 0