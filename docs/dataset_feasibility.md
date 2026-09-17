# Dataset feasibility report

## Target schema (ideal)

Hourly records with `timestamp, library_id, zone_id, total_capacity, occupied_seats, available_seats` plus resource context (`computer_usage`, `study_room_usage`, `book_demand`, `charging_station_usage`).

## Current status

| Item | Status | Notes |
| --- | --- | --- |
| Synthetic generator | Implemented | `src/data/generate.py` writes `data/raw/occupancy.csv` |
| Real sensor / Wi-Fi / gate counts | Not acquired | Swap via `src/data/load.py` → `load_raw()` |
| Licensing | N/A for synthetic | Record license + IRB for real data here |
| Missingness | Injected gaps | Cleaner interpolates within library/zone |
| Granularity | Hourly | Matches forecast horizons 1h / 3h / 6h |

## Feasibility questions (complete when evaluating a real source)

1. Can zone-level occupancy be reconstructed at hourly resolution for ≥ 1 academic year?
2. Are exam calendars and closures available as a join key?
3. Is personally identifiable tracking avoided (counts only)?
4. What is the lag between sensor event and warehouse availability?
5. Are capacities stable, or do rooms get reconfigured mid-semester?

## Swap-in procedure

1. Export a CSV with the required columns.
2. Place it at the path in `config.yaml` → `paths.raw_data`.
3. Re-run `python -m src.pipeline.train`. Downstream models do not depend on the generator.
