# Smart Library Occupancy & Resource Intelligence System — digital-twin integration report

Evidence reflects the workspace inspected on 2026-10-23. Runtime commands were attempted with the configured `python` executable. Any item marked **NOT RUN** or **NOT MEASURED** was not inferred.

## 1. Environment and health

- OS: Windows 11 (`Windows-11-10.0.26200-SP0`).
- Python: `Python 3.14.6`.
- `pytest -q`: **NOT RUN** successfully. `pytest` is not a recognized command, and `python -m pytest -q` reported `No module named pytest`. No test pass/fail counts or test failures are available.
- `python -m uvicorn app.api.main:app --port 8097`: **NOT STARTED**. It reported `No module named uvicorn`; HTTP `/health` was not reached.
- Dependency probe found `pandas=True`; `fastapi=False`, `uvicorn=False`, `pytest=False`, `xgboost=False`, `shap=False`.
- `data/raw/occupancy.csv`: present, 11,489,085 bytes.
- `models/*.joblib`: present: `classifier.joblib` (1,372,951 bytes), `meta.joblib` (475 bytes), `regressor.joblib` (789,672 bytes).
- `models/*.json`: present: `ablation.json`, `best_params.json`, `error_analysis.json`, `metrics.json`, `shap_summary.json`.

Full contents of `requirements.txt`:

```text
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
statsmodels>=0.14.0
shap>=0.45.0
joblib>=1.3.0
pyyaml>=6.0
fastapi>=0.110.0
uvicorn>=0.27.0
plotly>=5.20.0
matplotlib>=3.8.0
seaborn>=0.13.0
httpx>=0.27.0
pytest>=8.0.0
jupyter>=1.0.0
pydantic>=2.6.0
python-multipart>=0.0.9
ultralytics>=8.2.0
opencv-python>=4.9.0
```

## 2. `config.yaml`

Full contents:

```yaml
project:
  name: Smart Library Occupancy & Resource Intelligence
  seed: 42

paths:
  raw_data: data/raw/occupancy.csv
  processed_data: data/processed/occupancy.csv
  models_dir: models
  metrics_path: models/metrics.json
  ablation_path: models/ablation.json
  best_params_path: models/best_params.json
  shap_path: models/shap_summary.json
  error_analysis_path: models/error_analysis.json
  academic_calendar: data/calendar/academic_calendar.json

generation:
  start: "2024-08-01"
  end: "2026-01-31"
  freq: "h"
  outlier_rate: 0.004
  missing_gap_rate: 0.0015
  noise_scale: 0.06

libraries:
  - id: lib_central
    name: Central Library
    zones:
      - {id: z_quiet, name: Quiet Study, capacity: 140}
      - {id: z_collab, name: Collaborative, capacity: 90}
      - {id: z_computing, name: Computing Lab, capacity: 60}
  - id: lib_science
    name: Science Library
    zones:
      - {id: z_stacks, name: Stacks Reading, capacity: 110}
      - {id: z_group, name: Group Study, capacity: 70}
      - {id: z_media, name: Media Commons, capacity: 40}
  - id: lib_east
    name: East Campus Library
    zones:
      - {id: z_main, name: Main Hall, capacity: 160}
      - {id: z_rooms, name: Study Rooms, capacity: 48}
      - {id: z_late, name: Late-Night Wing, capacity: 80}

semester:
  fall_start_month: 8
  fall_start_day: 26
  spring_start_month: 1
  spring_start_day: 13
  weeks: 16

exam_windows:
  - {start: "2024-12-09", end: "2024-12-20"}
  - {start: "2025-05-05", end: "2025-05-16"}
  - {start: "2025-12-08", end: "2025-12-19"}

holidays:
  - "2024-09-02"
  - "2024-11-28"
  - "2024-11-29"
  - "2024-12-24"
  - "2024-12-25"
  - "2025-01-01"
  - "2025-01-20"
  - "2025-05-26"
  - "2025-07-04"
  - "2025-09-01"
  - "2025-11-27"
  - "2025-11-28"
  - "2025-12-24"
  - "2025-12-25"
  - "2026-01-01"

academic_events:
  - {start: "2024-10-14", end: "2024-10-18", name: midterms}
  - {start: "2025-03-10", end: "2025-03-14", name: midterms}
  - {start: "2025-10-13", end: "2025-10-17", name: midterms}

utilization_thresholds:
  low: 0.40
  moderate: 0.70
  high: 0.90
  # critical is above high

recommendation:
  near_capacity: 0.85
  prefer_same_library: true
  top_k: 5

splits:
  train: 0.70
  val: 0.15
  test: 0.15

lags: [1, 2, 3, 6, 24, 168]

model:
  target: occupied_seats
  random_state: 42
  n_estimators: 80
  max_depth: 6
  n_iter_search: 6
  shap_sample_size: 200
  cv_splits: 3

api:
  host: 127.0.0.1
  port: 8097

vision:
  max_upload_mb: 40
  max_frames: 180
  stride: 2
  conf: 0.35
```

## 3. Data layer

### Raw CSV facts

`data/raw/occupancy.csv` has 118,377 data rows; observed timestamp range is `2024-08-01 00:00:00` through `2026-01-31 00:00:00`; there are 9 distinct `(library_id, zone_id)` combinations.

First five records (header plus five rows):

```csv
timestamp,library_id,library_name,zone_id,zone_name,total_capacity,occupied_seats,available_seats,computer_usage,study_room_usage,book_demand,charging_station_usage
2024-08-01 00:00:00,lib_central,Central Library,z_collab,Collaborative,90,9.0,81.0,5.0,4.0,3,3
2024-08-01 00:00:00,lib_central,Central Library,z_computing,Computing Lab,60,5.0,55.0,1.0,1.0,0,3
2024-08-01 00:00:00,lib_central,Central Library,z_quiet,Quiet Study,140,10.0,130.0,4.0,0.0,0,2
2024-08-01 00:00:00,lib_east,East Campus Library,z_late,Late-Night Wing,80,1.0,79.0,0.0,1.0,0,0
2024-08-01 00:00:00,lib_east,East Campus Library,z_main,Main Hall,160,0.0,160.0,0.0,0.0,1,4
```

Actual `pandas.read_csv(..., parse_dates=["timestamp"]).dtypes` in the available pandas/Python environment:

```text
timestamp                 datetime64[us]
library_id                           str
library_name                         str
zone_id                              str
zone_name                            str
total_capacity                     int64
occupied_seats                   float64
available_seats                  float64
computer_usage                   float64
study_room_usage                  float64
book_demand                        int64
charging_station_usage             int64
```

Raw resource-field ranges and null counts:

| Column | Min | Max | Null count |
|---|---:|---:|---:|
| `book_demand` | 0 | 59 | 0 |
| `computer_usage` | 0 | 41 | 612 |
| `study_room_usage` | 0 | 22 | 612 |
| `charging_station_usage` | 0 | 38 | 0 |

The raw frame uses 36,223,494 bytes of pandas deep memory. After `clean_occupancy` and `prepare_frame`, the measured prepared frame has 116,865 rows and uses 57,145,649 bytes (54.498 MiB) of deep memory. These figures are pandas frame memory only, not training peak memory.

### Full source: `src/data/generate.py`

```python
"""Synthetic occupancy generator matching the ideal dataset schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar.store import active_term_for, load_calendar, semester_week
from src.config import ensure_dirs, load_config, data_path


HOUR_PROFILE = np.array(
    [
        0.08, 0.05, 0.04, 0.03, 0.03, 0.05, 0.12, 0.28,
        0.48, 0.62, 0.72, 0.78, 0.82, 0.80, 0.76, 0.74,
        0.78, 0.84, 0.80, 0.68, 0.50, 0.36, 0.22, 0.14,
    ]
)

WEEKDAY_MULT = np.array([0.92, 1.05, 1.08, 1.10, 1.02, 0.55, 0.42])  # Mon..Sun


def _in_windows(ts: pd.Timestamp, windows: list[dict]) -> bool:
    d = ts.normalize()
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        if start <= d <= end:
            return True
    return False


def generate_occupancy(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    calendar = load_calendar()
    rng = np.random.default_rng(cfg["project"]["seed"])
    gen = cfg["generation"]
    index = pd.date_range(gen["start"], gen["end"], freq=gen["freq"])

    rows: list[dict] = []
    for lib in cfg["libraries"]:
        lib_bias = rng.normal(1.0, 0.06)
        for zone in lib["zones"]:
            zone_bias = rng.normal(1.0, 0.08)
            capacity = int(zone["capacity"])
            late_boost = 1.25 if "late" in zone["id"] or "computing" in zone["id"] else 1.0
            for ts in index:
                hour = ts.hour
                base = HOUR_PROFILE[hour] * WEEKDAY_MULT[ts.weekday()]
                if hour >= 20:
                    base *= late_boost
                sem_week = semester_week(ts, calendar=calendar, cfg=cfg)
                term = active_term_for(ts, calendar.get("terms") or [])
                if term:
                    ramp = 0.75 + 0.25 * min(sem_week / 6, 1.0)
                    if term.get("classes_end") and term.get("classes_start"):
                        span = max(
                            1,
                            int(
                                (pd.Timestamp(term["classes_end"]) - pd.Timestamp(term["classes_start"])).days // 7
                            )
                            + 1,
                        )
                        if sem_week >= span - 1:
                            ramp *= 0.55
                else:
                    ramp = 0.35
                exam = _in_windows(ts, calendar.get("exam_windows") or [])
                event = _in_windows(ts, calendar.get("academic_events") or [])
                holiday = _in_windows(ts, calendar.get("closures") or [])
                mult = lib_bias * zone_bias * ramp
                if exam:
                    mult *= 1.55
                if event:
                    mult *= 1.18
                if holiday:
                    mult *= 0.18
                noise = rng.normal(0, gen["noise_scale"])
                util = float(np.clip(base * mult + noise, 0.0, 1.02))
                occupied = int(np.clip(round(util * capacity), 0, capacity))
                computer = int(np.clip(round(occupied * (0.25 + 0.2 * ("computing" in zone["id"] or "media" in zone["id"])) + rng.normal(0, 2)), 0, capacity))
                rooms = int(np.clip(round(occupied * (0.12 + 0.25 * ("rooms" in zone["id"] or "group" in zone["id"] or "collab" in zone["id"])) + rng.normal(0, 1)), 0, max(8, capacity // 4)))
                books = int(np.clip(round(occupied * 0.35 + rng.normal(0, 4)), 0, capacity * 2))
                charging = int(np.clip(round(occupied * 0.22 + rng.normal(0, 2)), 0, capacity))
                rows.append(
                    {
                        "timestamp": ts,
                        "library_id": lib["id"],
                        "library_name": lib["name"],
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "total_capacity": capacity,
                        "occupied_seats": occupied,
                        "available_seats": capacity - occupied,
                        "computer_usage": computer,
                        "study_room_usage": rooms,
                        "book_demand": books,
                        "charging_station_usage": charging,
                    }
                )

    df = pd.DataFrame(rows)

    n = len(df)
    n_out = max(1, int(n * gen["outlier_rate"]))
    outlier_idx = rng.choice(n, size=n_out, replace=False)
    df.loc[outlier_idx, "occupied_seats"] = np.clip(
        df.loc[outlier_idx, "occupied_seats"] + rng.integers(15, 45, size=n_out),
        0,
        df.loc[outlier_idx, "total_capacity"],
    )
    df.loc[outlier_idx, "available_seats"] = (
        df.loc[outlier_idx, "total_capacity"] - df.loc[outlier_idx, "occupied_seats"]
    )

    n_gaps = max(1, int(n * gen["missing_gap_rate"]))
    gap_starts = rng.choice(n - 6, size=n_gaps, replace=False)
    miss_cols = ["occupied_seats", "available_seats", "computer_usage", "study_room_usage"]
    for start in gap_starts:
        length = int(rng.integers(1, 5))
        df.loc[start : start + length, miss_cols] = np.nan

    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)


def write_raw(df: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    df = df if df is not None else generate_occupancy()
    path = data_path("raw_data")
    df.to_csv(path, index=False)
    return df


def main() -> None:
    df = write_raw()
    print(f"Wrote {len(df):,} rows to {data_path('raw_data')}")


if __name__ == "__main__":
    main()
```

### Full source: `src/data/load.py`

```python
"""Swappable raw-data loader. Replace generate/write_raw with a real CSV at the same path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import data_path
from src.data.generate import write_raw


def load_raw(path: Path | None = None, generate_if_missing: bool = True) -> pd.DataFrame:
    csv_path = path or data_path("raw_data")
    if not csv_path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Raw occupancy file not found: {csv_path}")
        write_raw()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)
```

### Full source: `src/data/clean.py`

```python
"""Missing-value handling and outlier treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import validate_occupancy


NUMERIC_FILL = [
    "occupied_seats",
    "available_seats",
    "computer_usage",
    "study_room_usage",
    "book_demand",
    "charging_station_usage",
]


def clean_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_occupancy(df, strict=False).copy()
    df = df.sort_values(["library_id", "zone_id", "timestamp"])

    group_cols = ["library_id", "zone_id"]
    for col in NUMERIC_FILL:
        if col not in df.columns:
            continue
        df[col] = df.groupby(group_cols)[col].transform(lambda s: s.interpolate(limit_direction="both"))
        df[col] = df.groupby(group_cols)[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(0)

    util = df["occupied_seats"] / df["total_capacity"].replace(0, np.nan)
    z = df.groupby(group_cols)["occupied_seats"].transform(
        lambda s: (s - s.median()) / (s.std(ddof=0) + 1e-6)
    )
    spike = (z.abs() > 4) & (util > 1.0)
    df.loc[spike, "occupied_seats"] = df.loc[spike, "total_capacity"]

    over = df["occupied_seats"] > df["total_capacity"]
    df.loc[over, "occupied_seats"] = df.loc[over, "total_capacity"]
    df.loc[df["occupied_seats"] < 0, "occupied_seats"] = 0
    df["available_seats"] = df["total_capacity"] - df["occupied_seats"]

    for col in NUMERIC_FILL:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    return df.sort_values(["timestamp", "library_id", "zone_id"]).reset_index(drop=True)
```

## 4. Features and models

### Full source: `src/features/transformers.py`

```python
"""sklearn-compatible transformers for the occupancy feature stack."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import load_config
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features
from src.features.temporal import add_temporal_features


class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_temporal_features(pd.DataFrame(X))


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict | None = None):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_calendar_features(pd.DataFrame(X), cfg=self.config or load_config())


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags: list[int] | None = None, value_col: str = "occupied_seats"):
        self.lags = lags
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_lag_features(pd.DataFrame(X), lags=self.lags, value_col=self.value_col)


FEATURE_SETS = {
    "A": ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity"],
    "B": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
    ],
    "C": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
        "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
    ],
    "D": [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
        "hour", "day", "weekday", "is_weekend", "month",
        "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
        "computer_usage", "study_room_usage", "book_demand", "charging_station_usage",
        "library_code", "zone_code",
    ],
}


def encode_locations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["library_code"] = pd.Categorical(out["library_id"]).codes
    out["zone_code"] = pd.Categorical(out["zone_id"]).codes
    return out


def occupancy_level(utilization: pd.Series, thresholds: dict | None = None) -> pd.Series:
    thresholds = thresholds or load_config()["utilization_thresholds"]
    bins = [-1e-9, thresholds["low"], thresholds["moderate"], thresholds["high"], 10]
    labels = ["Low", "Moderate", "High", "Critical"]
    return pd.cut(utilization, bins=bins, labels=labels).astype(str)


def prepare_frame(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = add_temporal_features(df)
    out = add_calendar_features(out, cfg=cfg)
    out = add_lag_features(out, lags=cfg["lags"])
    out = encode_locations(out)
    out["utilization"] = out["occupied_seats"] / out["total_capacity"]
    out["occupancy_level"] = occupancy_level(out["utilization"], cfg["utilization_thresholds"])
    lag_cols = [f"lag_{k}" for k in cfg["lags"]]
    out = out.dropna(subset=lag_cols)
    return out.reset_index(drop=True)
```

### Full source: feature helpers

```python title="src/features/__init__.py"
from src.features.temporal import add_temporal_features
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features
from src.features.transformers import (
    TemporalFeatures,
    CalendarFeatures,
    LagFeatures,
    FEATURE_SETS,
    prepare_frame,
    occupancy_level,
)

__all__ = [
    "add_temporal_features",
    "add_calendar_features",
    "add_lag_features",
    "TemporalFeatures",
    "CalendarFeatures",
    "LagFeatures",
    "FEATURE_SETS",
    "prepare_frame",
    "occupancy_level",
]
```

```python title="src/features/temporal.py"
"""Hour / day / weekday / weekend / month features."""

from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["day"] = ts.dt.day
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    return out
```

```python title="src/features/lags.py"
"""Per library/zone lag features without future leakage."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    value_col: str = "occupied_seats",
    group_cols: tuple[str, ...] = ("library_id", "zone_id"),
) -> pd.DataFrame:
    cfg_lags = lags or load_config()["lags"]
    out = df.sort_values([*group_cols, "timestamp"]).copy()
    grouped = out.groupby(list(group_cols), sort=False)[value_col]
    for lag in cfg_lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out
```

### Feature sets and lag facts

Exact feature lists, in order:

```text
A = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity"]
B = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
     "hour", "day", "weekday", "is_weekend", "month"]
C = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
     "hour", "day", "weekday", "is_weekend", "month",
     "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester"]
D = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_24", "lag_168", "total_capacity",
     "hour", "day", "weekday", "is_weekend", "month",
     "semester_week", "is_exam_period", "is_holiday", "is_academic_event", "is_in_semester",
     "computer_usage", "study_room_usage", "book_demand", "charging_station_usage",
     "library_code", "zone_code"]
```

`add_lag_features` sorts by group columns plus timestamp, groups by `(library_id, zone_id)`, and applies `shift(lag)` to `occupied_seats`. `prepare_frame` drops rows missing **any** configured lag. With current maximum lag 168, 168 initial rows per series are omitted (1,512 total across nine series in a continuous dataset).

The serialized model metadata was actually read from `models/meta.joblib`. Exact expected regressor columns, in order:

```text
lag_1
lag_2
lag_3
lag_6
lag_24
lag_168
total_capacity
hour
day
weekday
is_weekend
month
semester_week
is_exam_period
is_holiday
is_academic_event
is_in_semester
computer_usage
study_room_usage
book_demand
charging_station_usage
library_code
zone_code
```

The metadata additionally contains `best_regressor: "xgboost_tuned"`, `target: "occupied_seats"`, `levels: ["Low", "Moderate", "High", "Critical"]`, and integer-to-level mapping `{0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}`.

### Prediction, artifacts, and recommendation

Exact signatures:

```python title="src/pipeline/predict.py"
def artifact_dir(cfg: dict | None = None) -> Path: ...
def load_artifacts(cfg: dict | None = None) -> dict: ...
def predict_occupancy(
    df: pd.DataFrame,
    artifacts: dict | None = None,
    feature_set: str = "D",
) -> pd.DataFrame: ...
```

`load_artifacts` loads `regressor.joblib`, `classifier.joblib`, and `meta.joblib` when each exists. `predict_occupancy` runs `prepare_frame`, uses metadata `feature_cols` if present (otherwise the selected feature set), predicts occupied seats and utilization, and predicts classifier level if a classifier artifact exists. It filters feature columns to those that exist in prepared input; it does not itself assert all expected columns are available.

```python title="src/pipeline/recommend.py"
def recommend_alternatives(
    snapshot: pd.DataFrame,
    library_id: str,
    zone_id: str,
    cfg: dict | None = None,
) -> list[dict]: ...
```

`recommend_alternatives` returns `[]` if the requested location is absent; if its predicted utilization is below configured `near_capacity` (`0.85`), it returns `needed: false`. Otherwise it computes remaining seats as `total_capacity * (1 - predicted_utilization)`, ranks same-library rows first and then remaining capacity descending, and returns up to `top_k` (5) alternatives. `prefer_same_library` is configured but the function's behavior is implemented directly by sorting `same_library` descending.

### Model constructors and training

Exact model factory signatures:

```python title="src/models/regressors.py"
def make_linear(random_state: int = 42) -> Pipeline: ...
def make_random_forest(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> RandomForestRegressor: ...
def make_xgboost(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> XGBRegressor: ...
```

`make_linear` is `StandardScaler` then `LinearRegression`. Random forest uses `min_samples_leaf=4`, `n_jobs=-1`, and the supplied seed. XGBoost regressor uses learning rate `0.08`, subsample and column sample `0.8`, objective `"reg:squarederror"`, `n_jobs=-1`, and `tree_method="hist"`.

```python title="src/models/classifier.py"
LEVELS = ["Low", "Moderate", "High", "Critical"]

def make_peak_classifier(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> XGBClassifier: ...
def make_rf_classifier(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> RandomForestClassifier: ...
def encode_levels(series) -> tuple: ...
```

The peak classifier uses XGBoost `objective="multi:softprob"` and `num_class=4`; `encode_levels` maps names according to `LEVELS`. RF classifier factory exists but is not used in `train`.

```python title="src/models/baselines.py — exact signatures"
def historical_mean(train: pd.DataFrame, test: pd.DataFrame, target: str = "occupied_seats") -> np.ndarray: ...
def previous_interval(test: pd.DataFrame) -> np.ndarray: ...
def same_time_previous_day(test: pd.DataFrame) -> np.ndarray: ...
def same_time_previous_week(test: pd.DataFrame) -> np.ndarray: ...
def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str = "occupied_seats") -> dict[str, np.ndarray]: ...
```

`historical_mean` groups training targets by `(library_id, zone_id, hour)` and uses the global training mean when the key does not occur. The other three return the matching test lag column (`lag_1`, `lag_24`, `lag_168`).

```python title="src/models/sarima.py — exact signatures"
def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)): ...
def forecast_sarima(fitted, steps: int) -> np.ndarray: ...
```

`fit_sarima` fits `SARIMAX(series.astype(float), ...)` with `enforce_stationarity=False` and `enforce_invertibility=False`; `forecast_sarima` returns `np.asarray(fitted.get_forecast(steps=steps).predicted_mean)`. This optional model is **NOT called** in the default training path.

`train(cfg: dict | None = None) -> dict` is in `src/pipeline/train.py`. It loads/cleans raw CSV, writes processed CSV, builds features, does chronological 70/15/15 splits, fits linear/RF/XGBoost models, evaluates baselines and ablation A–D, tunes XGBoost using randomized search, trains a peak classifier, computes error summaries and SHAP, then writes joblib and JSON artifacts. Tune search: `n_estimators=[80,120,160]`, `max_depth=[4,6,8]`, `learning_rate=[0.05,0.08,0.12]`, `subsample=[0.7,0.85]`, `colsample_bytree=[0.7,0.9]`; `n_iter=6`; `TimeSeriesSplit(n_splits=3)`; scoring `neg_mean_absolute_error`; `n_jobs=-1`.

Training wall-clock duration: **NOT MEASURED**. The installed environment lacks XGBoost and SHAP, so training could not be run. Prepared feature frame: 116,865 rows, 57,145,649 bytes / 54.498 MiB deep pandas memory. Peak training memory is **NOT MEASURED**.

### Current saved metrics

From `models/metrics.json`:

```json
{
  "baselines": {
    "historical_mean": {"mae": 11.992138748444297, "rmse": 16.787439723446393, "r2": 0.5420008092854991, "mape": 177.50567788046513},
    "previous_interval": {"mae": 6.855284923849181, "rmse": 9.748845383426962, "r2": 0.8455449390916305, "mape": 107.84192181360501},
    "same_time_prev_day": {"mae": 9.410586960241858, "rmse": 14.484138126190683, "r2": 0.6590575375592467, "mape": 129.55600641919213},
    "same_time_prev_week": {"mae": 9.72089441560664, "rmse": 15.275368700610558, "r2": 0.6207905241506191, "mape": 145.4446824473274}
  },
  "regression": {
    "linear": {"mae": 3.3732024741243234, "rmse": 4.606158048409945, "r2": 0.9655195083957736, "mape": 64.91054894101642},
    "random_forest": {"mae": 3.823870485341323, "rmse": 5.207116906117551, "r2": 0.955935338889329, "mape": 69.85768229424774},
    "xgboost": {"mae": 2.6289622657052867, "rmse": 3.7963045466768506, "r2": 0.97657832943709, "mape": 47.53886484047721},
    "xgboost_tuned": {"mae": 2.5513211786895593, "rmse": 3.717654910265699, "r2": 0.977538749767335, "mape": 45.77456154967272}
  },
  "best_regressor": "xgboost_tuned",
  "classification": {
    "precision_macro": 0.8556376515550328,
    "recall_macro": 0.8234814408134361,
    "f1_macro": 0.8365424894818387,
    "precision_weighted": 0.9470916010019981,
    "recall_weighted": 0.9470651987907136,
    "f1_weighted": 0.94671764005472
  },
  "confusion_matrix": [[12687, 174, 0, 0], [232, 2611, 238, 0], [10, 138, 1165, 43], [6, 7, 80, 140]],
  "levels": ["Low", "Moderate", "High", "Critical"],
  "n_train": 81805,
  "n_val": 17529,
  "n_test": 17531,
  "split_dates": {
    "train_end": "2025-08-21 17:00:00",
    "val_end": "2025-11-10 21:00:00",
    "test_end": "2026-01-31 00:00:00"
  }
}
```

From `models/ablation.json` (each `features` list is precisely the corresponding A–D list above):

| Set | `n_features` | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| A | 7 | 5.444288157415801 | 7.6271606606430025 | 0.9054587005810519 | 109.01485706421752 |
| B | 12 | 4.649948102828832 | 6.538234374252594 | 0.930526888982176 | 92.26502399285928 |
| C | 17 | 4.4345195410512535 | 6.228810763057147 | 0.9369469568948522 | 85.058701379003 |
| D | 23 | 2.6289622657052867 | 3.7963045466768506 | 0.97657832943709 | 47.53886484047721 |

From `models/best_params.json`:

```json
{
  "best_params": {
    "subsample": 0.85,
    "n_estimators": 160,
    "max_depth": 6,
    "learning_rate": 0.08,
    "colsample_bytree": 0.9
  },
  "cv_mae": 2.584967931111654
}
```

## 5. API

### Full source: `app/api/main.py`

```python
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.demo import router as demo_router
from src.calendar.store import admin_token, load_calendar, save_calendar
from src.config import load_config, project_path
from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.features.transformers import FEATURE_SETS, occupancy_level, prepare_frame
from src.pipeline.explain import shap_instance
from src.pipeline.predict import load_artifacts
from src.pipeline.recommend import recommend_alternatives

app = FastAPI(title="Smart Library Intelligence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    return prepare_frame(clean_occupancy(load_raw()))


def _latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame["timestamp"].max()
    return frame[frame["timestamp"] == ts].copy()


app.include_router(demo_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/libraries")
def libraries():
    cfg = load_config()
    return {"libraries": cfg["libraries"]}


@app.get("/analytics")
def analytics():
    frame = _frame()
    by_hour = frame.groupby("hour")["occupied_seats"].mean().round(2)
    by_wd = frame.groupby("weekday")["occupied_seats"].mean().round(2)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    exam = frame.groupby("is_exam_period")["occupied_seats"].mean()
    daily = (
        frame.assign(date=frame["timestamp"].dt.date)
        .groupby("date")["occupied_seats"]
        .mean()
        .tail(60)
    )
    return {
        "by_hour": {str(k): float(v) for k, v in by_hour.items()},
        "by_weekday": {names[int(k)]: float(v) for k, v in by_wd.items()},
        "exam_vs_normal": {
            "exam": float(exam.get(1, 0)),
            "normal": float(exam.get(0, 0)),
        },
        "daily": [{"date": str(k), "occupied_seats": float(v)} for k, v in daily.items()],
    }


@app.get("/ablation")
def ablation():
    cfg = load_config()
    path = project_path(cfg["paths"]["ablation_path"])
    if not path.exists():
        raise HTTPException(503, "Train first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/shap")
def shap_summary():
    cfg = load_config()
    path = project_path(cfg["paths"]["shap_path"])
    if not path.exists():
        raise HTTPException(503, "Train first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/status")
def current_status():
    frame = _frame()
    snap = _latest_snapshot(frame)
    artifacts = load_artifacts()
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    if "regressor" in artifacts:
        snap["predicted_occupied"] = artifacts["regressor"].predict(snap[cols])
        snap["predicted_utilization"] = snap["predicted_occupied"] / snap["total_capacity"]
    else:
        snap["predicted_occupied"] = snap["occupied_seats"]
        snap["predicted_utilization"] = snap["occupied_seats"] / snap["total_capacity"]
    records = []
    for _, row in snap.iterrows():
        util = float(row["predicted_utilization"])
        records.append(
            {
                "timestamp": str(row["timestamp"]),
                "library_id": row["library_id"],
                "library_name": row.get("library_name", row["library_id"]),
                "zone_id": row["zone_id"],
                "zone_name": row.get("zone_name", row["zone_id"]),
                "total_capacity": int(row["total_capacity"]),
                "occupied_seats": int(row["occupied_seats"]),
                "available_seats": int(row["available_seats"]),
                "predicted_occupied": float(row["predicted_occupied"]),
                "predicted_utilization": util,
                "occupancy_level": occupancy_level(pd.Series([util])).iloc[0],
            }
        )
    return {"as_of": str(snap["timestamp"].iloc[0]), "locations": records}


def _recursive_forecast(horizon: int, library_id: str | None, zone_id: str | None) -> list[dict]:
    artifacts = load_artifacts()
    if "regressor" not in artifacts:
        raise HTTPException(503, "Model artifacts missing. Run python -m src.pipeline.train")
    frame = _frame()
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    model = artifacts["regressor"]
    cfg = load_config()
    lags = cfg["lags"]

    subset = frame
    if library_id:
        subset = subset[subset["library_id"] == library_id]
    if zone_id:
        subset = subset[subset["zone_id"] == zone_id]
    if subset.empty:
        raise HTTPException(404, "Unknown library/zone")

    out = []
    for (lib, zone), grp in subset.groupby(["library_id", "zone_id"]):
        grp = grp.sort_values("timestamp")
        last = grp.iloc[-1].copy()
        history = grp["occupied_seats"].tolist()
        for step in range(1, horizon + 1):
            feat = last[cols].copy()
            for lag in lags:
                feat[f"lag_{lag}"] = history[-lag] if len(history) >= lag else history[-1]
            pred = float(model.predict(pd.DataFrame([feat]))[0])
            pred = max(0.0, min(pred, float(last["total_capacity"])))
            history.append(pred)
            ts = pd.Timestamp(last["timestamp"]) + pd.Timedelta(hours=step)
            out.append(
                {
                    "timestamp": str(ts),
                    "horizon_h": step,
                    "library_id": lib,
                    "zone_id": zone,
                    "library_name": last.get("library_name", lib),
                    "zone_name": last.get("zone_name", zone),
                    "predicted_occupied": pred,
                    "predicted_utilization": pred / float(last["total_capacity"]),
                    "total_capacity": int(last["total_capacity"]),
                }
            )
    return out


@app.get("/forecast")
def forecast(
    horizon: int = Query(1, description="Hours ahead: 1, 3, or 6"),
    library_id: str | None = None,
    zone_id: str | None = None,
):
    if horizon not in (1, 3, 6, 12, 24):
        raise HTTPException(400, "horizon must be 1, 3, 6, 12, or 24")
    return {"horizon": horizon, "forecasts": _recursive_forecast(horizon, library_id, zone_id)}


@app.get("/explain")
def explain(library_id: str, zone_id: str):
    artifacts = load_artifacts()
    if "regressor" not in artifacts:
        raise HTTPException(503, "Model artifacts missing")
    frame = _frame()
    snap = _latest_snapshot(frame)
    row = snap[(snap["library_id"] == library_id) & (snap["zone_id"] == zone_id)]
    if row.empty:
        raise HTTPException(404, "Location not found in latest snapshot")
    cols = artifacts.get("meta", {}).get("feature_cols", FEATURE_SETS["D"])
    model = artifacts["regressor"]
    return shap_instance(model, row[cols])


@app.get("/recommend")
def recommend(library_id: str, zone_id: str, horizon: int = 1):
    forecasts = _recursive_forecast(horizon, None, None)
    fc = pd.DataFrame(forecasts)
    fc = fc[fc["horizon_h"] == horizon]
    status = current_status()
    snap = pd.DataFrame(status["locations"])
    merged = snap.merge(
        fc[["library_id", "zone_id", "predicted_occupied", "predicted_utilization"]],
        on=["library_id", "zone_id"],
        how="left",
        suffixes=("", "_fc"),
    )
    merged["predicted_utilization"] = merged["predicted_utilization_fc"].fillna(merged["predicted_utilization"])
    recs = recommend_alternatives(merged, library_id, zone_id)
    return {"library_id": library_id, "zone_id": zone_id, "horizon": horizon, "recommendations": recs}


def _require_admin(x_admin_token: str | None) -> None:
    expected = admin_token()
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "Admin token required")


@app.get("/calendar")
def get_calendar():
    return load_calendar()


@app.put("/calendar")
def put_calendar(payload: dict, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    try:
        saved = save_calendar(payload, updated_by="api")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _frame.cache_clear()
    return saved


@app.get("/metrics")
def metrics():
    cfg = load_config()
    path = project_path(cfg["paths"]["metrics_path"])
    if not path.exists():
        raise HTTPException(503, "Train first to produce metrics")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


WEB_DIR = ROOT / "app" / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
```

### Full source: `app/api/demo.py`

```python
"""Demo video upload + occupancy visualization routes."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import load_config, project_path
from src.vision.detect import process_video

router = APIRouter(prefix="/demo", tags=["demo"])

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_BYTES = 40 * 1024 * 1024


def _jobs_dir() -> Path:
    path = project_path("data", "uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _capacity(library_id: str, zone_id: str) -> tuple[int, str, str]:
    cfg = load_config()
    for lib in cfg["libraries"]:
        if lib["id"] != library_id:
            continue
        for zone in lib["zones"]:
            if zone["id"] == zone_id:
                return int(zone["capacity"]), lib["name"], zone["name"]
    raise HTTPException(404, "Unknown library or zone")


def _run_job(job_id: str, src: Path, library_id: str, zone_id: str) -> None:
    dest_dir = _jobs_dir() / job_id
    try:
        capacity, lib_name, zone_name = _capacity(library_id, zone_id)
        summary = process_video(
            src,
            dest_dir / "annotated.mp4",
            capacity=capacity,
            preview_dir=dest_dir / "frames",
        )
        summary.update(
            {
                "library_id": library_id,
                "zone_id": zone_id,
                "library_name": lib_name,
                "zone_name": zone_name,
            }
        )
        (dest_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with LOCK:
            JOBS[job_id].update({"status": "done", "summary": summary})
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update({"status": "error", "error": str(exc)})


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    library_id: str = Form(...),
    zone_id: str = Form(...),
):
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported format. Use {', '.join(sorted(ALLOWED))}")
    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(400, "File too large (40 MB max for this demo)")
    if not payload:
        raise HTTPException(400, "Empty file")
    _capacity(library_id, zone_id)
    job_id = uuid.uuid4().hex[:12]
    dest_dir = _jobs_dir() / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = dest_dir / f"source{suffix}"
    src.write_bytes(payload)
    with LOCK:
        JOBS[job_id] = {"status": "running", "summary": None, "error": None}
    thread = threading.Thread(target=_run_job, args=(job_id, src, library_id, zone_id), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        summary_path = _jobs_dir() / job_id / "summary.json"
        if summary_path.exists():
            return {"job_id": job_id, "status": "done", "summary": json.loads(summary_path.read_text(encoding="utf-8"))}
        raise HTTPException(404, "Unknown job")
    return {"job_id": job_id, **job}


@router.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    path = _jobs_dir() / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(path, media_type="video/mp4", filename="library-detection-demo.mp4")


@router.get("/jobs/{job_id}/frames/{name}")
def job_frame(job_id: str, name: str):
    path = (_jobs_dir() / job_id / "frames" / Path(name).name)
    if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(404, "Frame not found")
    return FileResponse(path)
```

### Routes, parameters, response examples

Examples below show response shapes directly derived from route return code. Values represented by `<...>` are explanatory placeholders, not literal response values. Error responses are FastAPI `HTTPException` status/detail values stated in route code.

| Method and path | Query/body/header parameters | Example success response |
|---|---|---|
| `GET /health` | None | `{"status":"ok"}` |
| `GET /libraries` | None | `{"libraries":[{"id":"lib_central","name":"Central Library","zones":[{"id":"z_quiet","name":"Quiet Study","capacity":140}, ...]}, ...]}` |
| `GET /analytics` | None | `{"by_hour":{"0":<mean>,...},"by_weekday":{"Mon":<mean>,...},"exam_vs_normal":{"exam":<mean>,"normal":<mean>},"daily":[{"date":"<date>","occupied_seats":<mean>}]}` |
| `GET /ablation` | None | JSON contents of `models/ablation.json`; `503 {"detail":"Train first"}` if file absent |
| `GET /shap` | None | JSON contents of `models/shap_summary.json`; `503 {"detail":"Train first"}` if file absent |
| `GET /status` | None | `{"as_of":"<latest timestamp>","locations":[{"timestamp":"<timestamp>","library_id":"<id>","library_name":"<name>","zone_id":"<id>","zone_name":"<name>","total_capacity":<int>,"occupied_seats":<int>,"available_seats":<int>,"predicted_occupied":<number>,"predicted_utilization":<number>,"occupancy_level":"<level>"}]}` |
| `GET /forecast` | Query: `horizon` defaults to 1; allowed 1, 3, 6, 12, 24. Optional `library_id`, `zone_id`. | `{"horizon":1,"forecasts":[{"timestamp":"<timestamp>","horizon_h":1,"library_id":"<id>","zone_id":"<id>","library_name":"<name>","zone_name":"<name>","predicted_occupied":<number>,"predicted_utilization":<number>,"total_capacity":<int>}]}` |
| `GET /explain` | Required query: `library_id`, `zone_id` | `{"base_value":<number>,"contributions":{"<feature>":<number>},"prediction":<number>}`; `503` if regressor missing; `404` if location absent in latest snapshot |
| `GET /recommend` | Required query: `library_id`, `zone_id`; `horizon` defaults to 1 | `{"library_id":"<id>","zone_id":"<id>","horizon":1,"recommendations":[{"needed":false,"reason":"Location is below the near-capacity threshold.","utilization":<number>,"alternatives":[]}]}` or needed=true with alternative records |
| `GET /calendar` | None | Entire normalized calendar JSON |
| `PUT /calendar` | JSON object body `payload`; optional header `X-Admin-Token` required in practice | Normalized saved calendar JSON; `401 {"detail":"Admin token required"}` or `400 {"detail":"<calendar validation error>"}` |
| `GET /metrics` | None | JSON contents of `models/metrics.json`; `503 {"detail":"Train first to produce metrics"}` if absent |
| `POST /demo/video` | Multipart form fields: `file`, `library_id`, `zone_id` | `{"job_id":"<12 hex chars>","status":"running"}` |
| `GET /demo/jobs/{job_id}` | Path `job_id` | `{"job_id":"<id>","status":"running","summary":null,"error":null}`; done/error variants contain summary/error |
| `GET /demo/jobs/{job_id}/video` | Path `job_id` | MP4 file response; `404 {"detail":"Annotated video not ready"}` before ready |
| `GET /demo/jobs/{job_id}/frames/{name}` | Path `job_id`, `name` | Image file response; `404 {"detail":"Frame not found"}` if missing or disallowed extension |

`POST /demo/video` rejects unsupported extensions with 400; supported set is `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`. Payload is read fully and limited to `40 * 1024 * 1024` bytes (40 MiB). Empty file is rejected. Unknown library/zone returns 404.

### Cache, latest snapshot, and recursive forecast

- `_frame()` has `@lru_cache(maxsize=1)` and lazily computes `prepare_frame(clean_occupancy(load_raw()))`. It remains in process memory until the cache is cleared or process exits. The only explicit invalidation in current API is `_frame.cache_clear()` after successful `PUT /calendar`. Appending/updating raw data does not clear it.
- `_latest_snapshot(frame)` finds `frame["timestamp"].max()` and returns a copied subset at that timestamp. Thus it represents the final timestamp present in prepared data, not a wall-clock “now”.
- `_recursive_forecast(horizon, library_id, zone_id)` requires the regressor artifact, starts from cached `_frame`, applies optional ID filters and errors 404 for an empty subset. For each `(library_id, zone_id)` group it sorts by timestamp, takes the last prepared row, makes a list of historical `occupied_seats`, then predicts each future step. Each step starts from `last[cols]`; configured lag columns are overwritten with values from the evolving history (`history[-lag]` or last value if history too short). Predicted occupied seats are clamped to `[0, total_capacity]`, appended to history, and timestamp is calculated as `last.timestamp + step hours`. It does not update other feature fields such as hour/calendar features between recursive steps; `last` is not advanced.

### CORS and authentication

- CORS allows all origins, methods, and headers (`"*"`).
- No auth dependency is applied to the occupancy/forecast/analytics routes.
- Calendar writes call `_require_admin`; `admin_token()` checks environment variable `LIBRARY_ADMIN_TOKEN` first, then config `admin.token`. `admin` is **NOT PRESENT** in the checked `config.yaml`. If neither environment nor config supplies token, calendar writes always reject. Token comparison uses `secrets.compare_digest`.
- Static UI is mounted at `/` from `app/web` if directory exists.

## 6. Web UI

### Full `app/web` file tree

```text
app/web/
├── css/
│   └── app.css
├── index.html
└── js/
    └── app.js
```

### Full source: `app/web/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stacks — Library Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;500;600&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/css/app.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
</head>
<body>
  <aside class="rail">
    <div class="brand">
      <span class="mark">▣</span>
      <div>
        <strong>Stacks</strong>
        <small>Library intelligence</small>
      </div>
    </div>
    <nav id="nav">
      <button data-page="overview" class="active">Overview</button>
      <button data-page="forecast">Forecast</button>
      <button data-page="analytics">Analytics</button>
      <button data-page="resources">Resources</button>
      <button data-page="explain">Explainability</button>
      <button data-page="performance">Model performance</button>
      <button data-page="calendar">Academic calendar</button>
      <button data-page="video">Video detection</button>
    </nav>
    <p class="rail-note">Demo UI · counts, not identities</p>
  </aside>
  <main>
    <header class="top">
      <div>
        <p class="kicker" id="kicker">Campus occupancy</p>
        <h1 id="title">Overview</h1>
      </div>
      <p class="as-of" id="asOf"></p>
    </header>
    <section id="view" class="view"></section>
  </main>
  <script src="js/app.js"></script>
</body>
</html>
```

### Full source: `app/web/js/app.js`

```javascript
const view = document.getElementById("view");
const title = document.getElementById("title");
const kicker = document.getElementById("kicker");
const asOf = document.getElementById("asOf");
let charts = [];

const titles = {
  overview: ["Campus occupancy", "Overview"],
  forecast: ["Demand ahead", "Forecast"],
  analytics: ["Patterns", "Analytics"],
  resources: ["When a zone fills", "Resources"],
  explain: ["Why the model moved", "Explainability"],
  performance: ["Holdout scores", "Model performance"],
  calendar: ["Term dates", "Academic calendar"],
  video: ["Demonstration", "Video detection"],
};

function destroyCharts() {
  charts.forEach((c) => c.destroy());
  charts = [];
}

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}

function metric(label, value) {
  return `<article class="card"><h3>${label}</h3><p>${value}</p></article>`;
}

function table(rows, cols) {
  if (!rows?.length) return "<p class='muted'>No rows.</p>";
  const head = cols.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map((c) => `<td>${c.fmt ? c.fmt(r[c.key]) : r[c.key] ?? ""}</td>`).join("")}</tr>`).join("");
  return `<div class="panel"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function pct(v) { return `${Math.round(Number(v) * 100)}%`; }
function num(v) { return Number(v).toFixed(1); }

function locationSelects(libs, prefix) {
  const first = libs[0];
  const zones = first?.zones || [];
  return `
    <div class="row">
      <div><label>Library</label>
        <select id="${prefix}-lib">${libs.map((l) => `<option value="${l.id}">${l.name}</option>`).join("")}</select>
      </div>
      <div><label>Zone</label>
        <select id="${prefix}-zone">${zones.map((z) => `<option value="${z.id}">${z.name} (${z.capacity})</option>`).join("")}</select>
      </div>
    </div>`;
}

function bindZoneSelect(libs, prefix) {
  const libEl = document.getElementById(`${prefix}-lib`);
  const zoneEl = document.getElementById(`${prefix}-zone`);
  libEl.addEventListener("change", () => {
    const lib = libs.find((l) => l.id === libEl.value);
    zoneEl.innerHTML = (lib?.zones || []).map((z) => `<option value="${z.id}">${z.name} (${z.capacity})</option>`).join("");
  });
}

function barChart(canvas, labels, values, color = "#3d6b54") {
  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: color }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
  });
  charts.push(chart);
}

async function renderOverview() {
  const status = await api("/status");
  asOf.textContent = status.as_of ? `As of ${status.as_of}` : "";
  const loc = status.locations || [];
  const util = loc.reduce((s, r) => s + r.predicted_utilization, 0) / (loc.length || 1);
  view.innerHTML = `
    <div class="metrics">
      ${metric("Locations", loc.length)}
      ${metric("Mean utilization", pct(util))}
      ${metric("Seats occupied", loc.reduce((s, r) => s + r.occupied_seats, 0))}
      ${metric("Seats available", loc.reduce((s, r) => s + r.available_seats, 0))}
    </div>
    ${table(loc, [
      { key: "library_name", label: "Library" },
      { key: "zone_name", label: "Zone" },
      { key: "occupied_seats", label: "Occupied" },
      { key: "available_seats", label: "Available" },
      { key: "predicted_utilization", label: "Predicted util.", fmt: pct },
      { key: "occupancy_level", label: "Level" },
    ])}`;
}

async function renderForecast() {
  const { libraries } = await api("/libraries");
  view.innerHTML = `
    ${locationSelects(libraries, "fc")}
    <div><label>Horizon</label>
      <select id="fc-h"><option>1</option><option selected>3</option><option>6</option></select>
    </div>
    <button class="btn" id="fc-go">Run forecast</button>
    <div id="fc-out"></div>`;
  bindZoneSelect(libraries, "fc");
  document.getElementById("fc-go").onclick = async () => {
    const q = new URLSearchParams({
      horizon: document.getElementById("fc-h").value,
      library_id: document.getElementById("fc-lib").value,
      zone_id: document.getElementById("fc-zone").value,
    });
    const data = await api(`/forecast?${q}`);
    const rows = data.forecasts || [];
    document.getElementById("fc-out").innerHTML = `
      ${table(rows, [
        { key: "timestamp", label: "Time" },
        { key: "horizon_h", label: "Hours" },
        { key: "predicted_occupied", label: "Predicted seats", fmt: num },
        { key: "predicted_utilization", label: "Utilization", fmt: pct },
      ])}
      <div class="panel"><canvas id="fc-chart"></canvas></div>`;
    barChart(document.getElementById("fc-chart"), rows.map((r) => `+${r.horizon_h}h`), rows.map((r) => r.predicted_occupied), "#c9a36a");
  };
}

async function renderAnalytics() {
  const a = await api("/analytics");
  view.innerHTML = `
    <div class="grid2">
      <div class="panel"><h3>By hour</h3><canvas id="c-hour"></canvas></div>
      <div class="panel"><h3>By weekday</h3><canvas id="c-wd"></canvas></div>
      <div class="panel"><h3>Exam vs normal</h3><canvas id="c-ex"></canvas></div>
      <div class="panel"><h3>Recent daily mean</h3><canvas id="c-day"></canvas></div>
    </div>`;
  barChart(document.getElementById("c-hour"), Object.keys(a.by_hour), Object.values(a.by_hour));
  barChart(document.getElementById("c-wd"), Object.keys(a.by_weekday), Object.values(a.by_weekday));
  barChart(document.getElementById("c-ex"), ["Normal", "Exam"], [a.exam_vs_normal.normal, a.exam_vs_normal.exam], "#b85c38");
  const day = new Chart(document.getElementById("c-day"), {
    type: "line",
    data: { labels: a.daily.map((d) => d.date), datasets: [{ data: a.daily.map((d) => d.occupied_seats), borderColor: "#1c332b", tension: 0.2 }] },
    options: { plugins: { legend: { display: false } } },
  });
  charts.push(day);
}

async function renderResources() {
  const { libraries } = await api("/libraries");
  view.innerHTML = `
    <p class="muted">When predicted utilization is near capacity, alternatives are ranked by remaining seats (same library first).</p>
    ${locationSelects(libraries, "rs")}
    <button class="btn" id="rs-go">Recommend</button>
    <div id="rs-out"></div>`;
  bindZoneSelect(libraries, "rs");
  document.getElementById("rs-go").onclick = async () => {
    const q = new URLSearchParams({
      library_id: document.getElementById("rs-lib").value,
      zone_id: document.getElementById("rs-zone").value,
      horizon: "1",
    });
    const data = await api(`/recommend?${q}`);
    const rec = (data.recommendations || [])[0] || {};
    document.getElementById("rs-out").innerHTML = `
      ${metric("Predicted utilization", pct(rec.utilization || 0))}
      <p>${rec.reason || ""}</p>
      ${table(rec.alternatives || [], [
        { key: "library_name", label: "Library" },
        { key: "zone_name", label: "Zone" },
        { key: "remaining_seats", label: "Remaining", fmt: num },
        { key: "predicted_utilization", label: "Util.", fmt: pct },
        { key: "same_library", label: "Same library" },
      ])}`;
  };
}

async function renderExplain() {
  const [{ libraries }, shap] = await Promise.all([api("/libraries"), api("/shap").catch(() => null)]);
  const items = Object.entries(shap?.mean_abs_shap || {});
  view.innerHTML = `
    <p class="muted">Global mean |SHAP| from the trained tree model, plus a per-location breakdown.</p>
    <div class="panel"><canvas id="shap-g"></canvas></div>
    ${locationSelects(libraries, "ex")}
    <button class="btn" id="ex-go">Explain latest snapshot</button>
    <div id="ex-out"></div>`;
  if (items.length) barChart(document.getElementById("shap-g"), items.slice(0, 12).map((x) => x[0]), items.slice(0, 12).map((x) => x[1]), "#c9a36a");
  bindZoneSelect(libraries, "ex");
  document.getElementById("ex-go").onclick = async () => {
    const q = new URLSearchParams({
      library_id: document.getElementById("ex-lib").value,
      zone_id: document.getElementById("ex-zone").value,
    });
    const data = await api(`/explain?${q}`);
    const contrib = Object.entries(data.contributions || {}).slice(0, 12);
    document.getElementById("ex-out").innerHTML = `${metric("Prediction", num(data.prediction))} <div class="panel"><canvas id="shap-l"></canvas></div>`;
    barChart(document.getElementById("shap-l"), contrib.map((x) => x[0]), contrib.map((x) => x[1]));
  };
}

async function renderPerformance() {
  const [metrics, ablation] = await Promise.all([api("/metrics"), api("/ablation").catch(() => null)]);
  const reg = Object.entries(metrics.regression || {}).map(([k, v]) => ({ model: k, ...v }));
  const base = Object.entries(metrics.baselines || {}).map(([k, v]) => ({ model: k, ...v }));
  const ab = Object.entries(ablation || {}).map(([k, v]) => ({ set: k, ...v }));
  view.innerHTML = `
    <p class="muted">Best regressor: <strong>${metrics.best_regressor || "—"}</strong></p>
    <h3>Regression</h3>
    ${table(reg, [{ key: "model", label: "Model" }, { key: "mae", label: "MAE", fmt: num }, { key: "rmse", label: "RMSE", fmt: num }, { key: "r2", label: "R²", fmt: num }])}
    <h3>Baselines</h3>
    ${table(base, [{ key: "model", label: "Baseline" }, { key: "mae", label: "MAE", fmt: num }, { key: "rmse", label: "RMSE", fmt: num }])}
    <h3>Peak classification</h3>
    <pre class="panel">${JSON.stringify(metrics.classification, null, 2)}</pre>
    <h3>Ablation A–D</h3>
    ${table(ab, [{ key: "set", label: "Set" }, { key: "n_features", label: "Features" }, { key: "mae", label: "MAE", fmt: num }, { key: "r2", label: "R²", fmt: num }])}`;
}

async function renderCalendar() {
  const cal = await api("/calendar");
  view.innerHTML = `
    <p class="muted">Anyone can read the published calendar. Saving requires the admin token.</p>
    <div class="panel"><pre>${JSON.stringify(cal.terms, null, 2)}</pre></div>
    <label>Admin token</label><input id="tok" type="password" />
    <label>Full calendar JSON</label>
    <textarea id="cal-json">${JSON.stringify(cal, null, 2)}</textarea>
    <button class="btn" id="cal-save">Save calendar</button>
    <p id="cal-msg" class="muted"></p>`;
  document.getElementById("cal-save").onclick = async () => {
    const msg = document.getElementById("cal-msg");
    try {
      const payload = JSON.parse(document.getElementById("cal-json").value);
      await api("/calendar", {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-Admin-Token": document.getElementById("tok").value },
        body: JSON.stringify(payload),
      });
      msg.textContent = "Saved. Retrain after material date changes.";
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "error";
    }
  };
}

async function renderVideo() {
  const { libraries } = await api("/libraries");
  view.innerHTML = `
    <div class="banner">
      Demonstration and testing only. Upload a short clip you have the right to use (phone video of a study hall is enough).
      A pretrained YOLOv8 person detector draws boxes, a clutter grid, and an estimated empty-seat count from the selected zone capacity.
      This is not live CCTV, not identity tracking, and not a calibrated seat map.
    </div>
    ${locationSelects(libraries, "vd")}
    <div class="drop">
      <p>Upload library video (mp4 / mov / webm, up to 40 MB). The first ~180 sampled frames are processed.</p>
      <input id="vd-file" type="file" accept="video/*" />
    </div>
    <button class="btn" id="vd-go">Detect people</button>
    <p id="vd-msg" class="muted"></p>
    <div id="vd-out"></div>`;
  bindZoneSelect(libraries, "vd");
  document.getElementById("vd-go").onclick = async () => {
    const file = document.getElementById("vd-file").files[0];
    const msg = document.getElementById("vd-msg");
    if (!file) { msg.textContent = "Choose a video first."; return; }
    const body = new FormData();
    body.append("file", file);
    body.append("library_id", document.getElementById("vd-lib").value);
    body.append("zone_id", document.getElementById("vd-zone").value);
    msg.textContent = "Uploading and running person detection… first run may download YOLO weights.";
    try {
      const job = await api("/demo/video", { method: "POST", body });
      await pollJob(job.job_id, msg);
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "error";
    }
  };
}

async function pollJob(id, msg) {
  for (let i = 0; i < 120; i += 1) {
    const job = await api(`/demo/jobs/${id}`);
    if (job.status === "running") {
      msg.textContent = "Detecting people in sampled frames…";
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (job.status === "error") throw new Error(job.error || "Detection failed");
    const s = job.summary;
    const peak = s.peak_occupancy || {};
    const mean = s.mean_occupancy || {};
    document.getElementById("vd-out").innerHTML = `
      <div class="metrics">
        ${metric("Peak people", peak.people_detected ?? s.peak_people)}
        ${metric("Empty seats (est.)", peak.empty_seats_estimate ?? "—")}
        ${metric("Peak clutter", pct(s.peak_clutter_ratio || 0))}
        ${metric("Level", peak.occupancy_level || "—")}
      </div>
      <p class="muted">${s.note || ""} Mean people ${s.mean_people} · ${s.library_name} / ${s.zone_name} (capacity ${mean.capacity}).</p>
      <video controls src="/demo/jobs/${id}/video"></video>
      <div class="film">${(s.previews || []).map((n) => `<img alt="annotated frame" src="/demo/jobs/${id}/frames/${n}" />`).join("")}</div>`;
    msg.textContent = "Done. Red-tinted cells are crowded; green-tinted cells had no person centroid.";
    return;
  }
  throw new Error("Timed out waiting for detection");
}

const pages = {
  overview: renderOverview,
  forecast: renderForecast,
  analytics: renderAnalytics,
  resources: renderResources,
  explain: renderExplain,
  performance: renderPerformance,
  calendar: renderCalendar,
  video: renderVideo,
};

async function show(name) {
  destroyCharts();
  asOf.textContent = "";
  kicker.textContent = titles[name][0];
  title.textContent = titles[name][1];
  document.querySelectorAll(".rail button").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  view.innerHTML = "<p class='muted'>Loading…</p>";
  try {
    await pages[name]();
  } catch (err) {
    view.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

document.getElementById("nav").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-page]");
  if (btn) show(btn.dataset.page);
});

show("overview");
```

### CSS first 80 lines and theme variables

```css title="app/web/css/app.css (lines 1-80)"
:root {
  --ink: #14211c;
  --pine: #1c332b;
  --moss: #3d6b54;
  --brass: #c9a36a;
  --cream: #f3eee4;
  --paper: #fbf8f2;
  --line: rgba(20, 33, 28, 0.12);
  --warn: #b85c38;
}

* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  display: grid;
  grid-template-columns: 240px 1fr;
  font-family: "Source Sans 3", sans-serif;
  background: var(--cream);
  color: var(--ink);
}

.rail {
  background: var(--pine);
  color: var(--cream);
  padding: 1.4rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1.4rem;
}
.brand { display: flex; gap: 0.7rem; align-items: center; }
.mark {
  width: 2.2rem; height: 2.2rem;
  display: grid; place-items: center;
  background: var(--brass); color: var(--ink);
  font-size: 1.1rem;
}
.brand small { display: block; opacity: 0.7; font-size: 0.75rem; }
.rail nav { display: flex; flex-direction: column; gap: 0.25rem; }
.rail button {
  appearance: none; border: 0; background: transparent;
  color: inherit; text-align: left; padding: 0.55rem 0.7rem;
  cursor: pointer; font: inherit; border-radius: 2px;
}
.rail button.active, .rail button:hover { background: rgba(243, 238, 228, 0.1); }
.rail-note { margin-top: auto; font-size: 0.75rem; opacity: 0.55; }

main { padding: 1.6rem 2rem 3rem; overflow: auto; }
.top { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: 1.4rem; }
.kicker { margin: 0; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.72rem; color: var(--moss); }
h1 { font-family: Fraunces, serif; font-weight: 500; margin: 0.15rem 0 0; font-size: 2rem; }
.as-of { color: #5b655f; font-size: 0.9rem; }

.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
.card, .panel {
  background: var(--paper);
  border: 1px solid var(--line);
  padding: 1rem;
}
.card h3 { margin: 0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5b655f; }
.card p { font-family: Fraunces, serif; font-size: 1.7rem; margin: 0.35rem 0 0; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th, td { text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid var(--line); }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #5b655f; }

label { display: block; font-size: 0.82rem; margin: 0.4rem 0 0.25rem; }
select, input, textarea, button.btn {
  font: inherit; padding: 0.45rem 0.6rem;
  border: 1px solid var(--line); background: #fff; color: var(--ink);
}
textarea { width: 100%; min-height: 5rem; }
.btn {
  cursor: pointer; background: var(--ink); color: var(--cream); border: 0;
  margin-top: 0.8rem;
}
.btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
.muted { color: #5b655f; font-size: 0.9rem; }
.error { color: var(--warn); }
.row { display: flex; gap: 0.8rem; flex-wrap: wrap; align-items: end; }
```

Theme: ink `#14211c`, pine `#1c332b`, moss `#3d6b54`, brass `#c9a36a`, cream `#f3eee4`, paper `#fbf8f2`, line `rgba(20, 33, 28, 0.12)`, warning `#b85c38`.

Pages are switched by the sidebar `data-page` buttons and `show(name)`, which destroys old Chart.js instances, updates heading/active button, shows loading, then calls the selected renderer in `pages`. Charts are created with `barChart()` or direct `new Chart(...)`; `destroyCharts()` destroys all tracked charts before each page render. The UI has no periodic dashboard/forecast polling. Video job polling is `pollJob`: up to 120 `GET /demo/jobs/{id}` checks with 1,500 ms delay while status is `running`, then returns on done or throws on error/timeout.

## 7. Calendar

### Full `data/calendar/academic_calendar.json`

```json
{
  "timezone": "Asia/Kolkata",
  "updated_at": null,
  "updated_by": "seed",
  "notes": "Official dates belong here so library admins can replace them each term without a code change. ETE/ETP batch-wise schedules from the Division of Examination can be added as extra exam windows.",
  "terms": [
    {
      "id": "2024-25-autumn-demo",
      "session": "2024-25",
      "name": "Autumn Term (synthetic history)",
      "programmes": "Demo coverage for generated occupancy history",
      "classes_start": "2024-08-26",
      "classes_end": "2024-12-20",
      "next_term_start": "2025-01-13",
      "notes": "Used only until real historical calendars are loaded."
    },
    {
      "id": "2024-25-spring-demo",
      "session": "2024-25",
      "name": "Spring Term (synthetic history)",
      "programmes": "Demo coverage for generated occupancy history",
      "classes_start": "2025-01-13",
      "classes_end": "2025-05-16",
      "next_term_start": "2025-08-26",
      "notes": "Used only until real historical calendars are loaded."
    },
    {
      "id": "2025-26-autumn-demo",
      "session": "2025-26",
      "name": "Autumn Term (synthetic history)",
      "programmes": "Demo coverage for generated occupancy history",
      "classes_start": "2025-08-26",
      "classes_end": "2025-12-19",
      "next_term_start": "2026-01-13",
      "notes": "Used only until real historical calendars are loaded."
    },
    {
      "id": "2026-27-autumn",
      "session": "2026-27",
      "name": "Autumn Term (Term-I)",
      "programmes": "All full-time programmes except B.Pharm, M.Pharm, BPT, MPT, MBA, Education, Physical Education, Agriculture, Law, LSAD, and School of Design (Interior and Product Design)",
      "classes_start": "2026-07-27",
      "classes_end": "2026-12-11",
      "next_term_start": "2027-01-11",
      "notes": "ETE dates are tentative and may vary. Capstone/Dissertation/Project ETEs follow separate guidelines. Examinations may be held on Sunday. Sports/cultural schedules are notified separately."
    },
    {
      "id": "2026-27-spring",
      "session": "2026-27",
      "name": "Spring Term (Term-II)",
      "programmes": "Same audience as Autumn Term-I unless the university publishes a different calendar",
      "classes_start": "2027-01-11",
      "classes_end": null,
      "next_term_start": null,
      "notes": "Classes commence 11 Jan 2027. Add close-of-term, MTT, and ETE dates when the Spring calendar is published."
    }
  ],
  "exam_windows": [
    {"start": "2024-12-09", "end": "2024-12-20", "name": "ETE (demo 2024)", "term_id": "2024-25-autumn-demo"},
    {"start": "2025-05-05", "end": "2025-05-16", "name": "ETE (demo 2025 spring)", "term_id": "2024-25-spring-demo"},
    {"start": "2025-12-08", "end": "2025-12-19", "name": "ETE (demo 2025 autumn)", "term_id": "2025-26-autumn-demo"},
    {"start": "2026-12-14", "end": "2026-12-30", "name": "End Term Examination (ETE)", "term_id": "2026-27-autumn"}
  ],
  "academic_events": [
    {"start": "2024-10-14", "end": "2024-10-18", "name": "Midterms (demo)", "term_id": "2024-25-autumn-demo"},
    {"start": "2025-03-10", "end": "2025-03-14", "name": "Midterms (demo)", "term_id": "2024-25-spring-demo"},
    {"start": "2025-10-13", "end": "2025-10-17", "name": "Midterms (demo)", "term_id": "2025-26-autumn-demo"},
    {"start": "2026-08-29", "end": "2026-08-29", "name": "Last date for Reappear Examination application", "term_id": "2026-27-autumn"},
    {"start": "2026-09-05", "end": "2026-09-05", "name": "Start of Reappear Examination (and onwards; batch-wise dates TBA)", "term_id": "2026-27-autumn"},
    {"start": "2026-10-01", "end": "2026-10-09", "name": "Mid Term Test (MTT)", "term_id": "2026-27-autumn"},
    {"start": "2026-11-14", "end": "2026-11-14", "name": "Advance registration for Spring electives/minors closes", "term_id": "2026-27-autumn"},
    {"start": "2026-12-28", "end": "2026-12-29", "name": "Annual Conference (Share the Vision) for Faculty and Staff", "term_id": "2026-27-autumn"}
  ],
  "closures": [
    {"start": "2024-09-02", "end": "2024-09-02", "name": "Holiday (demo)"},
    {"start": "2024-11-28", "end": "2024-11-29", "name": "Holiday (demo)"},
    {"start": "2024-12-24", "end": "2024-12-25", "name": "Holiday (demo)"},
    {"start": "2025-01-01", "end": "2025-01-01", "name": "Holiday (demo)"},
    {"start": "2025-01-20", "end": "2025-01-20", "name": "Holiday (demo)"},
    {"start": "2025-05-26", "end": "2025-05-26", "name": "Holiday (demo)"},
    {"start": "2025-07-04", "end": "2025-07-04", "name": "Holiday (demo)"},
    {"start": "2025-09-01", "end": "2025-09-01", "name": "Holiday (demo)"},
    {"start": "2025-11-27", "end": "2025-11-28", "name": "Holiday (demo)"},
    {"start": "2025-12-24", "end": "2025-12-25", "name": "Holiday (demo)"},
    {"start": "2026-01-01", "end": "2026-01-01", "name": "Holiday (demo)"},
    {"start": "2026-11-07", "end": "2026-11-10", "name": "Term Break for Students", "term_id": "2026-27-autumn"},
    {"start": "2026-12-13", "end": "2026-12-13", "name": "Preparatory leave for ETE", "term_id": "2026-27-autumn"},
    {"start": "2026-12-31", "end": "2027-01-10", "name": "Winter Vacation for students", "term_id": "2026-27-autumn"}
  ]
}
```

### Full source: `src/calendar/__init__.py`

```python
from src.calendar.store import load_calendar, save_calendar, calendar_path

__all__ = ["load_calendar", "save_calendar", "calendar_path"]
```

### Full source: `src/calendar/store.py`

```python
"""Editable academic calendar stored as JSON (not hardcoded in config)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, project_path

REQUIRED_TOP = ("terms", "exam_windows", "academic_events", "closures")


def calendar_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    rel = cfg.get("paths", {}).get("academic_calendar", "data/calendar/academic_calendar.json")
    return project_path(rel)


def calendar_mtime(path: Path | None = None) -> float:
    p = path or calendar_path()
    return p.stat().st_mtime if p.exists() else 0.0


def empty_calendar() -> dict[str, Any]:
    return {
        "timezone": "Asia/Kolkata",
        "updated_at": None,
        "updated_by": None,
        "notes": "",
        "terms": [],
        "exam_windows": [],
        "academic_events": [],
        "closures": [],
    }


def _as_windows(items: list) -> list[dict]:
    out = []
    for item in items or []:
        if isinstance(item, str):
            out.append({"start": item, "end": item, "name": "Holiday"})
            continue
        start = item.get("start")
        if not start:
            continue
        end = item.get("end") or start
        row = {
            "start": str(start)[:10],
            "end": str(end)[:10],
            "name": item.get("name") or "",
        }
        if item.get("term_id"):
            row["term_id"] = item["term_id"]
        if item.get("notes"):
            row["notes"] = item["notes"]
        out.append(row)
    return out


def normalize_calendar(raw: dict[str, Any]) -> dict[str, Any]:
    data = empty_calendar()
    data.update({k: v for k, v in (raw or {}).items() if k in data or k not in REQUIRED_TOP})
    data["timezone"] = raw.get("timezone") or "Asia/Kolkata"
    data["notes"] = raw.get("notes") or ""
    data["updated_at"] = raw.get("updated_at")
    data["updated_by"] = raw.get("updated_by")
    terms = []
    for t in raw.get("terms") or []:
        if not t.get("id") and not t.get("name"):
            continue
        terms.append(
            {
                "id": t.get("id") or "",
                "session": t.get("session") or "",
                "name": t.get("name") or "",
                "programmes": t.get("programmes") or "",
                "classes_start": str(t["classes_start"])[:10] if t.get("classes_start") else None,
                "classes_end": str(t["classes_end"])[:10] if t.get("classes_end") else None,
                "next_term_start": str(t["next_term_start"])[:10] if t.get("next_term_start") else None,
                "notes": t.get("notes") or "",
            }
        )
    data["terms"] = terms
    data["exam_windows"] = _as_windows(raw.get("exam_windows") or [])
    data["academic_events"] = _as_windows(raw.get("academic_events") or [])
    closures = raw.get("closures")
    if closures is None:
        closures = raw.get("holidays") or []
    data["closures"] = _as_windows(closures)
    return data


def validate_calendar(data: dict[str, Any]) -> list[str]:
    errors = []
    for key in REQUIRED_TOP:
        if key not in data or not isinstance(data[key], list):
            errors.append(f"Missing list: {key}")
    for label, rows in (
        ("exam_windows", data.get("exam_windows") or []),
        ("academic_events", data.get("academic_events") or []),
        ("closures", data.get("closures") or []),
    ):
        for i, row in enumerate(rows):
            try:
                start = pd.Timestamp(row["start"])
                end = pd.Timestamp(row["end"])
            except Exception:
                errors.append(f"{label}[{i}] has an invalid date")
                continue
            if end < start:
                errors.append(f"{label}[{i}] ends before it starts")
    for i, term in enumerate(data.get("terms") or []):
        start, end = term.get("classes_start"), term.get("classes_end")
        if start and end and pd.Timestamp(end) < pd.Timestamp(start):
            errors.append(f"terms[{i}] classes_end is before classes_start")
    return errors


def load_calendar(path: Path | None = None) -> dict[str, Any]:
    p = path or calendar_path()
    if not p.exists():
        return _from_config_fallback()
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return normalize_calendar(raw)


def save_calendar(data: dict[str, Any], path: Path | None = None, updated_by: str = "admin") -> dict[str, Any]:
    normalized = normalize_calendar(data)
    errors = validate_calendar(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    normalized["updated_at"] = datetime.now(timezone.utc).isoformat()
    normalized["updated_by"] = updated_by
    p = path or calendar_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(normalized, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return normalized


def _from_config_fallback() -> dict[str, Any]:
    cfg = load_config()
    data = empty_calendar()
    sem = cfg.get("semester") or {}
    data["exam_windows"] = _as_windows(cfg.get("exam_windows") or [])
    data["academic_events"] = _as_windows(cfg.get("academic_events") or [])
    data["closures"] = _as_windows(cfg.get("holidays") or [])
    if sem:
        data["terms"] = [
            {
                "id": "config-fallback",
                "session": "",
                "name": "Config fallback term",
                "programmes": "",
                "classes_start": None,
                "classes_end": None,
                "next_term_start": None,
                "notes": "Derived from config.yaml because no calendar file was found.",
            }
        ]
    return data


def admin_token(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return os.environ.get("LIBRARY_ADMIN_TOKEN") or str(cfg.get("admin", {}).get("token") or "")


def active_term_for(ts: pd.Timestamp, terms: list[dict] | None = None) -> dict | None:
    day = pd.Timestamp(ts).normalize()
    matched = None
    for term in terms or []:
        start = term.get("classes_start")
        if not start:
            continue
        start_ts = pd.Timestamp(start)
        end_raw = term.get("classes_end")
        end_ts = pd.Timestamp(end_raw) if end_raw else start_ts + pd.Timedelta(days=120)
        if start_ts <= day <= end_ts:
            matched = term
    return matched


def semester_week(ts: pd.Timestamp, calendar: dict | None = None, cfg: dict | None = None) -> int:
    calendar = calendar if calendar is not None else load_calendar()
    term = active_term_for(ts, calendar.get("terms") or [])
    if term and term.get("classes_start"):
        start = pd.Timestamp(term["classes_start"])
        end = pd.Timestamp(term["classes_end"]) if term.get("classes_end") else None
        week = int((pd.Timestamp(ts).normalize() - start).days // 7) + 1
        if end is not None:
            span = max(1, int((end - start).days // 7) + 1)
            return int(max(0, min(week, span + 4)))
        return int(max(0, week))
    cfg = cfg or load_config()
    sem = cfg.get("semester") or {}
    if not sem:
        return 0
    year = pd.Timestamp(ts).year
    month = pd.Timestamp(ts).month
    if month >= sem.get("fall_start_month", 8):
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    elif month <= 6:
        start = pd.Timestamp(year=year, month=sem["spring_start_month"], day=sem["spring_start_day"])
    else:
        start = pd.Timestamp(year=year, month=sem["fall_start_month"], day=sem["fall_start_day"])
    week = int((pd.Timestamp(ts).normalize() - start).days // 7) + 1
    return int(max(0, min(week, sem.get("weeks", 16) + 4)))


def in_semester(ts: pd.Timestamp, calendar: dict | None = None) -> bool:
    calendar = calendar if calendar is not None else load_calendar()
    return active_term_for(ts, calendar.get("terms") or []) is not None


def snapshot() -> dict[str, Any]:
    return deepcopy(load_calendar())
```

### Full source: `src/features/calendar.py`

```python
"""Academic calendar features from the editable JSON calendar store."""

from __future__ import annotations

import pandas as pd

from src.calendar.store import in_semester, load_calendar, semester_week
from src.config import load_config


def _flag_windows(ts: pd.Series, windows: list[dict]) -> pd.Series:
    flag = pd.Series(False, index=ts.index)
    if not windows:
        return flag.astype(int)
    days = ts.dt.normalize()
    for w in windows:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w.get("end") or w["start"])
        flag |= (days >= start) & (days <= end)
    return flag.astype(int)


def add_calendar_features(df: pd.DataFrame, cfg: dict | None = None, ts_col: str = "timestamp") -> pd.DataFrame:
    cfg = cfg or load_config()
    calendar = load_calendar()
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["semester_week"] = ts.map(lambda t: semester_week(t, calendar=calendar, cfg=cfg))
    out["is_exam_period"] = _flag_windows(ts, calendar.get("exam_windows") or [])
    out["is_academic_event"] = _flag_windows(ts, calendar.get("academic_events") or [])
    out["is_holiday"] = _flag_windows(ts, calendar.get("closures") or [])
    out["is_in_semester"] = ts.map(lambda t: int(in_semester(t, calendar=calendar)))
    return out
```

## 8. Resource demand

Present resource columns, raw observed ranges, and missingness:

| Column | Raw range | Missing |
|---|---:|---:|
| `book_demand` | 0–59 | 0 |
| `computer_usage` | 0–41 | 612 |
| `study_room_usage` | 0–22 | 612 |
| `charging_station_usage` | 0–38 | 0 |

All four are included in `FEATURE_SETS["D"]`; they are **features**, not targets. `config.yaml` sets `model.target: occupied_seats`. There is no separate resource-demand model or resource-target prediction endpoint/page. The UI page named “Resources” requests seat recommendations through `/recommend`; it does not forecast the four resource columns.

## 9. Recommender and vision

Full `recommend_alternatives` source is in Section 4 above (`src/pipeline/recommend.py`).

### Vision source signatures and key logic

`src/vision/` contains `__init__.py` and `detect.py`. `src/vision/__init__.py` is empty. The detector is explicitly for uploaded-video demonstration, not a live camera integration.

```python title="src/vision/detect.py — signatures"
def occupancy_from_count(n_people: int, capacity: int) -> dict: ...
def clutter_from_boxes(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> dict: ...
def draw_overlay(frame: np.ndarray, boxes: list, occupancy: dict, clutter: dict) -> np.ndarray: ...
def _yolo(): ...
def detect_people(frame: np.ndarray, conf: float = 0.35) -> list[tuple[float, float, float, float]]: ...
def process_video(
    source: Path,
    dest: Path,
    capacity: int,
    max_frames: int = 180,
    stride: int = 2,
    preview_dir: Path | None = None,
) -> dict: ...
```

- Grid is hardcoded to 4 rows × 6 columns. Box centroids are counted per cell; clutter ratio is occupied cells divided by 24.
- Occupancy clamps capacity to at least 1, person count to at least 0, reports `min(n_people, capacity)` occupied estimate, and derives levels at `<0.4`, `<0.7`, `<0.9`, else Critical.
- `_yolo()` caches one YOLO instance. It looks for `models/yolov8n.pt` at project root, otherwise requests `"yolov8n.pt"` from Ultralytics.
- `detect_people` selects class 0 (person), uses confidence default 0.35, and returns xyxy coordinates.
- `process_video` opens OpenCV capture/writer, samples every `stride` frame until `max_frames` or end, detects/annotates each, saves every eighth retained frame as preview, and returns mean/peak person counts, clutter values, occupancy summaries, previews, and a calibration caveat. `VideoCapture`/writer are released after the loop; there is no `try/finally` around the processing loop in the inspected code.

## 10. Constraints and gotchas

- API history `_frame` is cached once. New CSV rows or external changes remain invisible until restart or explicit cache clear. Only successful `PUT /calendar` clears it currently.
- Current generated history ends `2026-01-31 00:00:00`; `_latest_snapshot` means maximum timestamp in the cached feature frame, not real-time latest sensor status.
- `prepare_frame` drops the first 168 records per library/zone due to `lag_168`; new live series also need a history strategy before inference.
- Recursive forecast uses observed resource columns and calendar/time values copied from the final row. Only lag columns change each step; future hour/day/calendar fields and resource signals are not recalculated. Forecast can therefore use stale contextual features for multi-step horizons.
- `predict_occupancy` filters expected columns down to those present instead of failing early; a missing expected column can result in model-input mismatch.
- `encode_locations` derives categorical codes from categories present in the frame. If a live append introduces new IDs or the subset changes, numeric mappings can change; saved model codes assume training category ordering. Existing library/zone IDs should keep stable explicit mappings for a simulation.
- `write_raw()` writes a newly generated dataset to the configured CSV when invoked without a dataframe; it is not an append operation.
- `train()` retrains against current raw CSV and fixed config. Its split is positional after chronological timestamp sort, not grouped by timestamp; rows from a single timestamp may be split across boundaries.
- Training duration: **NOT MEASURED**. Current Python environment cannot execute training because XGBoost and SHAP are missing. pandas prepared frame is 54.498 MiB deep memory; this is not total process memory.
- Uvicorn cannot start in the inspected environment because it is missing; FastAPI, pytest, XGBoost, and SHAP are missing too. No claim is made about app runtime success in an installed environment.
- `admin.token` config key is **NOT PRESENT**; admin token must come from `LIBRARY_ADMIN_TOKEN` or be added to config.
- CORS is open to all origins/methods/headers; most routes have no authentication.
- Upload uses `await file.read()` before checking 40 MiB, so size enforcement occurs after full payload has been read into memory.
- Resource features are synthetic, correlated to occupied seats, and are used by the occupancy regressor. They are not independently forecast.
- Vision counts people in a video, not calibrated seats; README/UI explicitly frame it as demo only.
- TODO/FIXME search across Python/JS/HTML/CSS/YAML found **no TODO or FIXME comments**. One `pass` is present in `src/data/validate.py` in the empty `ValidationError(ValueError)` class body; it is not a TODO marker.
- Current `models` artifacts exist, but the installed environment cannot load/run them through model dependencies because XGBoost is absent.

## 11. Proposed integration points

These are exact existing seams; no simulation code or `/sim` endpoint is currently present.

| Simulation capability | Existing file/function to connect | Current suitability |
|---|---|---|
| Generate one hourly row per configured zone | `src/data/generate.py:generate_occupancy(cfg: dict \| None = None) -> pd.DataFrame`; reuse `HOUR_PROFILE`, `WEEKDAY_MULT`, `_in_windows`, `active_term_for`, `semester_week` and resource equations | Generator currently generates the entire configured date range with seeded RNG and post-processes whole-frame outliers/missing gaps. It is not a one-step row generator; factor per-timestamp/per-zone formula into a helper for deterministic tick generation rather than calling `generate_occupancy` each tick. |
| Append simulation output to history | API `app/api/main.py:_frame()` and `_latest_snapshot(frame)`; source ingestion helpers `src/data/load.py:load_raw(...)` and `src/features/transformers.py:prepare_frame(...)` | `_frame()` caches a full frame and provides no append/invalidation interface. Establish a simulation-owned history store/frame and update its feature context at each tick; decide persistence (CSV vs separate store). Do not append to raw CSV and assume cached `_frame` sees it. |
| Re-run forecast | `app/api/main.py:_recursive_forecast(horizon, library_id, zone_id)`; model loader `src/pipeline/predict.py:load_artifacts(...)` | Reusable inference logic only if simulation context replaces `_frame()` input and future timestamps/context features are built correctly. Existing recursive method operates on observed history and stale-copies non-lag features. |
| Re-run recommendations | `app/api/main.py:recommend(...)` and `src/pipeline/recommend.py:recommend_alternatives(snapshot, library_id, zone_id, cfg=None)` | Recommendation rules reusable as-is for a snapshot containing current predicted utilization and all zones. API endpoint currently always forecasts from global cached history. |
| Add simulation API | `app/api/main.py` router registration (`app.include_router(demo_router)`), existing function structure for `/forecast`, `/recommend`; new router is **NOT PRESENT** | Add separately scoped `/sim` routes/state machine; define start/pause/step/reset/status and clock semantics explicitly. Ensure simulation state has owner/locking and does not mutate the production cached historical frame accidentally. |
| Add UI controls | `app/web/index.html` sidebar nav; `app/web/js/app.js` `titles`, `pages`, `show(name)` and new render/bind functions; `app/web/css/app.css` theme/layout | Existing SPA page registration and chart lifecycle are reusable. Add navigation button, renderer, API calls, and chart state cleanup. No periodic polling exists except video job polling. |

### Existing functions reusable as-is

- `src/features/temporal.py:add_temporal_features` for timestamps.
- `src/features/calendar.py:add_calendar_features` and `src/calendar/store.py` functions for academic context.
- `src/features/lags.py:add_lag_features` where chronological history is complete and correctly grouped.
- `src/features/transformers.py:prepare_frame` for whole-history preprocessing; it will drop rows without sufficient lag history and encode IDs based on categories in the input.
- `src/pipeline/predict.py:load_artifacts` to load saved estimators and metadata.
- `src/pipeline/recommend.py:recommend_alternatives` for a complete one-time snapshot.
- `app/api/main.py:_latest_snapshot` for maximum timestamp selection on a frame.

`generate_occupancy`, `_frame`, and `_recursive_forecast` should not be treated as live-tick primitives without changing their full-range generation, cached-history, and static-context assumptions respectively.
