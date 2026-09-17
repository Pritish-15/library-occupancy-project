# Smart Library Occupancy & Resource Intelligence System

A campus library intelligence platform: it forecasts **how full each zone will be**, flags **peak occupancy**, estimates **remaining seats**, and (in a demo) overlays **person detection** on an uploaded video so you can see crowding vs empty floor area.

This repository is an end-to-end **research + engineering prototype**. Occupancy today comes from a **synthetic hourly dataset** (or a CSV you drop in). Models are trained **from scratch** on that table. The video page uses a **pretrained YOLOv8 person detector** only for demonstration—not live CCTV and not student identity.

---

## Table of contents

1. [Problem the project solves](#1-problem-the-project-solves)
2. [What works today vs what production still needs](#2-what-works-today-vs-what-production-still-needs)
3. [Technology stack](#3-technology-stack)
4. [Quick start](#4-quick-start)
5. [End-to-end data and request flow](#5-end-to-end-data-and-request-flow)
6. [Repository layout](#6-repository-layout)
7. [Configuration](#7-configuration)
8. [Data schema](#8-data-schema)
9. [Python modules and functions](#9-python-modules-and-functions)
10. [HTTP API](#10-http-api)
11. [Web UI](#11-web-ui)
12. [Notebooks, docs, tests, artifacts](#12-notebooks-docs-tests-artifacts)
13. [Academic calendar](#13-academic-calendar)
14. [How to swap in real college data](#14-how-to-swap-in-real-college-data)
15. [Future requirements](#15-future-requirements)

---

## 1. Problem the project solves

Students waste time hunting for seats at peak hours; some rooms overflow while others sit empty; computers, study rooms, and books also spike (especially around MTT and ETE).

This system is meant to:

| Need | How this repo approaches it |
| --- | --- |
| Occupancy prediction | Hourly seat counts → lag + calendar features → XGBoost/RF/linear vs naive baselines |
| Demand forecasting | Recursive 1h / 3h / 6h forecasts per library/zone |
| Peak-hour detection | Utilization bins Low / Moderate / High / Critical + XGBoost classifier |
| Seat availability | `capacity − occupied` (tabular) and `capacity − people_detected` (video demo) |
| Resource demand | Columns exist (`book_demand`, PCs, rooms, charging) as **features**; they are **not** yet separate forecast targets |
| Alternative recommendation | Rule layer: if a zone is near capacity, rank other zones by remaining seats |
| Occupancy dashboard | FastAPI static UI in `app/web` |
| ML pipeline | `python -m src.pipeline.train` writes `models/*.joblib` |
| Deployment | Local Uvicorn on port **8097** |

---

## 2. What works today vs what production still needs

**Works now**

- Synthetic 1.5-year hourly occupancy for 3 libraries × 3 zones
- Leakage-safe lags, chronological split, ablation A–D, time-series CV, SHAP
- FastAPI UI: overview, forecast, analytics, resources, explainability, metrics, calendar editor, video demo
- Editable academic calendar JSON (2026–27 Autumn Term seeded)

**Not production-ready**

- No live sensors; API caches the **entire** history in memory (`_frame`)
- No authentication on most routes (calendar save uses a shared admin token)
- Resource demand is not its own model
- Video seats are **not** a calibrated chair map
- CORS is open (`*`)

---

## 3. Technology stack

| Layer | Libraries | Role |
| --- | --- | --- |
| Data | pandas, NumPy, PyYAML | Tables, arrays, config |
| ML | scikit-learn, XGBoost, joblib | Models, split, search, serialize |
| Time series (optional) | statsmodels | SARIMA helper (`src/models/sarima.py`); not in the default train loop |
| Explain | SHAP | Tree attributions |
| Vision demo | OpenCV, Ultralytics YOLOv8n | Person boxes on uploaded clips |
| API | FastAPI, Uvicorn, python-multipart | JSON API + file upload |
| UI | HTML / CSS / JS + Chart.js (CDN) | Single-page app in `app/web` |
| EDA notebooks | Jupyter, Plotly, matplotlib, seaborn | Offline exploration |
| Tests | pytest | Validation, lags, metrics, recommend, calendar, vision math |

---

## 4. Quick start

Python 3.11+ recommended (the environment this was built in may be newer).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.generate
python -m src.pipeline.train
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8097
```

- App: [http://127.0.0.1:8097](http://127.0.0.1:8097)
- OpenAPI: [http://127.0.0.1:8097/docs](http://127.0.0.1:8097/docs)

```bash
pytest -q
```

The first video-detection run downloads `yolov8n.pt` (gitignored). Training writes `models/regressor.joblib`, `classifier.joblib`, `meta.joblib`, and metric JSON files.

---

## 5. End-to-end data and request flow

```text
config.yaml + academic_calendar.json
        │
        ▼
src.data.generate  ──writes──►  data/raw/occupancy.csv
        │                         ▲
        │                         │  (or your real CSV)
        ▼                         │
src.data.load.load_raw  ──────────┘
        │
        ▼
validate → clean → prepare_frame
  (temporal, calendar from JSON, lags per library/zone, occupancy_level)
        │
        ▼
chronological_split (70 / 15 / 15 by time, no shuffle)
        │
        ├─► baselines (mean, lag-1, lag-24, lag-168)
        ├─► Linear / RF / XGBoost occupancy regression
        ├─► RandomizedSearchCV + TimeSeriesSplit on XGBoost
        ├─► peak classifier (4 classes)
        ├─► ablation A–D
        ├─► error-by-hour / weekday / exam
        └─► SHAP on the tree model
        │
        ▼
models/*.joblib + models/*.json
        │
        ▼
FastAPI  (app/api/main.py + app/api/demo.py)
        │
        ├─ GET /status, /forecast, /recommend, /explain, /analytics, /metrics…
        ├─ GET/PUT /calendar
        └─ POST /demo/video  →  YOLOv8 overlay  →  annotated mp4 + stills
        │
        ▼
app/web  (index.html + css/app.css + js/app.js)
```

**Inference path (forecast):** latest rows per zone → fill lag features from history → predict next hour → append prediction as the new lag-1 → repeat for 3h/6h.

**Video path (demo):** upload → background thread → sample frames → detect class `person` only → clutter grid from box centroids → empty seats ≈ `zone_capacity − count` → save `data/uploads/<job_id>/`.

---

## 6. Repository layout

```text
library_occupancy/
  config.yaml                 # paths, libraries, capacities, model/split/vision settings
  requirements.txt
  LICENSE                     # MIT
  plan.md                     # original research roadmap (implementation notes)
  pytest.ini
  README.md                   # this file
  data/
    raw/occupancy.csv         # generated or replaced; gitignored
    processed/occupancy.csv   # cleaned copy from train
    calendar/academic_calendar.json
    uploads/                  # demo videos; gitignored
  models/                     # joblib + metrics JSON; large binaries gitignored
  notebooks/                  # 01–06 Jupyter
  docs/                       # literature / feasibility / paper templates
  tests/
  src/                        # library code (import as src.*)
  app/
    api/main.py               # FastAPI app
    api/demo.py               # video jobs
    web/                      # static UI mounted at /
```

---

## 7. Configuration

`src/config.py` loads `config.yaml` from the repo root.

| Key | Purpose |
| --- | --- |
| `project.seed` | RNG seed for the generator |
| `paths.*` | CSV, models, calendar, metric JSON locations |
| `generation.*` | Date range, hourly freq, noise, outlier/missing rates |
| `libraries` | IDs, names, zone capacities used by generator, UI, and video empty-seat math |
| `semester` / `exam_windows` / `holidays` / `academic_events` | **Legacy fallback** if the calendar JSON is missing |
| `utilization_thresholds` | Low &lt; 0.40, Moderate &lt; 0.70, High &lt; 0.90, else Critical |
| `recommendation.near_capacity` | Default 0.85; when to suggest other zones |
| `recommendation.top_k` | How many alternatives to return |
| `splits` | Chronological 0.70 / 0.15 / 0.15 |
| `lags` | `[1, 2, 3, 6, 24, 168]` hours |
| `model.*` | Target `occupied_seats`, tree size, search iters, SHAP sample, CV folds |
| `api.host` / `api.port` | Uvicorn bind (8097) |
| `admin.token` | Calendar PUT; override with env `LIBRARY_ADMIN_TOKEN` |
| `vision.*` | Upload/demo limits (also enforced in `demo.py`) |

`load_config()` is cached. `project_path()` / `data_path()` resolve relative to the repo root. `ensure_dirs()` creates data/models/notebooks/docs/tests/uploads folders.

---

## 8. Data schema

**Required columns** (`src/data/validate.py`):

`timestamp`, `library_id`, `zone_id`, `total_capacity`, `occupied_seats`, `available_seats`

**Optional context:** `library_name`, `zone_name`, `computer_usage`, `study_room_usage`, `book_demand`, `charging_station_usage`

After `prepare_frame`: hour/day/weekday/weekend/month/weekofyear, semester_week, exam/holiday/event flags, lags, `library_code`, `zone_code`, `utilization`, `occupancy_level`. Rows with incomplete lags (first week per zone) are dropped.

---

## 9. Python modules and functions

### `src/config.py`

| Function | Why |
| --- | --- |
| `load_config(path=None)` | Cached YAML load |
| `project_path(*parts)` | Absolute path under repo root |
| `data_path(key)` | Resolve `paths[key]` from config |
| `ensure_dirs()` | Create expected folders |

### `src/data/`

**`generate.py`** — synthetic “ideal dataset.” Diurnal profile, weekday/weekend, semester ramp, exam surge, holiday drop, noise, outliers, missing gaps.

| Function | Why |
| --- | --- |
| `_in_windows` | Timestamp inside a start/end window |
| `generate_occupancy` | Build the full table from config + calendar JSON |
| `write_raw` | Save `data/raw/occupancy.csv` |
| `main` | CLI: `python -m src.data.generate` |

**`load.py`**

| Function | Why |
| --- | --- |
| `load_raw` | **Single swap point** for real data. If the CSV is missing and `generate_if_missing=True`, it generates. |

**`validate.py`**

| Symbol | Why |
| --- | --- |
| `REQUIRED_COLUMNS` / `OPTIONAL_CONTEXT` | Schema contract |
| `ValidationError` | Typed failure |
| `validate_occupancy(df, strict=False)` | Types, positive capacity, non-negative occupancy; strict mode also checks occupied ≤ capacity and occupied + available = capacity |

**`clean.py`**

| Function | Why |
| --- | --- |
| `clean_occupancy` | Interpolate/fill numeric gaps per library/zone, clip occupancy to capacity, recompute `available_seats` |

### `src/calendar/`

JSON is the live calendar. Admins update it without editing Python.

| Function | Why |
| --- | --- |
| `calendar_path` / `calendar_mtime` | File location and cache-busting |
| `empty_calendar` | Canonical empty document |
| `_as_windows` | Normalize date strings or `{start,end,name}` |
| `normalize_calendar` / `validate_calendar` | Shape and date-order checks |
| `load_calendar` | Read JSON, or fall back to YAML windows |
| `save_calendar` | Write JSON, stamp `updated_at` / `updated_by` |
| `admin_token` | Env `LIBRARY_ADMIN_TOKEN` or `config.admin.token` |
| `active_term_for` | Term whose `classes_start`…`classes_end` contains the timestamp |
| `semester_week` | Week index inside that term (YAML semester months if no term matches) |
| `in_semester` | Boolean in-term |
| `snapshot` | Deep copy for callers that must not mutate cache |

### `src/features/`

| Module / symbol | Why |
| --- | --- |
| `temporal.add_temporal_features` | Clock/calendar-of-week features from `timestamp` |
| `calendar.add_calendar_features` | Exam / event / closure flags + semester week from the JSON store |
| `lags.add_lag_features` | `groupby(library_id, zone_id).shift(k)` so lags never mix zones or look ahead |
| `TemporalFeatures`, `CalendarFeatures`, `LagFeatures` | sklearn `TransformerMixin` wrappers for Pipeline composition |
| `FEATURE_SETS` | Ablation A history → B + time → C + academic → D + resources + location codes |
| `encode_locations` | Integer codes for library/zone |
| `occupancy_level` | Utilization → Low/Moderate/High/Critical |
| `prepare_frame` | Full feature table used by train, API, and notebooks |

### `src/models/`

| Function | Why |
| --- | --- |
| `historical_mean` | Train mean occupancy by library, zone, hour of day |
| `previous_interval` | `lag_1` (last hour) |
| `same_time_previous_day` | `lag_24` |
| `same_time_previous_week` | `lag_168` |
| `baseline_predictions` | All four together — the bar ML must beat |
| `make_linear` | StandardScaler + LinearRegression |
| `make_random_forest` / `make_xgboost` | Occupancy regressors |
| `LEVELS` | Class names |
| `make_peak_classifier` / `make_rf_classifier` | 4-way peak models |
| `encode_levels` | Labels → 0..3 |
| `fit_sarima` / `forecast_sarima` | Optional per-series SARIMA; **not** called by default `train()` |

### `src/evaluation/`

| Function | Why |
| --- | --- |
| `chronological_split` | Sort by time; cut train/val/test by configured fractions |
| `mape` | MAPE with `eps=1` so zeros do not explode |
| `regression_metrics` | MAE, RMSE, R², MAPE |
| `classification_metrics` | Macro and weighted P/R/F1 |
| `run_ablation` | Clone XGBoost, fit each feature set A–D, score test MAE |
| `summarize_errors` | MAE by hour, weekday, exam vs normal vs holiday |

### `src/pipeline/`

| Function | Why |
| --- | --- |
| `build_regressor_pipeline` / `build_scaled_pipeline` | Column select ± scaler + estimator (sklearn `Pipeline`) |
| `train` | Full experiment + persist artifacts (`python -m src.pipeline.train`) |
| `_dump` | JSON writer |
| `artifact_dir` / `load_artifacts` | Load joblib bundle |
| `predict_occupancy` | Batch predict occupied seats, utilization, optional peak class |
| `recommend_alternatives` | Constraint ranking of other zones |
| `shap_global` / `shap_instance` | Mean \|SHAP\| and one-row contributions |

### `src/vision/`

Demo only. COCO class 0 (`person`). No faces, no tracking IDs.

| Function | Why |
| --- | --- |
| `occupancy_from_count` | People vs zone capacity → empty-seat estimate and level |
| `clutter_from_boxes` | 4×6 grid; a cell is “cluttered” if a box centroid falls in it |
| `draw_overlay` | HUD + boxes + tinted grid |
| `_yolo` | Cached Ultralytics model (`models/yolov8n.pt` if present) |
| `detect_people` | Boxes at `conf=0.35` |
| `process_video` | Sample frames (`stride`, `max_frames`), write annotated mp4 + JPEG previews + summary |

### `src/viz/plots.py`

Plotly figures for notebooks: hour, weekday, weekend, exam vs normal, histogram, correlation heatmap, missingness, daily trend. The live UI uses Chart.js instead.

### `app/api/demo.py`

| Symbol | Why |
| --- | --- |
| `JOBS` / `LOCK` | In-memory job status |
| `_jobs_dir` | `data/uploads` |
| `_capacity` | Look up zone capacity |
| `_run_job` | Background `process_video` |
| `POST /demo/video` | Multipart upload, 40 MB, video extensions only |
| `GET /demo/jobs/{id}` | Poll status/summary |
| `GET /demo/jobs/{id}/video` | Annotated mp4 |
| `GET /demo/jobs/{id}/frames/{name}` | Preview JPEG |

### `app/api/main.py` helpers

| Function | Why |
| --- | --- |
| `_frame` | Cached cleaned+featurized history for API reads |
| `_latest_snapshot` | Rows at `max(timestamp)` |
| `_recursive_forecast` | Multi-horizon occupancy |
| `_require_admin` | Constant-time token compare for calendar writes |

Static files from `app/web` are **mounted last** so API routes win.

---

## 10. HTTP API

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | no | Liveness |
| GET | `/libraries` | no | Config libraries/zones |
| GET | `/status` | no | Latest occupancy + model scores |
| GET | `/forecast?horizon=&library_id=&zone_id=` | no | 1, 3, 6 (also 12, 24) hour forecast |
| GET | `/recommend?library_id=&zone_id=&horizon=` | no | Alternatives if near capacity |
| GET | `/explain?library_id=&zone_id=` | no | SHAP for latest row |
| GET | `/analytics` | no | Hour/weekday/exam/daily aggregates |
| GET | `/metrics` `/ablation` `/shap` | no | Training JSON |
| GET | `/calendar` | no | Published calendar |
| PUT | `/calendar` | `X-Admin-Token` | Replace calendar; clears `_frame` cache |
| POST | `/demo/video` | no | Start detection job |
| GET | `/demo/jobs/{id}` | no | Job status |
| GET | `/` | no | UI |

---

## 11. Web UI

`app/web/index.html` — shell and nav  
`app/web/css/app.css` — layout  
`app/web/js/app.js` — fetches the API, Chart.js, calendar JSON editor, video FormData + poll

| Page | Behavior |
| --- | --- |
| Overview | `/status` metrics and table |
| Forecast | Library/zone/horizon → `/forecast` |
| Analytics | `/analytics` charts |
| Resources | `/recommend` |
| Explainability | `/shap` + `/explain` |
| Model performance | `/metrics` + `/ablation` |
| Academic calendar | Read JSON; save with admin token |
| Video detection | Upload clip, YOLO overlay, clutter grid, estimated empty seats |

---

## 12. Notebooks, docs, tests, artifacts

**Notebooks:** `01_data_exploration` … `06_explainability` — schema, EDA plots, then metrics/ablation/SHAP after training.

**Docs templates (you fill prose):** `docs/literature_review_matrix.md`, `dataset_feasibility.md`, `paper_outline.md`.

**Tests:** `conftest.py` puts the repo on `sys.path`. Coverage includes validation, lag no-leakage, metrics, recommendations, calendar (including 2026–27 ETE), vision occupancy/clutter/overlay (no YOLO download in unit tests).

**Artifacts after train:** `regressor.joblib`, `classifier.joblib`, `meta.joblib`, `metrics.json`, `ablation.json`, `best_params.json`, `shap_summary.json`, `error_analysis.json`.

---

## 13. Academic calendar

File: `data/calendar/academic_calendar.json`

- `terms` — `classes_start` / `classes_end` / `next_term_start`, programme notes  
- `exam_windows` — e.g. ETE 14–30 Dec 2026  
- `academic_events` — MTT, reappear, registration, conferences  
- `closures` — term break, preparatory leave, winter vacation  

UI **Academic calendar** or `PUT /calendar`. After you change exam or term dates, retrain. Seeded 2026–27 Autumn Term-I; Spring Term-II start is 11 Jan 2027 with remaining dates left for the official circular. Demo historical terms remain so the synthetic 2024–2026 series still has exam flags.

---

## 14. How to swap in real college data

1. Export hourly (or aggregatable) counts with the required columns.  
2. Replace `data/raw/occupancy.csv` (or change `paths.raw_data`).  
3. Put real libraries/zones/capacities in `config.yaml`.  
4. Publish the real academic calendar.  
5. `python -m src.pipeline.train` — do **not** keep synthetic-trained `.joblib` for campus use.  
6. Point ingest at a database later; keep calling `load_raw()` (or replace that function only).

Ask management for: zone maps and capacities, ≥1 year occupancy, exam/holiday/closure calendar, optional PC/room/circulation aggregates, and written approval to use **counts not identities**.

---

## 15. Future requirements

These are the gaps between the official project outcome (“deployed system for students and admins”) and this repo.

### Data and sensors

- Live hourly ingest (gates, Wi-Fi estimates, seat sensors) instead of a static CSV and `_frame` cache  
- Per-zone camera homography / seat polygons if video counts must match real chairs  
- Separate time series and models for **book demand**, **PC usage**, and **study rooms**  
- Handling of closed rooms and exam seating layouts  

### Models

- Retrain cadence (term start / weekly) and model versioning  
- Optional LSTM/GRU (roadmap extension; out of scope here)  
- Use SARIMA in the default train comparison if a univariate baseline is required  
- Recalibrate utilization thresholds on real fill rates  

### Product

- Campus SSO; student vs staff roles; hide `/docs` in production  
- Proper public UI hosting (CDN) rather than API + static on one process  
- Rate limits, TLS, secrets not in `config.yaml`  
- Recommendations that respect accessibility, quiet floors, and opening hours  
- Label all forecasts as estimates; this is not a seat reservation system unless you add one  

### Video demo → operations (only with legal/IT approval)

- Do not silently turn the upload page into 24/7 CCTV  
- If cameras are approved: edge inference, retention limits, no face recognition, counts only  
- Calibrated empty-seat maps, not capacity minus YOLO count  

### Reliability

- Job queue (Redis/RQ) instead of in-memory `JOBS`  
- Health checks on missing hours and occupancy &gt; capacity  
- Load testing for exam week  

### Research write-up

- Complete literature matrix, feasibility report, and paper from the `docs/` templates using **real** holdout metrics, not synthetic ones  

---

## License

MIT. See `LICENSE`.
