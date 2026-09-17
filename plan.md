---
todos:
  - id: scaffold
    status: completed
    content: 'Scaffold repo structure, requirements.txt, LICENSE, .gitignore, config.yaml + src/config.py'
  - id: data-gen
    status: completed
    content: 'Build synthetic dataset generator (src/data/generate.py) with ideal schema, seasonality, exam surges, noise/outliers/missingness; add swappable load.py seam'
  - id: data-eng
    status: completed
    content: Implement validate/clean and temporal/calendar/lag feature modules as sklearn-compatible transformers
  - id: eda
    status: completed
    content: 'Create EDA notebook + plotting helpers (temporal, academic effects, distributions, correlations, missingness, anomalies)'
  - id: baselines
    status: completed
    content: 'Implement baseline predictors (historical mean, previous interval, same-time prev day/week)'
  - id: models-eval
    status: completed
    content: Implement regression + peak-classification models with chronological splits and MAE/RMSE/R2/MAPE + precision/recall/F1 metrics
  - id: ablation
    status: completed
    content: Implement feature ablation experiment (sets A-D) producing the core research comparison
  - id: tuning
    status: completed
    content: Add time-series-aware hyperparameter tuning (RandomizedSearchCV) for the strongest model
  - id: error-shap
    status: completed
    content: Add error analysis breakdowns and SHAP explainability for the best tree model
  - id: recommend
    status: completed
    content: Build constraint-based resource recommendation layer over model outputs
  - id: pipeline
    status: completed
    content: Assemble reproducible sklearn Pipeline + train/predict entrypoints; serialize artifacts to models/
  - id: app
    status: completed
    content: Build FastAPI backend and 6-page Streamlit dashboard; run on an uncommon port
  - id: docs-tests
    status: completed
    content: 'Add docs templates (literature matrix, dataset feasibility, paper outline) and pytest tests'
  - id: run-preview
    status: completed
    content: 'Generate data, train pipeline, launch dashboard, and emit a Preview card'
name: Smart Library Intelligence
overview: 'Build the Smart Library Occupancy & Resource Intelligence System from the roadmap as a runnable, end-to-end Python project: synthetic-but-realistic data, a reproducible scikit-learn pipeline, forecasting + peak-classification models with time-aware evaluation, SHAP explainability, a resource-recommendation layer, and a FastAPI + Streamlit interface.'
isProject: false
---
# Smart Library Occupancy & Resource Intelligence System

## Direction and key decisions

- Stack (per the Project 06 / roadmap spec): Python 3.11, pandas/numpy, scikit-learn, XGBoost, statsmodels (SARIMA), SHAP, FastAPI, Streamlit, joblib, pytest.
- Data strategy: ship a realistic synthetic generator that mimics the roadmap's "ideal dataset" schema, hidden behind a swappable loader so real datasets can replace it later. Everything downstream runs unchanged on real data.
- Scope of this first iteration: build the full engineering spine of the roadmap (Phases 3-16) so the system is demonstrable end-to-end. Academic-writing phases (1, 2, 17) become structured `docs/` templates you fill in.
- Time-series discipline: strictly chronological train/validation/test split, lag features computed without leakage, no random shuffling.

## Repository layout (matches the roadmap)

```
smart-library-intelligence/
  data/{raw,processed}/
  notebooks/            # 01_data_exploration ... 06_explainability
  src/
    data/               # generate.py, load.py, validate.py, clean.py
    features/           # temporal.py, lags.py, calendar.py, transformers.py
    models/             # baselines.py, regressors.py, classifier.py, sarima.py
    evaluation/         # metrics.py, splits.py, error_analysis.py, ablation.py
    pipeline/           # build_pipeline.py, train.py, predict.py, recommend.py, explain.py
    config.py
  app/{api,frontend}/   # FastAPI + Streamlit
  models/               # serialized artifacts (.joblib)
  tests/  docs/
  requirements.txt  README.md  LICENSE  config.yaml
```

## Build phases

### 1. Scaffold + config (Phase 18 / repo hygiene)
- Create the directory tree, `requirements.txt`, `LICENSE` (MIT), `.gitignore`, `config.yaml` (paths, libraries/zones, capacities, exam windows, model params, split ratios), and `src/config.py` to load it.

### 2. Synthetic dataset generator (Phase 2 surrogate)
- `src/data/generate.py`: hourly records across several libraries and zones over ~1.5 academic years. Realistic diurnal + weekly seasonality, semester ramps, exam-period surges, holiday drops, capacity ceilings, gaussian noise, injected outliers and missing gaps. Columns: `timestamp, library_id, zone_id, total_capacity, occupied_seats, available_seats` plus contextual/resource fields (`computer_usage`, `study_room_usage`, `book_demand`, `charging_station_usage`).
- Writes `data/raw/occupancy.csv`. `src/data/load.py` exposes one `load_raw()` seam for swapping in real data.

### 3. Data engineering (Phase 4)
- `validate.py` (schema/range checks) -> `clean.py` (missing-value handling, outlier treatment) -> feature modules: `temporal.py` (hour/day/weekday/weekend/month), `calendar.py` (semester week, exam period, holiday, academic event), `lags.py` (t-1, t-2, t-3, t-6, t-24, t-168, computed per library/zone without leakage).
- `features/transformers.py`: sklearn-compatible transformers so all of this composes in a `Pipeline`/`ColumnTransformer` (Phase 14 requirement).

### 4. EDA (Phase 5)
- `notebooks/02_eda.ipynb` + reusable plotting helpers: occupancy by hour/weekday, weekday vs weekend, semester trends, normal vs exam period, distributions, correlations, missingness and anomaly views.

### 5. Baselines (Phase 6)
- `models/baselines.py`: historical mean, previous interval, same-time-previous-day, same-time-previous-week. These are the bar every ML model must beat.

### 6. Models + time-aware evaluation (Phases 7, 10)
- Regression (Task A): LinearRegression, RandomForest, XGBoost; optional SARIMA in `sarima.py`.
- Classification (Task B): Low/Moderate/High/Critical derived from capacity utilization thresholds (justified in config); train a classifier reporting precision/recall/F1 + confusion matrix.
- `evaluation/splits.py`: chronological train/val/test. `metrics.py`: MAE, RMSE, R2, MAPE (guarded for zeros).

### 7. Ablation experiment (Phase 8)
- `evaluation/ablation.py`: feature sets A (history only) -> B (+temporal) -> C (+academic calendar) -> D (+other contextual). Produces the core research comparison table (how much context adds).

### 8. Hyperparameter tuning (Phase 9)
- `RandomizedSearchCV` with a time-series-aware CV splitter over the strongest model (XGBoost); persist best params.

### 9. Error analysis + explainability (Phases 11, 12)
- `error_analysis.py`: error by hour / weekday / exam vs normal / unusual events.
- `explain.py`: SHAP on the best tree model; global importance + per-prediction contribution breakdown surfaced to the app.

### 10. Resource recommendation (Phase 13)
- `pipeline/recommend.py`: constraint/rule layer over predicted occupancy that ranks alternative libraries/zones/rooms when a location is near capacity (no separate ML model needed initially).

### 11. Reproducible pipeline + training entrypoint (Phase 14)
- `pipeline/build_pipeline.py` assembles preprocessing + model into one serializable `Pipeline`. `train.py` trains and writes `models/*.joblib` + metrics JSON. `predict.py` loads artifacts for inference.

### 12. Application: FastAPI + Streamlit (Phases 15, 16)
- `app/api`: FastAPI endpoints for current status, multi-horizon forecast (1h/3h/6h), explanation, and recommendation, loading the serialized pipeline.
- `app/frontend`: Streamlit dashboard with the 6 pages (Overview, Forecast, Analytics, Resources, Explainability, Model Performance), bound to an uncommon port. This is the previewable surface.
- README documents how to generate data, train, and run both services locally.

### 13. Docs templates + tests (Phases 1, 2, 17, testing)
- `docs/`: literature-review matrix template, dataset-feasibility-report template, research-paper outline (for you to complete).
- `tests/`: pytest coverage for validation, lag correctness (no leakage), metrics, and the recommendation layer.

## Runnable outcome
After implementation I'll generate the synthetic data, train the pipeline, launch the Streamlit dashboard (and FastAPI) on an uncommon port, and emit a Preview card so you can click through the six pages immediately. Real data can later replace the generator via the single `load_raw()` seam.

## Out of scope for this iteration
- Writing actual literature-review/paper prose (templates only).
- Real sensor/university data acquisition and licensing (the loader seam is provided).
- Deep-learning LSTM/GRU models (roadmap marks these as an optional extension).
