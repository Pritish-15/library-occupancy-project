# Smart Library Occupancy & Resource Intelligence

End-to-end occupancy forecasting for academic libraries: synthetic-but-realistic hourly data, a leakage-safe scikit-learn pipeline, peak classification, SHAP explanations, rule-based resource recommendations, and a FastAPI + Streamlit interface.

## Setup

Python 3.11 recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate data, train, run

```bash
python -m src.data.generate
python -m src.pipeline.train
uvicorn app.api.main:app --host 127.0.0.1 --port 8097
streamlit run app/frontend/dashboard.py --server.port 8533
```

Dashboard: [http://127.0.0.1:8533](http://127.0.0.1:8533)  
API docs: [http://127.0.0.1:8097/docs](http://127.0.0.1:8097/docs)

Swap in a real CSV with the same schema by replacing `data/raw/occupancy.csv` (or the path in `config.yaml`). All downstream steps go through `src.data.load.load_raw()`.

## Pages

1. Overview — latest occupancy and utilization by library/zone  
2. Forecast — 1h / 3h / 6h recursive forecasts  
3. Analytics — temporal, academic, distribution, correlation, missingness  
4. Resources — alternatives when a zone is near capacity  
5. Explainability — global SHAP and per-location contributions  
6. Model Performance — baselines, ablation A–D, classification, error breakdowns  

## Tests

```bash
pytest -q
```

## Layout

See `plan.md` for the research roadmap. Academic writing lives as templates under `docs/`.
