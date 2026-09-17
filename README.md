# Smart Library Occupancy & Resource Intelligence

End-to-end occupancy forecasting for academic libraries: synthetic-but-realistic hourly data, a leakage-safe scikit-learn pipeline, peak classification, SHAP explanations, rule-based resource recommendations, and a FastAPI web UI.

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
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8097
```

App: [http://127.0.0.1:8097](http://127.0.0.1:8097)  
API docs: [http://127.0.0.1:8097/docs](http://127.0.0.1:8097/docs)

Swap in a real CSV with the same schema by replacing `data/raw/occupancy.csv` (or the path in `config.yaml`). All downstream steps go through `src.data.load.load_raw()`.

## Pages

1. Overview — latest occupancy and utilization by library/zone  
2. Forecast — 1h / 3h / 6h recursive forecasts  
3. Analytics — hour, weekday, exam vs normal, daily trend  
4. Resources — alternatives when a zone is near capacity  
5. Explainability — global SHAP and per-location contributions  
6. Model performance — baselines, ablation A–D, classification  
7. Academic calendar — publish term/exam/closure dates (token to save)  
8. Video detection — **demo**: upload a short library clip, overlay person boxes, clutter grid, and estimated empty seats  

The video page uses a pretrained YOLOv8 *person* detector for demonstration and testing. It is not live CCTV and does not identify students. Seat counts are estimated from people vs the selected zone capacity, not a calibrated chair map.

Academic dates live in `data/calendar/academic_calendar.json`. Saving uses `PUT /calendar` with `X-Admin-Token`. Set `LIBRARY_ADMIN_TOKEN` (or `admin.token` in `config.yaml`). Retrain after material calendar changes.

## Tests

```bash
pytest -q
```
