# Research paper outline

## Title (working)

Smart Library Occupancy and Resource Intelligence: Context-Aware Forecasting for Academic Libraries

## Abstract

- Problem: students and staff cannot see which library zones will be crowded.
- Method: chronological ML pipeline on (currently synthetic) hourly occupancy with academic calendar features.
- Result placeholder: ablation A–D, comparison to seasonal baselines, SHAP drivers.
- Contribution: end-to-end reproducible system plus a constraint-based recommendation layer.

## 1. Introduction

- Crowding during exams, unused capacity at other sites.
- Research questions:
  1. Do temporal + academic features beat history-only lags?
  2. Can peak classes (Low–Critical) be recovered with usable F1?
  3. Can predicted occupancy drive simple resource rerouting?

## 2. Related work

- Complete from `docs/literature_review_matrix.md`.

## 3. Data

- Schema, coverage, exam windows, known limitations of the synthetic surrogate.

## 4. Methods

- Cleaning and lag construction without leakage.
- Chronological split.
- Baselines vs linear / RF / XGBoost (+ optional SARIMA).
- Peak classification thresholds (justify from `config.yaml`).
- Ablation A–D and time-series CV tuning.
- SHAP and rule-based recommendations.

## 5. Experiments

- Metrics: MAE, RMSE, R², MAPE; precision / recall / F1.
- Error by hour, weekday, exam vs normal.
- Tables/figures from `models/metrics.json` and `models/ablation.json`.

## 6. Discussion

- When calendar context matters.
- Operational use of the dashboard.
- Ethics: counts not identities.

## 7. Limitations and future work

- Real university data, LSTM/GRU optional extension, live sensors.

## 8. Conclusion

## References
