"""SHAP explainability for the best tree model."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def shap_global(model, X: pd.DataFrame, sample_size: int = 250, random_state: int = 42) -> dict:
    if shap is None:
        return {"error": "shap is not installed"}
    rng = np.random.default_rng(random_state)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X.iloc[idx]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[0]
    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=list(sample.columns))
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )
    return {
        "n_samples": int(n),
        "mean_abs_shap": importance,
        "base_value": float(np.ravel(explainer.expected_value)[0]),
    }


def shap_instance(model, X_row: pd.DataFrame, feature_names: list[str] | None = None) -> dict:
    if shap is None:
        return {"error": "shap is not installed"}
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_row)
    if isinstance(values, list):
        values = values[0]
    row_vals = np.ravel(values)[: X_row.shape[1]]
    names = feature_names or list(X_row.columns)
    contrib = (
        pd.Series(row_vals, index=names)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(4)
        .to_dict()
    )
    return {
        "base_value": float(np.ravel(explainer.expected_value)[0]),
        "contributions": contrib,
        "prediction": float(model.predict(X_row)[0]),
    }
