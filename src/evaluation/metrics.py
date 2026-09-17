"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }
