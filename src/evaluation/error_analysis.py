"""Error breakdowns by hour, weekday, and exam vs normal periods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics


def _attach_error(frame: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    out = frame.copy()
    out["y_true"] = np.asarray(y_true)
    out["y_pred"] = np.asarray(y_pred)
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    return out


def error_by_hour(frame: pd.DataFrame, y_true, y_pred) -> dict:
    out = _attach_error(frame, y_true, y_pred)
    return out.groupby("hour")["abs_error"].mean().round(3).to_dict()


def error_by_weekday(frame: pd.DataFrame, y_true, y_pred) -> dict:
    out = _attach_error(frame, y_true, y_pred)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grouped = out.groupby("weekday")["abs_error"].mean()
    return {names[int(k)]: round(v, 3) for k, v in grouped.items()}


def error_exam_vs_normal(frame: pd.DataFrame, y_true, y_pred) -> dict:
    out = _attach_error(frame, y_true, y_pred)
    exam = out[out["is_exam_period"] == 1]
    normal = out[out["is_exam_period"] == 0]
    result = {}
    if len(exam):
        result["exam"] = regression_metrics(exam["y_true"], exam["y_pred"])
    if len(normal):
        result["normal"] = regression_metrics(normal["y_true"], normal["y_pred"])
    unusual = out[out["is_holiday"] == 1]
    if len(unusual):
        result["holiday"] = regression_metrics(unusual["y_true"], unusual["y_pred"])
    return result


def summarize_errors(frame: pd.DataFrame, y_true, y_pred) -> dict:
    return {
        "by_hour": error_by_hour(frame, y_true, y_pred),
        "by_weekday": error_by_weekday(frame, y_true, y_pred),
        "by_regime": error_exam_vs_normal(frame, y_true, y_pred),
        "overall": regression_metrics(y_true, y_pred),
    }
