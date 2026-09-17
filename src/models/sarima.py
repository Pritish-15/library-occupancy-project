"""Optional SARIMA baseline on a single library/zone series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)):
    model = SARIMAX(
        series.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarima(fitted, steps: int) -> np.ndarray:
    pred = fitted.get_forecast(steps=steps)
    return np.asarray(pred.predicted_mean)
