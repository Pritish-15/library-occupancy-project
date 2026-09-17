import numpy as np

from src.evaluation.metrics import classification_metrics, mape, regression_metrics


def test_regression_metrics_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["r2"] == 1
    assert m["mape"] == 0


def test_mape_guards_zeros():
    y = np.array([0.0, 0.0, 10.0])
    pred = np.array([1.0, 2.0, 10.0])
    value = mape(y, pred, eps=1.0)
    assert np.isfinite(value)
    assert value > 0


def test_classification_metrics_keys():
    y = [0, 1, 2, 3, 0, 1]
    pred = [0, 1, 2, 2, 0, 1]
    m = classification_metrics(y, pred, labels=[0, 1, 2, 3])
    assert "f1_macro" in m
    assert 0 <= m["f1_macro"] <= 1
