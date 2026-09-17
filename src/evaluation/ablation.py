"""Feature-set ablation A (history) -> D (full context)."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from src.evaluation.metrics import regression_metrics
from src.features.transformers import FEATURE_SETS


def run_ablation(model, train: pd.DataFrame, test: pd.DataFrame, target: str = "occupied_seats") -> dict:
    results = {}
    y_train = train[target]
    y_test = test[target]
    for name, cols in FEATURE_SETS.items():
        use = [c for c in cols if c in train.columns]
        est = clone(model)
        est.fit(train[use], y_train)
        pred = est.predict(test[use])
        results[name] = {
            "features": use,
            "n_features": len(use),
            **regression_metrics(y_test, pred),
        }
    return results
