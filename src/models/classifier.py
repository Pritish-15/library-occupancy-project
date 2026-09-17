"""Peak occupancy classification (Low / Moderate / High / Critical)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


LEVELS = ["Low", "Moderate", "High", "Critical"]


def make_peak_classifier(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
    )


def make_rf_classifier(n_estimators: int = 120, max_depth: int = 8, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )


def encode_levels(series) -> tuple:
    mapping = {name: i for i, name in enumerate(LEVELS)}
    y = series.map(mapping).astype(int)
    return y, mapping
