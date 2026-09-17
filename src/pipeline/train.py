"""Train regressors, classifier, ablation, tuning, SHAP; serialize artifacts."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import ensure_dirs, load_config, project_path
from src.data.clean import clean_occupancy
from src.data.load import load_raw
from src.evaluation.ablation import run_ablation
from src.evaluation.error_analysis import summarize_errors
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.evaluation.splits import chronological_split
from src.features.transformers import FEATURE_SETS, prepare_frame
from src.models.baselines import baseline_predictions
from src.models.classifier import LEVELS, encode_levels, make_peak_classifier
from src.models.regressors import make_linear, make_random_forest, make_xgboost
from src.pipeline.explain import shap_global


def _dump(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def train(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    raw = load_raw()
    cleaned = clean_occupancy(raw)
    processed_path = project_path(cfg["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)

    frame = prepare_frame(cleaned, cfg)
    train_df, val_df, test_df = chronological_split(frame, cfg)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    target = cfg["model"]["target"]
    seed = cfg["model"]["random_state"]
    cols = FEATURE_SETS["D"]
    X_fit, y_fit = fit_df[cols], fit_df[target]
    X_test, y_test = test_df[cols], test_df[target]

    models = {
        "linear": make_linear(seed),
        "random_forest": make_random_forest(cfg["model"]["n_estimators"], cfg["model"]["max_depth"], seed),
        "xgboost": make_xgboost(cfg["model"]["n_estimators"], cfg["model"]["max_depth"], seed),
    }
    regression_results = {}
    fitted = {}
    for name, est in models.items():
        est.fit(X_fit, y_fit)
        pred = est.predict(X_test)
        regression_results[name] = regression_metrics(y_test, pred)
        fitted[name] = est

    base_preds = baseline_predictions(fit_df, test_df, target)
    baseline_results = {name: regression_metrics(y_test, pred) for name, pred in base_preds.items()}

    best_name = min(regression_results, key=lambda k: regression_results[k]["mae"])
    if "xgboost" in fitted:
        best_name = "xgboost" if regression_results["xgboost"]["mae"] <= regression_results[best_name]["mae"] * 1.05 else best_name
    best_model = fitted[best_name]

    ablation = run_ablation(make_xgboost(cfg["model"]["n_estimators"], cfg["model"]["max_depth"], seed), fit_df, test_df, target)

    param_dist = {
        "n_estimators": [80, 120, 160],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.08, 0.12],
        "subsample": [0.7, 0.85],
        "colsample_bytree": [0.7, 0.9],
    }
    searcher = RandomizedSearchCV(
        make_xgboost(random_state=seed),
        param_distributions=param_dist,
        n_iter=cfg["model"]["n_iter_search"],
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=cfg["model"]["cv_splits"]),
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    searcher.fit(X_fit, y_fit)
    tuned = searcher.best_estimator_
    tuned_pred = tuned.predict(X_test)
    tuned_metrics = regression_metrics(y_test, tuned_pred)
    if tuned_metrics["mae"] <= regression_results.get(best_name, {}).get("mae", 1e9):
        best_model = tuned
        best_name = "xgboost_tuned"
        regression_results["xgboost_tuned"] = tuned_metrics

    y_cls_fit, mapping = encode_levels(fit_df["occupancy_level"])
    y_cls_test, _ = encode_levels(test_df["occupancy_level"])
    clf = make_peak_classifier(cfg["model"]["n_estimators"], cfg["model"]["max_depth"], seed)
    clf.fit(X_fit, y_cls_fit)
    cls_pred = clf.predict(X_test)
    cls_metrics = classification_metrics(y_cls_test, cls_pred, labels=list(range(4)))
    cm = confusion_matrix(y_cls_test, cls_pred, labels=list(range(4))).tolist()

    errors = summarize_errors(test_df, y_test, best_model.predict(X_test))
    tree_for_shap = best_model
    if best_name == "linear":
        tree_for_shap = fitted["xgboost"]
    shap_summary = shap_global(tree_for_shap, X_test, cfg["model"]["shap_sample_size"], seed)

    models_dir = project_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "regressor.joblib")
    joblib.dump(clf, models_dir / "classifier.joblib")
    meta = {
        "feature_cols": cols,
        "best_regressor": best_name,
        "level_mapping": {int(v): k for k, v in mapping.items()},
        "levels": LEVELS,
        "target": target,
    }
    joblib.dump(meta, models_dir / "meta.joblib")

    metrics = {
        "baselines": baseline_results,
        "regression": regression_results,
        "best_regressor": best_name,
        "classification": cls_metrics,
        "confusion_matrix": cm,
        "levels": LEVELS,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "split_dates": {
            "train_end": str(train_df["timestamp"].max()),
            "val_end": str(val_df["timestamp"].max()),
            "test_end": str(test_df["timestamp"].max()),
        },
    }
    _dump(project_path(cfg["paths"]["metrics_path"]), metrics)
    _dump(project_path(cfg["paths"]["ablation_path"]), ablation)
    _dump(project_path(cfg["paths"]["best_params_path"]), {"best_params": searcher.best_params_, "cv_mae": float(-searcher.best_score_)})
    _dump(project_path(cfg["paths"]["shap_path"]), shap_summary)
    _dump(project_path(cfg["paths"]["error_analysis_path"]), errors)

    print(json.dumps({"best": best_name, "regression": regression_results, "classification": cls_metrics}, indent=2, default=str))
    return metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
