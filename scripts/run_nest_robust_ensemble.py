#!/usr/bin/env python3
"""Validation-selected robust ensemble for the difficult first-observation task."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from run_nest_original import RAW_DYNAMIC, STATIC, engineer, make_xgb, rank01


ROOT = Path(r"D:\BI\DDI")
PANEL = ROOT / "data" / "processed" / "nest_ddi_prediction_panel_4q.parquet"
OUT = ROOT / "results" / "nest_ddi" / "nest_first_robust_ensemble.json"
TARGET = "target_first_next_quarter"


SPECS = {
    "raw_hgb31": ("raw", "hist", {"learning_rate": 0.07, "max_iter": 240, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.5}),
    "raw_hgb63": ("raw", "hist", {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 45, "l2_regularization": 2.0}),
    "eng_hgb31": ("eng", "hist", {"learning_rate": 0.06, "max_iter": 260, "max_leaf_nodes": 31, "min_samples_leaf": 35, "l2_regularization": 1.5}),
    "eng_hgb63": ("eng", "hist", {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 45, "l2_regularization": 2.0}),
    "raw_xgb": ("raw", "xgb", {"n_estimators": 420, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10, "reg_lambda": 4.0, "reg_alpha": 0.1}),
    "eng_xgb": ("eng", "xgb", {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 8, "reg_lambda": 3.0, "reg_alpha": 0.1}),
}


def matrices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = frame[STATIC + RAW_DYNAMIC].astype("float32").fillna(0)
    eng, _ = engineer(frame)
    return {"raw": raw, "eng": eng}


def fit_predict(spec, train_x, y, pred_x):
    _, family, params = spec
    model = make_xgb(y, params) if family == "xgb" else HistGradientBoostingClassifier(random_state=42, **params)
    model.fit(train_x, y)
    return model.predict_proba(pred_x)[:, 1]


def metric(y, score):
    return {
        "aupr": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
    }


def main() -> None:
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["at_risk_first"].eq(1)].copy()
    early = panel[panel["fold"].eq("2024q3")]
    valid = panel[panel["fold"].eq("2024q4")]
    train = panel[panel["fold"].isin(["2024q3", "2024q4"])]
    test = panel[panel["fold"].eq("2025q1")]
    x_early, x_valid = matrices(early), matrices(valid)
    y_early, y_valid = early[TARGET].to_numpy(), valid[TARGET].to_numpy()

    valid_predictions = {}
    validation_metrics = {}
    for name, spec in SPECS.items():
        kind = spec[0]
        prediction = fit_predict(spec, x_early[kind], y_early, x_valid[kind])
        valid_predictions[name] = prediction
        validation_metrics[name] = metric(y_valid, prediction)
        print(name, validation_metrics[name])

    blend_grid = []
    for first, second in combinations(SPECS, 2):
        for mode in ("probability", "rank"):
            a = valid_predictions[first] if mode == "probability" else rank01(valid_predictions[first])
            b = valid_predictions[second] if mode == "probability" else rank01(valid_predictions[second])
            for weight in np.linspace(0.05, 0.95, 19):
                score = weight * a + (1 - weight) * b
                blend_grid.append((average_precision_score(y_valid, score), first, second, mode, float(weight)))
    blend_grid.sort(reverse=True)
    selected = blend_grid[0]
    print("selected", selected)

    x_train, x_test = matrices(train), matrices(test)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    final_predictions = {}
    for name in (selected[1], selected[2]):
        spec = SPECS[name]
        kind = spec[0]
        final_predictions[name] = fit_predict(spec, x_train[kind], y_train, x_test[kind])
    if selected[3] == "rank":
        a, b = rank01(final_predictions[selected[1]]), rank01(final_predictions[selected[2]])
    else:
        a, b = final_predictions[selected[1]], final_predictions[selected[2]]
    final_score = selected[4] * a + (1 - selected[4]) * b
    result = {
        "selection_rule": "all model identities and blend weights selected on 2024q4 validation only",
        "validation_metrics": validation_metrics,
        "selected": {"validation_aupr": float(selected[0]), "first": selected[1], "second": selected[2], "mode": selected[3], "weight_first": selected[4]},
        "test_components": {name: metric(y_test, pred) for name, pred in final_predictions.items()},
        "test_ensemble": metric(y_test, final_score),
        "test_rows": int(len(test)),
        "test_positives": int(y_test.sum()),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
