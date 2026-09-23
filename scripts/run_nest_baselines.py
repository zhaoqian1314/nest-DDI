#!/usr/bin/env python3
"""Run fair static and temporal baselines on the NEST-DDI rolling panel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(r"D:\BI\DDI")
PANEL = ROOT / "data" / "processed" / "nest_ddi_prediction_panel_4q.parquet"
OUT = ROOT / "results" / "nest_ddi"
STATIC = ["logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]
PAIR_HISTORY = [
    "pair_total", "pair_active_days", "pair_days_since",
    "pair_recent_7", "pair_decay_7", "pair_recent_30", "pair_decay_30",
    "pair_recent_90", "pair_decay_90",
]
DRUG_HISTORY = [
    c for side in (1, 2) for c in (
        f"drug{side}_total", f"drug{side}_partners", f"drug{side}_days_since",
        f"drug{side}_recent_7", f"drug{side}_decay_7",
        f"drug{side}_recent_30", f"drug{side}_decay_30",
        f"drug{side}_recent_90", f"drug{side}_decay_90",
    )
]
RAW_CROSS = [
    f"{prefix}_{days}"
    for days in (7, 30, 90)
    for prefix in ("cross_excitation", "activity_asymmetry", "pair_share")
]


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "aupr": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "positive_rate": float(np.mean(y)),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
    }


def fit_models(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> dict[str, np.ndarray]:
    logistic = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )
    logistic.fit(x_train, y_train)

    hist = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=220, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42,
    )
    hist.fit(x_train.fillna(0), y_train)

    positive = max(int(y_train.sum()), 1)
    xgb = XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=8, reg_lambda=2.0, reg_alpha=0.05,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=-1,
        scale_pos_weight=(len(y_train) - positive) / positive, random_state=42,
    )
    xgb.fit(x_train.fillna(0), y_train)
    return {
        "logistic": logistic.predict_proba(x_test)[:, 1],
        "hist_gradient_boosting": hist.predict_proba(x_test.fillna(0))[:, 1],
        "xgboost": xgb.predict_proba(x_test.fillna(0))[:, 1],
    }


def run_task(panel: pd.DataFrame, target: str, at_risk_only: bool) -> dict:
    frame = panel.loc[panel["at_risk_first"].eq(1)].copy() if at_risk_only else panel
    train = frame[frame["fold"].isin(["2024q3", "2024q4"])]
    test = frame[frame["fold"].eq("2025q1")]
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()
    feature_sets = {
        "static": STATIC,
        "history": PAIR_HISTORY + DRUG_HISTORY,
        "static_history": STATIC + PAIR_HISTORY + DRUG_HISTORY,
        "all_raw": STATIC + PAIR_HISTORY + DRUG_HISTORY + RAW_CROSS,
    }
    results = {}
    for name, features in feature_sets.items():
        predictions = fit_models(train[features], y_train, test[features])
        for model, score in predictions.items():
            key = f"{name}_{model}"
            results[key] = metrics(y_test, score)
            print(target, key, results[key])
    persistence = 1.0 - np.exp(-test["pair_decay_30"].to_numpy())
    results["persistence_score"] = metrics(y_test, persistence)
    return {
        "target": target,
        "at_risk_only": at_risk_only,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "results": results,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    output = {
        "protocol": "train rolling folds 2024q3+2024q4; test 2025q1->2025q2",
        "any_next_quarter": run_task(panel, "target_any_next_quarter", False),
        "first_next_quarter": run_task(panel, "target_first_next_quarter", True),
    }
    (OUT / "baseline_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
