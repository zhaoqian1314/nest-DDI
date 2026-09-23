#!/usr/bin/env python3
"""Fit leakage-free baselines for first pair--event discovery.

Model selection uses 2024Q4 only.  The 2025Q1 panel is loaded once for the
final frozen evaluation and is never used to choose a hyperparameter.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
TARGET = "target_first_pair_event_next_quarter"
STATIC = ["logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]
EVENT_HISTORY = [
    "pair_history_count", "event_history_count",
    "drug1_event_history_count", "drug2_event_history_count",
]
FEATURES = STATIC + EVENT_HISTORY


def metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    out = {"auroc": float(roc_auc_score(y, score)), "aupr_sampled": float(average_precision_score(y, score))}
    for k in (100, 500, 1000, 5000):
        out[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return out


def build_models(y: np.ndarray) -> dict[str, object]:
    pos = max(int(y.sum()), 1)
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(),
                                  LogisticRegression(max_iter=300, class_weight="balanced", C=0.5)),
        "hgb_shallow": HistGradientBoostingClassifier(learning_rate=0.07, max_iter=250, max_leaf_nodes=31,
                                                         min_samples_leaf=100, l2_regularization=2.0, random_state=42),
        "hgb_deep": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=350, max_leaf_nodes=63,
                                                      min_samples_leaf=80, l2_regularization=3.0, random_state=42),
        "xgboost": XGBClassifier(n_estimators=450, max_depth=7, learning_rate=0.05, subsample=0.85,
                                  colsample_bytree=0.9, min_child_weight=30, reg_lambda=5.0, reg_alpha=0.1,
                                  scale_pos_weight=(len(y) - pos) / pos, objective="binary:logistic",
                                  eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42),
    }


def fit_predict(model: object, x_train: pd.DataFrame, y_train: np.ndarray, x_eval: pd.DataFrame) -> np.ndarray:
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def audit_prevalence(fold: str, suffix: str) -> dict:
    return json.loads((DATA / f"nest_first_event_discovery_{fold}{suffix}_audit.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="")
    parser.add_argument("--output-tag", default="first_event_baseline")
    args = parser.parse_args()
    suffix = f"_{args.input_tag}" if args.input_tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / f"nest_first_event_discovery_2024q3{suffix}.parquet")
    valid = pd.read_parquet(DATA / f"nest_first_event_discovery_2024q4{suffix}.parquet")
    test = pd.read_parquet(DATA / f"nest_first_event_discovery_2025q1{suffix}.parquet")
    x_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    x_valid, y_valid = valid[FEATURES], valid[TARGET].to_numpy()
    validation = {}
    for name, model in build_models(y_train).items():
        score = fit_predict(model, x_train, y_train, x_valid)
        validation[name] = metric(y_valid, score)
        print("validation", name, validation[name], flush=True)
    selected = max(validation, key=lambda name: validation[name]["aupr_sampled"])

    combined_x = pd.concat([x_train, x_valid], ignore_index=True)
    combined_y = np.concatenate([y_train, y_valid])
    selected_model = build_models(combined_y)[selected]
    test_score = fit_predict(selected_model, combined_x, combined_y, test[FEATURES])
    all_features_hgb = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=320, max_leaf_nodes=63,
                                                       min_samples_leaf=100, l2_regularization=3.0, random_state=42)
    all_features_score = fit_predict(all_features_hgb, combined_x, combined_y, test[FEATURES])
    result = {
        "protocol": "train 2024q3, choose model on 2024q4, refit train+validation, test once on 2025q1",
        "features": FEATURES,
        "validation": validation,
        "selected_by_validation_aupr_sampled": selected,
        "frozen_test": {selected: metric(test[TARGET].to_numpy(), test_score), "hgb_full_feature": metric(test[TARGET].to_numpy(), all_features_score)},
        "sampling_audit": {fold: audit_prevalence(fold, suffix) for fold in ("2024q3", "2024q4", "2025q1")},
        "warning": "AUPR is computed on the documented case-control sample. AUROC and Precision@K are the primary cross-fold ranking measures.",
    }
    np.savez_compressed(OUT / f"{args.output_tag}_scores_2025q1.npz", y=test[TARGET].to_numpy(),
                        selected_score=test_score, hgb_full_feature_score=all_features_score)
    (OUT / f"{args.output_tag}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["frozen_test"], indent=2))


if __name__ == "__main__":
    main()
