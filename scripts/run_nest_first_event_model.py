#!/usr/bin/env python3
"""NEST-DDI event-discovery model under the frozen first-event protocol.

NEST separates a structure/co-report background from event-specific propensity
and learns a residual reporting component.  Ensemble weights are selected only
on the 2024Q4 validation fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\BI\DDI")
DATA, OUT = ROOT / "data" / "processed", ROOT / "results" / "nest_ddi"
TARGET = "target_first_pair_event_next_quarter"
STATIC = ["logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]
BACKGROUND = STATIC + ["pair_history_count"]
RAW_EVENT = ["event_history_count", "drug1_event_history_count", "drug2_event_history_count"]


def metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    result = {"auroc": float(roc_auc_score(y, score)), "aupr_sampled": float(average_precision_score(y, score))}
    for k in (100, 500, 1000, 5000):
        result[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return result


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[BACKGROUND + RAW_EVENT].copy()
    for col in ["pair_history_count"] + RAW_EVENT:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))
    a, b = out["drug1_event_history_count"], out["drug2_event_history_count"]
    out["drug_event_geo"] = np.sqrt(a * b)
    out["drug_event_min"] = np.minimum(a, b)
    out["drug_event_max"] = np.maximum(a, b)
    out["drug_event_asymmetry"] = np.abs(np.log1p(a) - np.log1p(b))
    out["event_pair_exposure_ratio"] = out["event_history_count"] / (1 + out["pair_history_count"])
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def background_model() -> object:
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(),
                         SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced", max_iter=8,
                                       tol=1e-3, early_stopping=False, random_state=19))


def residual_model() -> object:
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(),
                         SGDClassifier(loss="log_loss", alpha=3e-6, class_weight="balanced", max_iter=8,
                                       tol=1e-3, early_stopping=False, random_state=42))


def fit_nest(train: pd.DataFrame, y: np.ndarray, evaluation: pd.DataFrame) -> np.ndarray:
    # Cross-fitting prevents residual learners from seeing a background score
    # that was fitted on the same label of the same row.
    bg_train = np.zeros(len(train), dtype=float)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=19)
    for fit_idx, holdout_idx in splitter.split(train, y):
        fold_background = background_model().fit(train.iloc[fit_idx][BACKGROUND], y[fit_idx])
        bg_train[holdout_idx] = fold_background.predict_proba(train.iloc[holdout_idx][BACKGROUND])[:, 1]
    background = background_model().fit(train[BACKGROUND], y)
    bg_eval = background.predict_proba(evaluation[BACKGROUND])[:, 1]
    x_train, x_eval = engineer(train), engineer(evaluation)
    x_train["background_logit"] = logit(np.clip(bg_train, 1e-5, 1-1e-5))
    x_eval["background_logit"] = logit(np.clip(bg_eval, 1e-5, 1-1e-5))
    residual = residual_model().fit(x_train, y)
    return residual.predict_proba(x_eval)[:, 1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / "nest_first_event_discovery_2024q3.parquet")
    valid = pd.read_parquet(DATA / "nest_first_event_discovery_2024q4.parquet")
    test = pd.read_parquet(DATA / "nest_first_event_discovery_2025q1.parquet")
    y_train, y_valid, y_test = train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy()
    validation_score = fit_nest(train, y_train, valid)
    validation = metric(y_valid, validation_score)
    combined = pd.concat([train, valid], ignore_index=True)
    score = fit_nest(combined, np.concatenate([y_train, y_valid]), test)
    result = {
        "protocol": "cross-fitted scalable linear background and linear residual fit on train; final models refit on train+validation; test opened once",
        "background_features": BACKGROUND,
        "event_residual_features": list(engineer(train).columns),
        "validation": validation,
        "frozen_test": metric(y_test, score),
        "warning": "The background is a predictive reporting-risk component, not a causal DDI mechanism estimator.",
    }
    np.savez_compressed(OUT / "first_event_nest_scores_2025q1.npz", y=y_test, nest_score=score)
    (OUT / "first_event_nest_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["frozen_test"], indent=2))


if __name__ == "__main__":
    main()
