#!/usr/bin/env python3
"""Select a leakage-safe first-pair ensemble on the prior rolling fold, then test once."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from run_nest_baselines import DRUG_HISTORY, PAIR_HISTORY, RAW_CROSS, STATIC
from run_nest_original import make_xgb, rank01


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "nest_first_optimized_release_local_metrics.json"
FEATURES = STATIC + PAIR_HISTORY + DRUG_HISTORY + RAW_CROSS


def metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    return {
        "aupr": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "precision_at_100": float(y[order[:100]].mean()),
    }


def main() -> None:
    panel = pd.read_parquet(ROOT / "data" / "processed" / "nest_ddi_prediction_panel_4q_release.parquet")
    panel = panel[panel["at_risk_first"].eq(1)]
    train = panel[panel["fold"].eq("2024q3")]
    valid = panel[panel["fold"].eq("2024q4")]
    test = panel[panel["fold"].eq("2025q1")]
    x_train, x_valid, x_test = (frame[FEATURES].fillna(0) for frame in (train, valid, test))
    y_train, y_valid, y_test = (frame["target_first_next_quarter"].to_numpy() for frame in (train, valid, test))

    hgb_configs = [
        {"learning_rate": 0.04, "max_iter": 260, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 2.0},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 45, "l2_regularization": 2.0},
        {"learning_rate": 0.03, "max_iter": 420, "max_leaf_nodes": 63, "min_samples_leaf": 55, "l2_regularization": 3.0},
    ]
    xgb_configs = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 12, "reg_lambda": 5.0, "reg_alpha": 0.1},
        {"n_estimators": 420, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10, "reg_lambda": 4.0, "reg_alpha": 0.1},
        {"n_estimators": 520, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.95, "min_child_weight": 12, "reg_lambda": 6.0, "reg_alpha": 0.2},
    ]

    candidates = []
    for hi, hp in enumerate(hgb_configs):
        hgb = HistGradientBoostingClassifier(random_state=42, **hp).fit(x_train, y_train)
        h_valid = rank01(hgb.predict_proba(x_valid)[:, 1])
        h_test = rank01(hgb.predict_proba(x_test)[:, 1])
        for xi, xp in enumerate(xgb_configs):
            xgb = make_xgb(y_train, xp).fit(x_train, y_train)
            x_valid_score = rank01(xgb.predict_proba(x_valid)[:, 1])
            x_test_score = rank01(xgb.predict_proba(x_test)[:, 1])
            for weight in (0.25, 0.5, 0.75):
                valid_score = weight * h_valid + (1 - weight) * x_valid_score
                candidates.append({
                    "hgb_index": hi, "xgb_index": xi, "hgb_weight": weight,
                    "valid": metric(y_valid, valid_score),
                    "valid_score": valid_score,
                    "test_score": weight * h_test + (1 - weight) * x_test_score,
                })
    candidates.sort(key=lambda item: item["valid"]["aupr"], reverse=True)
    best = candidates[0]
    # Refit the selected configuration on all pre-test labels before the one-time test.
    x_pretest = pd.concat([x_train, x_valid], axis=0)
    y_pretest = np.concatenate([y_train, y_valid])
    hgb_final = HistGradientBoostingClassifier(random_state=42, **hgb_configs[best["hgb_index"]]).fit(x_pretest, y_pretest)
    xgb_final = make_xgb(y_pretest, xgb_configs[best["xgb_index"]]).fit(x_pretest, y_pretest)
    h_final = rank01(hgb_final.predict_proba(x_test)[:, 1])
    x_final = rank01(xgb_final.predict_proba(x_test)[:, 1])
    final_score = best["hgb_weight"] * h_final + (1 - best["hgb_weight"]) * x_final
    test_metrics = metric(y_test, final_score)
    result = {
        "protocol": "select on 2024Q3->2024Q4 validation; evaluate once on frozen 2025Q1->2025Q2 test",
        "feature_count": len(FEATURES),
        "candidate_count": len(candidates),
        "selected": {k: v for k, v in best.items() if k not in ("test_score", "valid_score")},
        "test": test_metrics,
        "baseline_nest_test": {"aupr": 0.1478071612016273, "auroc": 0.9037738475084676, "precision_at_100": 0.30},
        "note": "Validation selection never reads frozen test labels; test score is retained only for the selected candidate.",
    }
    np.savez_compressed(OUT.with_suffix(".npz"), y_valid=y_valid, valid_score=best["valid_score"], y_test=y_test, optimized_score=final_score)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
