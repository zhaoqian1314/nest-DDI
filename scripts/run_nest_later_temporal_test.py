#!/usr/bin/env python3
"""Evaluate the pair-level NEST comparator on the later 2025Q2-->2025Q3 fold."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from run_nest_baselines import DRUG_HISTORY, PAIR_HISTORY, STATIC
from run_nest_final_comparison import metrics, paired_bootstrap
from run_nest_original import engineer, fit_background


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="5q")
    parser.add_argument("--output-tag", default="later_temporal_test")
    args = parser.parse_args()
    panel = pd.read_parquet(DATA / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    train = panel[panel["fold"].isin(["2024q3", "2024q4", "2025q1"])]
    test = panel[panel["fold"].eq("2025q2")]
    y_train = train["target_any_next_quarter"].to_numpy()
    y_test = test["target_any_next_quarter"].to_numpy()
    baseline_features = STATIC + PAIR_HISTORY + DRUG_HISTORY
    baseline = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=220, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42,
    ).fit(train[baseline_features].fillna(0), y_train)
    baseline_score = baseline.predict_proba(test[baseline_features].fillna(0))[:, 1]

    x_train, features = engineer(train)
    x_test, _ = engineer(test)
    base_test, background = fit_background(x_train, y_train, x_test)
    base_train = background.predict_proba(x_train[STATIC])[:, 1]
    x_train["background_logit"] = logit(np.clip(base_train, 1e-5, 1 - 1e-5))
    x_test["background_logit"] = logit(np.clip(base_test, 1e-5, 1 - 1e-5))
    features += ["background_logit"]
    nest = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=260, max_leaf_nodes=31, min_samples_leaf=35,
        l2_regularization=1.5, random_state=42,
    ).fit(x_train[features], y_train)
    nest_score = nest.predict_proba(x_test[features])[:, 1]

    result = {
        "task": "any previously observed pair reported in 2025Q3",
        "train_folds": ["2024q3", "2024q4", "2025q1"], "test_fold": "2025q2",
        "test_positive_rate": float(y_test.mean()), "baseline": metrics(y_test, baseline_score),
        "nest": metrics(y_test, nest_score),
        "paired_bootstrap": paired_bootstrap(y_test, nest_score, baseline_score),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.output_tag}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"{args.output_tag}_scores.npz", y=y_test, baseline=baseline_score, nest=nest_score)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
