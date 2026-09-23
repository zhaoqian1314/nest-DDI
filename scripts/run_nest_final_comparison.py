#!/usr/bin/env python3
"""Reproduce selected NEST models and perform paired temporal-test bootstrap."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from run_nest_baselines import DRUG_HISTORY, PAIR_HISTORY, RAW_CROSS, STATIC
from run_nest_original import engineer, fit_background, make_xgb, rank01


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"


def metrics(y, score):
    order = np.argsort(-score)
    result = {"aupr": float(average_precision_score(y, score)), "auroc": float(roc_auc_score(y, score))}
    for k in (100, 500, 1000):
        result[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return result


def paired_bootstrap(y, candidate, baseline, repeats=300, seed=42):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas_ap, deltas_auc = [], []
    for _ in range(repeats):
        idx = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        yi = y[idx]
        deltas_ap.append(average_precision_score(yi, candidate[idx]) - average_precision_score(yi, baseline[idx]))
        deltas_auc.append(roc_auc_score(yi, candidate[idx]) - roc_auc_score(yi, baseline[idx]))
    def summary(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
                "p_improvement": float((values > 0).mean())}
    return {"repeats": repeats, "delta_aupr": summary(deltas_ap), "delta_auroc": summary(deltas_auc)}


def any_task(panel):
    train = panel[panel["fold"].isin(["2024q3", "2024q4"])]
    test = panel[panel["fold"].eq("2025q1")]
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
    return y_test, baseline_score, nest_score


def first_task(panel):
    panel = panel[panel["at_risk_first"].eq(1)]
    train = panel[panel["fold"].isin(["2024q3", "2024q4"])]
    test = panel[panel["fold"].eq("2025q1")]
    y_train = train["target_first_next_quarter"].to_numpy()
    y_test = test["target_first_next_quarter"].to_numpy()
    features = STATIC + PAIR_HISTORY + DRUG_HISTORY + RAW_CROSS
    x_train, x_test = train[features].fillna(0), test[features].fillna(0)
    baseline = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=220, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42,
    ).fit(x_train, y_train)
    baseline_score = baseline.predict_proba(x_test)[:, 1]
    smooth = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=63, min_samples_leaf=45,
        l2_regularization=2.0, random_state=42,
    ).fit(x_train, y_train)
    boosted = make_xgb(y_train, {
        "n_estimators": 420, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.9,
        "colsample_bytree": 0.9, "min_child_weight": 10, "reg_lambda": 4.0, "reg_alpha": 0.1,
    }).fit(x_train, y_train)
    nest_score = 0.55 * rank01(smooth.predict_proba(x_test)[:, 1]) + 0.45 * rank01(boosted.predict_proba(x_test)[:, 1])
    return y_test, baseline_score, nest_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="4q")
    parser.add_argument("--output-tag", default="final_comparison")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "processed" / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    result = {}
    arrays = {}
    for name, runner in (("any_next_quarter", any_task), ("first_next_quarter", first_task)):
        y, baseline, nest = runner(panel)
        result[name] = {
            "baseline": metrics(y, baseline),
            "nest": metrics(y, nest),
            "paired_bootstrap": paired_bootstrap(y, nest, baseline),
        }
        arrays[name + "_y"] = y
        arrays[name + "_baseline"] = baseline
        arrays[name + "_nest"] = nest
        print(name, result[name])
    np.savez_compressed(OUT / f"{args.output_tag}_scores.npz", **arrays)
    (OUT / f"{args.output_tag}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
