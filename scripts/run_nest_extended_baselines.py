#!/usr/bin/env python3
"""Additional fair baselines for the NEST-DDI rolling prediction panel.

All learned models use the same training folds, test fold, candidate pool and
labels as run_nest_baselines.py.  The Hawkes-basis model is deliberately named
as a discrete-time proxy: it uses fixed exponential-decay sufficient statistics
rather than claiming to be a fitted continuous-time Hawkes process.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_nest_baselines import DRUG_HISTORY, PAIR_HISTORY, RAW_CROSS, STATIC


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"
ALL_RAW = STATIC + PAIR_HISTORY + DRUG_HISTORY + RAW_CROSS
HAWKES_BASIS = STATIC + [
    "pair_total", "pair_active_days", "pair_days_since",
    "pair_decay_7", "pair_decay_30", "pair_decay_90",
    "drug1_decay_7", "drug2_decay_7",
    "drug1_decay_30", "drug2_decay_30",
    "drug1_decay_90", "drug2_decay_90",
]


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score)
    result = {
        "auroc": float(roc_auc_score(y, score)),
        "aupr": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "positive_rate": float(np.mean(y)),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
    }
    for k in (100, 500, 1000):
        result[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return result


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def heuristic_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    recency = rank01(frame["pair_decay_30"].to_numpy())
    frequency = rank01(np.log1p(frame["pair_total"].to_numpy()))
    popularity = rank01(
        np.sqrt(np.maximum(frame["drug1_decay_30"].to_numpy(), 0)
                * np.maximum(frame["drug2_decay_30"].to_numpy(), 0))
    )
    return {
        "edgebank_frequency": frequency,
        "temporal_recency": recency,
        "recency_popularity": 0.6 * recency + 0.4 * popularity,
    }


def fitted_scores(train: pd.DataFrame, y: np.ndarray, test: pd.DataFrame) -> dict[str, np.ndarray]:
    x_train = train[ALL_RAW].fillna(0)
    x_test = test[ALL_RAW].fillna(0)

    extra = ExtraTreesClassifier(
        n_estimators=180, max_features="sqrt", min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ).fit(x_train, y)

    forest = RandomForestClassifier(
        n_estimators=160, max_depth=24, max_features="sqrt", min_samples_leaf=4,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    ).fit(x_train, y)

    hawkes_logit = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", C=0.5),
    ).fit(train[HAWKES_BASIS], y)

    # A non-negative log-link intensity baseline.  Ranking metrics remain valid
    # even though the fitted mean is an event rate rather than a calibrated risk.
    poisson = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        PoissonRegressor(alpha=1.0, max_iter=300),
    ).fit(train[HAWKES_BASIS], y)

    mlp = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=2048,
            learning_rate_init=1e-3, max_iter=25, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=4, random_state=42,
        ),
    ).fit(train[ALL_RAW], y)

    return {
        "extra_trees_all_raw": extra.predict_proba(x_test)[:, 1],
        "random_forest_all_raw": forest.predict_proba(x_test)[:, 1],
        "hawkes_basis_logistic": hawkes_logit.predict_proba(test[HAWKES_BASIS])[:, 1],
        "poisson_intensity_glm": poisson.predict(test[HAWKES_BASIS]),
        "mlp_all_raw": mlp.predict_proba(test[ALL_RAW])[:, 1],
    }


def run_task(panel: pd.DataFrame, target: str, at_risk_only: bool) -> dict:
    frame = panel.loc[panel["at_risk_first"].eq(1)].copy() if at_risk_only else panel
    train = frame[frame["fold"].isin(["2024q3", "2024q4"])]
    test = frame[frame["fold"].eq("2025q1")]
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    predictions = heuristic_scores(test)
    predictions.update(fitted_scores(train, y_train, test))
    results = {}
    for name, score in predictions.items():
        results[name] = metrics(y_test, score)
        print(target, name, results[name], flush=True)
    return {
        "target": target,
        "at_risk_only": at_risk_only,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="4q")
    parser.add_argument("--output-tag", default="extended_baseline")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "processed" / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    result = {
        "protocol": "train 2024q3+2024q4; test 2025q1->2025q2; identical risk sets to baseline_metrics.json",
        "notes": {
            "hawkes_basis_logistic": "discrete-time fixed exponential-basis proxy, not a continuous-time Hawkes likelihood",
            "poisson_intensity_glm": "log-link event-intensity ranking baseline",
            "temporal_heuristics": "test scores use history available at the cutoff only",
        },
        "any_next_quarter": run_task(panel, "target_any_next_quarter", False),
        "first_next_quarter": run_task(panel, "target_first_next_quarter", True),
    }
    (OUT / f"{args.output_tag}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
