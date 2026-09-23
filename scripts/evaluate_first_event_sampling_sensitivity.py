#!/usr/bin/env python3
"""Evaluate fixed linear ranking scores across case-control sampling variants."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
TARGET = "target_first_pair_event_next_quarter"
FEATURES = [
    "logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto",
    "pair_history_count", "event_history_count",
    "drug1_event_history_count", "drug2_event_history_count",
]


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    result = {"auroc": float(roc_auc_score(y, score)), "aupr_sampled": float(average_precision_score(y, score))}
    for k in (100, 500, 1000, 5000):
        result[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.concat([
        pd.read_parquet(DATA / "nest_first_event_discovery_2024q3.parquet"),
        pd.read_parquet(DATA / "nest_first_event_discovery_2024q4.parquet"),
    ], ignore_index=True)
    x = train[FEATURES].to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy()
    scaler = StandardScaler().fit(x)
    model = SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced",
                          max_iter=12, tol=1e-3, random_state=42).fit(scaler.transform(x), y)
    variants = {
        "r10_s42": DATA / "nest_first_event_discovery_2025q1.parquet",
        "r5_s42": DATA / "nest_first_event_discovery_2025q1_r5_s42.parquet",
        "r10_s7": DATA / "nest_first_event_discovery_2025q1_r10_s7.parquet",
    }
    output = {
        "model": "fixed SGD logistic score trained once on 2024q3+2024q4; not the main XGBoost result",
        "features": FEATURES, "variants": {},
    }
    for name, path in variants.items():
        panel = pd.read_parquet(path)
        score = model.predict_proba(scaler.transform(panel[FEATURES].to_numpy(dtype=np.float32)))[:, 1]
        output["variants"][name] = {
            "rows": int(len(panel)), "positive_rate_sampled": float(panel[TARGET].mean()),
            "metrics": metrics(panel[TARGET].to_numpy(), score),
        }
        print(name, output["variants"][name], flush=True)
    (OUT / "first_event_sampling_sensitivity.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
