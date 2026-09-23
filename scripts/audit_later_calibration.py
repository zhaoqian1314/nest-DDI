#!/usr/bin/env python3
"""Probability-calibration diagnostics for the later recurrence holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"


def calibration(y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    order = np.argsort(score)
    groups = np.array_split(order, 10)
    bins, ece = [], 0.0
    for idx in groups:
        observed = float(y[idx].mean())
        predicted = float(score[idx].mean())
        weight = len(idx) / len(y)
        ece += weight * abs(observed - predicted)
        bins.append({"n": int(len(idx)), "mean_predicted": predicted, "observed_rate": observed})
    return {"brier": float(brier_score_loss(y, score)), "ece_equal_frequency_10": float(ece), "bins": bins}


def main() -> None:
    scores = np.load(OUT / "later_temporal_release_local_scores.npz")
    result = {
        "task": "later 2025Q3 recurrence, release-local case versions",
        "definition": "in-sample holdout calibration diagnostic; it is not patient-risk or clinical calibration",
        "baseline": calibration(scores["y"], scores["baseline"]),
        "nest": calibration(scores["y"], scores["nest"]),
    }
    (OUT / "later_temporal_release_local_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
