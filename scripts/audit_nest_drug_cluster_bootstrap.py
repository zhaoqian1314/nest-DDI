#!/usr/bin/env python3
"""Drug-block bootstrap sensitivity for the later pair-recurrence holdout.

Pairs are canonically assigned to their first drug identifier, so every pair is
included in exactly one resampled drug block.  This is a conservative
dependence sensitivity, not a replacement for an independent reporting system.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    x = np.asarray(values)
    return {
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))],
        "p_improvement": float((x > 0).mean()),
    }


def main() -> None:
    panel = pd.read_parquet(DATA / "nest_ddi_prediction_panel_5q_release.parquet")
    test = panel.loc[panel["fold"].eq("2025q2"), ["drug1_id"]].reset_index(drop=True)
    scores = np.load(OUT / "later_temporal_release_local_scores.npz")
    y, baseline, nest = scores["y"], scores["baseline"], scores["nest"]
    if len(test) != len(y):
        raise RuntimeError("test-panel and score lengths differ")
    blocks = test["drug1_id"].to_numpy()
    unique, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(42)
    ap, auc = [], []
    for _ in range(300):
        selected = rng.integers(len(unique), size=len(unique))
        index = np.concatenate([members[i] for i in selected])
        yi = y[index]
        ap.append(average_precision_score(yi, nest[index]) - average_precision_score(yi, baseline[index]))
        auc.append(roc_auc_score(yi, nest[index]) - roc_auc_score(yi, baseline[index]))
    result = {
        "task": "later 2025Q3 recurrence, release-local case versions",
        "block_definition": "canonical first drug identifier in unordered pair",
        "unique_drug_blocks": int(len(unique)), "repeats": 300,
        "delta_aupr": summarize(ap), "delta_auroc": summarize(auc),
        "limitation": "one-way drug-block resampling addresses shared-drug dependence only partially",
    }
    (OUT / "later_temporal_release_local_drug_cluster_bootstrap.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
