#!/usr/bin/env python3
"""Paired bootstrap for the validation-selected first-pair optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "nest_first_optimized_release_local_bootstrap.json"


def compare(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, repeats: int = 300, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    dap, dauc = [], []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        dap.append(average_precision_score(y[idx], candidate[idx]) - average_precision_score(y[idx], reference[idx]))
        dauc.append(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], reference[idx]))
    def summary(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))], "p_improvement": float((values > 0).mean())}
    return {"repeats": repeats, "delta_aupr": summary(dap), "delta_auroc": summary(dauc)}


def main() -> None:
    opt = np.load(OUT.with_name("nest_first_optimized_release_local_metrics.npz"))
    old = np.load(ROOT / "results" / "nest_ddi" / "final_comparison_release_local_scores.npz")
    y = opt["y_test"]
    result = {
        "protocol": "paired stratified bootstrap on frozen 2025Q1->2025Q2 first-pair test; optimizer selected on prior fold",
        "optimized_vs_original_nest": compare(y, opt["optimized_score"], old["first_next_quarter_nest"]),
        "optimized_vs_hgb_reference": compare(y, opt["optimized_score"], old["first_next_quarter_baseline"]),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
