#!/usr/bin/env python3
"""Evaluate a fixed equal-weight rank fusion of optimized NEST and GraphSAGE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "nest_graph_fusion_release_local_metrics.json"


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(len(values) - 1, 1)


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    return {"aupr": float(average_precision_score(y, score)), "auroc": float(roc_auc_score(y, score)),
            "precision_at_100": float(y[order[:100]].mean()), "precision_at_500": float(y[order[:500]].mean()),
            "precision_at_1000": float(y[order[:1000]].mean())}


def bootstrap(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, repeats: int = 300, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    ap, auc = [], []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        ap.append(average_precision_score(y[idx], candidate[idx]) - average_precision_score(y[idx], reference[idx]))
        auc.append(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], reference[idx]))
    def summary(v):
        v = np.asarray(v); return {"mean": float(v.mean()), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))], "p_improvement": float((v > 0).mean())}
    return {"repeats": repeats, "delta_aupr": summary(ap), "delta_auroc": summary(auc)}


def main() -> None:
    opt = np.load(ROOT / "results" / "nest_ddi" / "nest_first_optimized_release_local_metrics.npz")
    graph = np.load(ROOT / "results" / "nest_ddi" / "graph_deep_baselines_release_local_scores.npz")
    y = opt["y_test"]
    if not np.array_equal(y, graph["target_first_next_quarter__y"]):
        raise ValueError("score files do not share the same frozen test labels")
    optimized = rank01(opt["optimized_score"])
    sage = rank01(graph["target_first_next_quarter__sage_fusion"])
    fused = 0.5 * optimized + 0.5 * sage
    triage_fused = 0.3 * optimized + 0.7 * sage
    result = {
        "protocol": "fixed 1:1 rank fusion; optimized NEST selected on prior validation fold; GraphSAGE trained on historical q3+q4 graph; frozen 2025Q1->2025Q2 test",
        "fusion_weight_optimized_nest": 0.5,
        "fusion_weight_graphsage": 0.5,
        "fused": metrics(y, fused),
        "triage_fused": metrics(y, triage_fused),
        "optimized_nest": metrics(y, optimized),
        "graphsage": metrics(y, sage),
        "bootstrap_vs_optimized_nest": bootstrap(y, fused, optimized),
        "bootstrap_vs_graphsage": bootstrap(y, fused, sage),
        "triage_bootstrap_vs_graphsage": bootstrap(y, triage_fused, sage),
        "note": "Equal weights are fixed for this audit and are not selected from frozen test labels; this is a first-pair optimization, not a recurrence model claim.",
    }
    np.savez_compressed(OUT.with_suffix(".npz"), y_test=y, fused_score=fused, triage_fused_score=triage_fused, optimized_score=optimized, graphsage_score=sage)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
