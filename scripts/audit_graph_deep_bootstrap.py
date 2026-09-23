#!/usr/bin/env python3
"""Paired frozen-test bootstrap for graph baselines versus NEST-DDI."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"

def summary(y, candidate, nest, repeats=300, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ap, auc = [], []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        ap.append(average_precision_score(y[idx], candidate[idx]) - average_precision_score(y[idx], nest[idx]))
        auc.append(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], nest[idx]))
    def s(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))], "p_candidate_better": float(np.mean(v > 0))}
    return {"delta_candidate_minus_nest_aupr": s(ap), "delta_candidate_minus_nest_auroc": s(auc), "repeats": repeats}

def main():
    graph = np.load(OUT / "graph_deep_baselines_release_local_scores.npz")
    base = np.load(OUT / "final_comparison_release_local_scores.npz")
    result = {}
    for task, key in (("any_next_quarter", "target_any_next_quarter"), ("first_next_quarter", "target_first_next_quarter")):
        y = graph[f"{key}__y"]
        nest = base[task + "_nest"]
        result[task] = {name: summary(y, graph[f"{key}__{name}"], nest) for name in ("gcn", "gcn_fusion", "sage_fusion")}
    (OUT / "graph_deep_baselines_release_local_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
