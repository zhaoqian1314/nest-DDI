#!/usr/bin/env python3
"""Select NEST/GraphSAGE fusion weight on the prior fold and test once."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_nest_graph_fusion import bootstrap, metrics, rank01
from run_nest_graph_deep_baselines import fit_model, graph_matrix


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "nest_graph_fusion_validated_release_local_metrics.json"


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/nest_ddi_prediction_panel_4q_release.parquet")
    opt = np.load(ROOT / "results/nest_ddi/nest_first_optimized_release_local_metrics.npz")
    valid = panel[panel["fold"].eq("2024q4") & panel["at_risk_first"].eq(1)].copy()
    train_valid = panel[panel["fold"].eq("2024q3") & panel["at_risk_first"].eq(1)].copy()
    if not np.array_equal(opt["y_valid"], valid["target_first_next_quarter"].to_numpy()):
        raise ValueError("optimized and graph validation labels are misaligned")
    adjacency_v, mapping_v = graph_matrix(panel, "2024q3")
    graph_valid = fit_model("sage_fusion", adjacency_v, mapping_v, train_valid, valid, "target_first_next_quarter", 6, 16384, 42)
    nest_valid = rank01(opt["valid_score"])
    graph_valid = rank01(graph_valid)
    candidates = []
    for w in (0.25, 0.5, 0.75):
        score = w * nest_valid + (1 - w) * graph_valid
        candidates.append({"nest_weight": w, "validation": metrics(valid["target_first_next_quarter"].to_numpy(), score)})
    best = max(candidates, key=lambda x: x["validation"]["aupr"])

    pretest = panel[panel["fold"].isin(["2024q3", "2024q4"]) & panel["at_risk_first"].eq(1)].copy()
    test = panel[panel["fold"].eq("2025q1") & panel["at_risk_first"].eq(1)].copy()
    adjacency_t, mapping_t = graph_matrix(panel, "2024q4")
    graph_test = rank01(fit_model("sage_fusion", adjacency_t, mapping_t, pretest, test, "target_first_next_quarter", 6, 16384, 42))
    nest_test = rank01(opt["optimized_score"])
    y_test = test["target_first_next_quarter"].to_numpy()
    fused = best["nest_weight"] * nest_test + (1 - best["nest_weight"]) * graph_test
    result = {"protocol": "weight selected on 2024Q3->Q4 validation; components refit on pre-test data; frozen 2025Q1->Q2 tested once", "candidates": candidates, "selected": best, "test": metrics(y_test, fused), "bootstrap_vs_graphsage": bootstrap(y_test, fused, graph_test), "bootstrap_vs_optimized_nest": bootstrap(y_test, fused, nest_test)}
    np.savez_compressed(OUT.with_suffix(".npz"), y_test=y_test, fused_score=fused, graph_score=graph_test, nest_score=nest_test)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
