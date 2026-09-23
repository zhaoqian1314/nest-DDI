#!/usr/bin/env python3
"""Train only the prior-fold GraphSAGE model to select a fusion weight."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_nest_graph_deep_baselines import fit_model, graph_matrix
from run_nest_original import rank01
from evaluate_nest_graph_fusion import metrics


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "graph_fusion_weight_selection.json"


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/nest_ddi_prediction_panel_4q_release.parquet")
    opt = np.load(ROOT / "results/nest_ddi/nest_first_optimized_release_local_metrics.npz")
    train = panel[panel["fold"].eq("2024q3") & panel["at_risk_first"].eq(1)].copy()
    valid = panel[panel["fold"].eq("2024q4") & panel["at_risk_first"].eq(1)].copy()
    if not np.array_equal(opt["y_valid"], valid["target_first_next_quarter"].to_numpy()):
        raise ValueError("validation labels are not aligned")
    adjacency, mapping = graph_matrix(panel, "2024q3")
    score = rank01(fit_model("sage_fusion", adjacency, mapping, train, valid, "target_first_next_quarter", 6, 16384, 42))
    nest = rank01(opt["valid_score"]); y = valid["target_first_next_quarter"].to_numpy()
    candidates = [{"nest_weight": w, "metrics": metrics(y, w * nest + (1 - w) * score)} for w in (0.25, 0.5, 0.75)]
    best = max(candidates, key=lambda item: item["metrics"]["aupr"])
    result = {"protocol": "2024Q3 train -> 2024Q4 validation only; frozen 2025Q1->2025Q2 labels are never read", "candidates": candidates, "selected": best}
    np.savez_compressed(OUT.with_suffix(".npz"), y_valid=y, nest_valid=nest, graph_valid=score)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
