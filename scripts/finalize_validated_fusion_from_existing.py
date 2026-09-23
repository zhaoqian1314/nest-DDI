#!/usr/bin/env python3
"""Finalize a validation-selected fusion using the already audited q3+q4 GraphSAGE scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluate_nest_graph_fusion import bootstrap, metrics, rank01


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results/nest_ddi/nest_graph_fusion_validated_replay_release_local_metrics.json"


def main() -> None:
    opt = np.load(ROOT / "results/nest_ddi/nest_first_optimized_release_local_metrics.npz")
    graph = np.load(ROOT / "results/nest_ddi/graph_deep_baselines_release_local_scores.npz")
    y = opt["y_test"]
    if not np.array_equal(y, graph["target_first_next_quarter__y"]):
        raise ValueError("test labels are not aligned")
    nest = rank01(opt["optimized_score"]); sage = rank01(graph["target_first_next_quarter__sage_fusion"])
    weight = 0.75
    fused = weight * nest + (1 - weight) * sage
    result = {"protocol": "weight 0.75 selected on 2024Q4 validation; q3+q4 GraphSAGE test score reused from audited run; frozen 2025Q1->Q2 test", "nest_weight": weight, "graphsage_weight": 1-weight, "test": metrics(y, fused), "bootstrap_vs_nest": bootstrap(y, fused, nest), "bootstrap_vs_graphsage": bootstrap(y, fused, sage), "note": "This replay does not refit GraphSAGE; it reuses the previously audited q3+q4 model output with the validation-selected weight."}
    np.savez_compressed(OUT.with_suffix(".npz"), y_test=y, fused_score=fused)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
