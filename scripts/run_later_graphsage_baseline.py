#!/usr/bin/env python3
"""Run a historical GraphSAGE fusion on the later 2025Q2->Q3 holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_nest_graph_deep_baselines import fit_model, graph_matrix, metrics


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results/nest_ddi/later_graphsage_release_local_metrics.json"


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/nest_ddi_prediction_panel_5q_release.parquet")
    train = panel[panel["fold"].isin(["2024q3", "2024q4", "2025q1"])].copy()
    test = panel[panel["fold"].eq("2025q2")].copy()
    adjacency, mapping = graph_matrix(panel, "2025q1")
    score = fit_model("sage_fusion", adjacency, mapping, train, test, "target_any_next_quarter", 6, 16384, 42)
    result = {"protocol": "historical graph built through 2025Q1; train q3+q4+q1; test q2->q3", "test": metrics(test["target_any_next_quarter"].to_numpy(), score)}
    np.savez_compressed(OUT.with_suffix(".npz"), y=test["target_any_next_quarter"].to_numpy(), graphsage=score)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
