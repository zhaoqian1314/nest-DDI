#!/usr/bin/env python3
"""Summarize two pre-specified seeds for the strongest neural baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"

def read(path: str, model: str):
    obj = json.loads((OUT / path).read_text(encoding="utf-8"))
    return {task: obj["tasks"][task]["results"][model] for task in ("target_any_next_quarter", "target_first_next_quarter")}

def main():
    pairs = {
        "ft_transformer": (read("deep_baselines_release_local_metrics.json", "ft_transformer"), read("deep_ft_seed7_metrics.json", "ft_transformer")),
        "sage_fusion": (read("graph_deep_baselines_release_local_metrics.json", "sage_fusion"), read("graph_sage_seed7_metrics.json", "sage_fusion")),
    }
    out = {"seeds": [42, 7], "note": "Two fixed seeds; no seed was selected using the frozen test fold.", "models": {}}
    for model, runs in pairs.items():
        out["models"][model] = {}
        for task in ("target_any_next_quarter", "target_first_next_quarter"):
            vals = {metric: [runs[0][task][metric], runs[1][task][metric]] for metric in ("auroc", "aupr", "brier", "precision_at_100")}
            out["models"][model][task] = {metric: {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v))} for metric, v in vals.items()}
    (OUT / "deep_graph_seed_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
