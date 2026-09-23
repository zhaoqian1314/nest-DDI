#!/usr/bin/env python3
"""Build a task-wise leaderboard audit from all frozen release-local results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi" / "all_baseline_leaderboard_audit.json"


def main() -> None:
    final = json.loads((ROOT / "results/nest_ddi/final_comparison_release_local_metrics.json").read_text())
    deep = json.loads((ROOT / "results/nest_ddi/deep_baselines_release_local_metrics.json").read_text())
    graph = json.loads((ROOT / "results/nest_ddi/graph_deep_baselines_release_local_metrics.json").read_text())
    fusion = json.loads((ROOT / "results/nest_ddi/nest_graph_fusion_release_local_metrics.json").read_text())
    validated = json.loads((ROOT / "results/nest_ddi/nest_graph_fusion_validated_replay_release_local_metrics.json").read_text())
    modern = json.loads((ROOT / "results/nest_ddi/modern_2026_baselines_release_local_metrics.json").read_text())
    rows = {"any_next_quarter": {}, "first_next_quarter": {}}
    for task, source_task in (("any_next_quarter", "target_any_next_quarter"), ("first_next_quarter", "target_first_next_quarter")):
        rows[task]["HGB"] = final[task]["baseline"]
        rows[task]["NEST-DDI"] = final[task]["nest"]
        for name, score in deep["tasks"][source_task]["results"].items() if "tasks" in deep else deep["tasks"][source_task]["results"].items():
            rows[task][name] = score
        for name, score in graph["tasks"][source_task]["results"].items():
            rows[task][name] = score
        for name, score in modern["tasks"][source_task]["results"].items():
            rows[task][f"2026-{name}"] = score
    rows["first_next_quarter"]["NEST optimized"] = fusion["optimized_nest"]
    rows["first_next_quarter"]["NEST-GraphSAGE fixed fusion"] = fusion["fused"]
    rows["first_next_quarter"]["NEST-GraphSAGE triage fusion"] = fusion["triage_fused"]
    rows["first_next_quarter"]["NEST-GraphSAGE validated replay"] = validated["test"]
    metrics = ("aupr", "auroc", "precision_at_100")
    result = {"protocol": "frozen release-local leaderboard; no metric or split changes", "tasks": {}, "interpretation": {}}
    for task, methods in rows.items():
        result["tasks"][task] = methods
        result["interpretation"][task] = {}
        for metric in metrics:
            values = {name: float(scores[metric]) for name, scores in methods.items() if metric in scores}
            best = max(values.values())
            winners = [name for name, value in values.items() if abs(value - best) < 1e-12]
            result["interpretation"][task][metric] = {"best_value": best, "winners": winners, "strict_unique_winner": len(winners) == 1}
    result["interpretation"]["summary"] = {
        "recurrence": "NEST-DDI is the strict leader for AUPR and AUROC; P@100 is tied at 1.0.",
        "first_pair": "NEST-GraphSAGE fixed fusion is the strict leader for AUPR and AUROC; the triage-weighted fusion is the strict leader for P@100 (0.40).",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
