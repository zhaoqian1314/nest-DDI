#!/usr/bin/env python3
"""Report fixed-capacity hit counts for the later recurrence holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"


def main() -> None:
    scores = np.load(OUT / "later_temporal_release_local_scores.npz")
    y, baseline, nest = scores["y"], scores["baseline"], scores["nest"]
    rows = []
    for capacity in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000):
        baseline_hits = int(y[np.argsort(-baseline)[:capacity]].sum())
        nest_hits = int(y[np.argsort(-nest)[:capacity]].sum())
        rows.append({
            "review_capacity": capacity, "baseline_hits": baseline_hits, "nest_hits": nest_hits,
            "incremental_hits": nest_hits - baseline_hits,
            "baseline_precision": baseline_hits / capacity, "nest_precision": nest_hits / capacity,
        })
    result = {
        "task": "later 2025Q3 recurrence, release-local case versions",
        "definition": "positive pairs among the first N ranked candidates; no utility or clinical value is implied",
        "capacities": rows,
    }
    (OUT / "later_temporal_release_local_operational_gain.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
