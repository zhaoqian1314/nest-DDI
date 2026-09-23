#!/usr/bin/env python3
"""Role-stratified later-holdout evaluation from cutoff-available DRUG records."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from build_nest_pilot import ascii_file, load_latest_case_metadata, norm


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
QUARTERS = ("2024q3", "2024q4", "2025q1", "2025q2")


def role_label(left: set[str], right: set[str]) -> str:
    a, b = ";".join(sorted(left)), ";".join(sorted(right))
    return "|".join(sorted((a, b)))


def main() -> None:
    panel = pd.read_parquet(DATA / "nest_ddi_prediction_panel_5q_release.parquet")
    test = panel.loc[panel["fold"].eq("2025q2"), ["pair_id"]].reset_index(drop=True)
    candidate_pairs = set(test["pair_id"])
    drugs = pd.read_csv(DATA / "ddinter_drugs.csv", dtype=str)
    name_map = {norm(name): drug_id for drug_id, name in zip(drugs["id"], drugs["name"])}
    metadata = load_latest_case_metadata(QUARTERS, "release_local")
    role_counts: dict[str, Counter] = defaultdict(Counter)
    for quarter in QUARTERS:
        valid_ids = set(metadata.loc[metadata["quarter"].eq(quarter), "primaryid"])
        case_roles: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for chunk in pd.read_csv(
            ascii_file(quarter, "DRUG"), sep="$", usecols=["primaryid", "drugname", "role_cod"],
            dtype=str, chunksize=150_000,
        ):
            chunk = chunk.loc[chunk["primaryid"].isin(valid_ids)]
            for row in chunk.itertuples(index=False):
                drug = name_map.get(norm(row.drugname))
                if drug:
                    case_roles[row.primaryid][drug].add(str(row.role_cod or "UNK"))
        for roles in case_roles.values():
            if len(roles) < 2 or len(roles) > 20:
                continue
            for left, right in combinations(sorted(roles), 2):
                pair = f"{left}|{right}"
                if pair in candidate_pairs:
                    role_counts[pair][role_label(roles[left], roles[right])] += 1
        print(quarter, "role-labelled pairs", len(role_counts), flush=True)
    strata = {pair: counts.most_common(1)[0][0] for pair, counts in role_counts.items()}
    scores = np.load(OUT / "later_temporal_release_local_scores.npz")
    y, baseline, nest = scores["y"], scores["baseline"], scores["nest"]
    if len(test) != len(y):
        raise RuntimeError("test panel and score lengths differ")
    labels = test["pair_id"].map(strata).fillna("NO_HISTORICAL_ROLE").to_numpy()
    rows = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 100 or y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        rows.append({
            "dominant_role": label, "pairs": int(len(idx)), "positives": int(y[idx].sum()),
            "positive_rate": float(y[idx].mean()),
            "baseline_aupr": float(average_precision_score(y[idx], baseline[idx])),
            "nest_aupr": float(average_precision_score(y[idx], nest[idx])),
            "baseline_auroc": float(roc_auc_score(y[idx], baseline[idx])),
            "nest_auroc": float(roc_auc_score(y[idx], nest[idx])),
        })
    rows.sort(key=lambda x: x["pairs"], reverse=True)
    result = {
        "task": "later 2025Q3 recurrence, cutoff-available dominant report role",
        "role_construction": "most frequent unordered DRUG role combination from 2024Q3--2025Q2; no exposure, dose, indication or perpetrator-victim inference",
        "role_labelled_pairs": int(len(strata)), "strata": rows,
    }
    (OUT / "later_temporal_release_local_role_strata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
