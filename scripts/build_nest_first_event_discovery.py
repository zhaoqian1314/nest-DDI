#!/usr/bin/env python3
"""Build a no-full-period-selection first pair--event discovery test panel.

Vocabulary, pair eligibility and all covariates are selected at the 2025Q1
cutoff.  Raw Q2 reports are read only to assign the future label.
"""
from __future__ import annotations

import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_nest_pilot import ascii_file, load_latest_case_metadata, norm


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
FOLDS = {
    "2024q3": (pd.Timestamp("2024-09-30"), pd.Timestamp("2024-12-31"), ("2024q3", "2024q4")),
    "2024q4": (pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31"), ("2024q3", "2024q4", "2025q1")),
    "2025q1": (pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30"), ("2024q3", "2024q4", "2025q1", "2025q2")),
}
TOP_EVENTS = 100
MIN_PAIR_HISTORY = 3
NEGATIVE_RATIO = 10


def load_raw_pair_events(
    event_vocab: set[str], eligible_pairs: set[str], name_map: dict[str, str], metadata: pd.DataFrame,
    quarters: tuple[str, ...], cutoff: pd.Timestamp,
) -> Counter:
    """Return report counts for selected pair-event triplets from raw files."""
    counts: Counter = Counter()
    for quarter in quarters:
        meta = metadata.loc[metadata["quarter"].eq(quarter)].set_index("primaryid")
        valid = set(meta.index)
        drug_sets: dict[str, set[str]] = defaultdict(set)
        for chunk in pd.read_csv(
            ascii_file(quarter, "DRUG"), sep="$", usecols=["primaryid", "drugname"],
            dtype=str, chunksize=150_000,
        ):
            chunk = chunk[chunk["primaryid"].isin(valid)]
            for row in chunk.itertuples(index=False):
                drug_id = name_map.get(norm(row.drugname))
                if drug_id:
                    drug_sets[row.primaryid].add(drug_id)
        reactions: dict[str, set[str]] = defaultdict(set)
        for chunk in pd.read_csv(
            ascii_file(quarter, "REAC"), sep="$", usecols=["primaryid", "pt"],
            dtype=str, chunksize=200_000,
        ):
            chunk = chunk[chunk["primaryid"].isin(valid)]
            for row in chunk.itertuples(index=False):
                if row.pt and str(row.pt) != "nan" and str(row.pt).strip() in event_vocab:
                    reactions[row.primaryid].add(str(row.pt).strip())
        for primaryid, drugs in drug_sets.items():
            if len(drugs) < 2 or len(drugs) > 20:
                continue
            date = pd.Timestamp(meta.at[primaryid, "report_date"])
            events = reactions.get(primaryid, set())
            if not events:
                continue
            ordered = sorted(drugs)
            for i, drug1 in enumerate(ordered):
                for drug2 in ordered[i + 1:]:
                    pair_id = f"{drug1}|{drug2}"
                    if pair_id not in eligible_pairs:
                        continue
                    for event in events:
                        counts[(pair_id, event, date <= cutoff)] += 1
        print(quarter, "selected triplets", len(counts), flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=sorted(FOLDS), default="2025q1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--input-tag", default="4q")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--version-policy", choices=("global_latest", "release_local"), default="global_latest")
    args = parser.parse_args()
    cutoff, horizon, quarters = FOLDS[args.fold]
    OUT.mkdir(parents=True, exist_ok=True)
    event_daily = pd.read_parquet(DATA / f"nest_pv_event_daily_{args.input_tag}.parquet")
    history_event = event_daily.loc[event_daily["report_date"].le(cutoff)].groupby("meddra_pt")["report_count"].sum()
    event_vocab = set(history_event.nlargest(TOP_EVENTS).index)
    pair_daily = pd.read_csv(DATA / f"nest_ddi_pair_daily_{args.input_tag}.csv", parse_dates=["report_date"])
    pair_history = pair_daily.loc[pair_daily["report_date"].le(cutoff)].groupby("pair_id")["report_count"].sum()
    eligible_pairs = set(pair_history.loc[pair_history.ge(MIN_PAIR_HISTORY)].index)

    drugs = pd.read_csv(DATA / "ddinter_drugs.csv", dtype=str)
    name_map = {norm(name): drug_id for drug_id, name in zip(drugs["id"], drugs["name"])}
    metadata = load_latest_case_metadata(quarters, args.version_policy)
    raw_counts = load_raw_pair_events(event_vocab, eligible_pairs, name_map, metadata, quarters, cutoff)
    history = {(pair, event) for (pair, event, before), n in raw_counts.items() if before and n > 0}
    future = {(pair, event) for (pair, event, before), n in raw_counts.items() if not before and n > 0}
    positives = sorted(future - history)

    # Uniform case-control sampling from the *cutoff-defined* risk set.  The
    # seed, negative ratio and actual denominator are persisted for correction.
    rng = np.random.default_rng(args.seed)
    n_total = len(eligible_pairs) * len(event_vocab) - len(history)
    wanted = min(len(positives) * args.negative_ratio, n_total - len(positives))
    negatives: set[tuple[str, str]] = set()
    pairs = np.array(sorted(eligible_pairs), dtype=object)
    events = np.array(sorted(event_vocab), dtype=object)
    while len(negatives) < wanted:
        batch = max(10_000, (wanted - len(negatives)) * 2)
        for pi, ei in zip(rng.integers(len(pairs), size=batch), rng.integers(len(events), size=batch)):
            item = (pairs[pi], events[ei])
            if item not in history and item not in future:
                negatives.add(item)
            if len(negatives) == wanted:
                break
    rows = [(pair, event, 1) for pair, event in positives] + [(pair, event, 0) for pair, event in negatives]
    panel = pd.DataFrame(rows, columns=["pair_id", "meddra_pt", "target_first_pair_event_next_quarter"])
    panel["pair_history_count"] = panel["pair_id"].map(pair_history).fillna(0)
    panel["event_history_count"] = panel["meddra_pt"].map(history_event).fillna(0)
    panel[["drug1_id", "drug2_id"]] = panel["pair_id"].str.split("|", n=1, expand=True)

    drug_event = pd.read_parquet(DATA / f"nest_pv_drug_event_daily_{args.input_tag}.parquet")
    drug_event = drug_event.loc[drug_event["report_date"].le(cutoff)].groupby(["drug_id", "meddra_pt"])["report_count"].sum()
    for side in (1, 2):
        keys = pd.MultiIndex.from_arrays([panel[f"drug{side}_id"], panel["meddra_pt"]])
        panel[f"drug{side}_event_history_count"] = drug_event.reindex(keys, fill_value=0).to_numpy()

    static = pd.read_csv(DATA / "obspu_ddi_v2.csv").drop_duplicates("pair_id")
    static = static[["pair_id", "logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]]
    panel = panel.merge(static, on="pair_id", how="left").fillna(0)
    suffix = "" if (args.seed, args.negative_ratio) == (42, NEGATIVE_RATIO) else f"_r{args.negative_ratio}_s{args.seed}"
    suffix += f"_{args.output_tag}" if args.output_tag else ""
    panel.to_parquet(DATA / f"nest_first_event_discovery_{args.fold}{suffix}.parquet", index=False)
    audit = {
        "fold": args.fold, "cutoff": str(cutoff.date()), "horizon": str(horizon.date()),
        "event_vocabulary": TOP_EVENTS, "eligible_pairs": len(eligible_pairs),
        "history_triplets_excluded": len(history), "first_future_positives": len(positives),
        "sampled_negatives": len(negatives), "full_cutoff_risk_set": n_total,
        "negative_sampling": "uniform from cutoff-defined pair x event risk set", "seed": args.seed,
        "negative_ratio_requested": args.negative_ratio,
        "rows": len(panel), "positive_rate_sampled": float(panel.iloc[:, 2].mean()),
    }
    (DATA / f"nest_first_event_discovery_{args.fold}{suffix}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
