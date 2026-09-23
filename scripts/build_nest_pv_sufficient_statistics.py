#!/usr/bin/env python3
"""Build case-level sufficient statistics for prospective DDI signal baselines.

The output contains counts, not patient-level records.  A report contributes at
most once to each drug, event, and drug-event tuple, after the same identity
mapping, latest-case-version rule and polypharmacy cap used by the NEST pilot.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from pathlib import Path

import pandas as pd

from build_nest_pilot import DEFAULT_QUARTERS, DATA, ascii_file, load_latest_case_metadata, norm


ROOT = Path(r"D:\BI\DDI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarters", nargs="+", default=list(DEFAULT_QUARTERS))
    parser.add_argument("--output-tag", default="4q")
    parser.add_argument("--version-policy", choices=("global_latest", "release_local"), default="global_latest")
    args = parser.parse_args()
    quarters = tuple(args.quarters)
    drugs = pd.read_csv(DATA / "ddinter_drugs.csv", dtype=str)
    name_map = {norm(name): drug_id for drug_id, name in zip(drugs["id"], drugs["name"])}
    metadata = load_latest_case_metadata(quarters, args.version_policy)

    case_daily: Counter = Counter()
    drug_daily: Counter = Counter()
    event_daily: Counter = Counter()
    drug_event_daily: Counter = Counter()
    audit = {}
    for quarter in quarters:
        meta = metadata.loc[metadata["quarter"].eq(quarter)].set_index("primaryid")
        valid_ids = set(meta.index)
        drug_sets: dict[str, set[str]] = defaultdict(set)
        for chunk in pd.read_csv(
            ascii_file(quarter, "DRUG"), sep="$", usecols=["primaryid", "drugname"],
            dtype=str, chunksize=150_000,
        ):
            chunk = chunk[chunk["primaryid"].isin(valid_ids)]
            for row in chunk.itertuples(index=False):
                drug_id = name_map.get(norm(row.drugname))
                if drug_id:
                    drug_sets[row.primaryid].add(drug_id)

        reactions: dict[str, set[str]] = defaultdict(set)
        for chunk in pd.read_csv(
            ascii_file(quarter, "REAC"), sep="$", usecols=["primaryid", "pt"],
            dtype=str, chunksize=200_000,
        ):
            chunk = chunk[chunk["primaryid"].isin(valid_ids)]
            for row in chunk.itertuples(index=False):
                if row.pt and str(row.pt) != "nan":
                    reactions[row.primaryid].add(str(row.pt).strip())

        skipped = 0
        retained = 0
        for primaryid, row in meta.iterrows():
            date = pd.Timestamp(row.report_date).date().isoformat()
            # The background cohort is every valid report, including reports
            # without a mapped drug or a recorded reaction.
            case_daily[(date, quarter)] += 1
            drug_ids = drug_sets.get(primaryid, set())
            if len(drug_ids) > 20:
                skipped += 1
                continue
            events = reactions.get(primaryid, set())
            if drug_ids:
                retained += 1
            for event in events:
                event_daily[(event, date, quarter)] += 1
            for drug_id in drug_ids:
                drug_daily[(drug_id, date, quarter)] += 1
                for event in events:
                    drug_event_daily[(drug_id, event, date, quarter)] += 1
        audit[quarter] = {
            "valid_cases": int(len(meta)),
            "mapped_drug_cases_retained": retained,
            "polypharmacy_cases_skipped": skipped,
        }
        print(quarter, audit[quarter], flush=True)

    def save(counter: Counter, columns: list[str], filename: str) -> None:
        frame = pd.DataFrame([(*key, value) for key, value in counter.items()], columns=columns + ["report_count"])
        frame["report_date"] = pd.to_datetime(frame["report_date"])
        frame.sort_values(["report_date"] + columns[:-2], inplace=True)
        frame.to_parquet(DATA / filename, index=False)
        print(filename, len(frame), flush=True)

    save(case_daily, ["report_date", "quarter"], f"nest_pv_case_daily_{args.output_tag}.parquet")
    save(drug_daily, ["drug_id", "report_date", "quarter"], f"nest_pv_drug_daily_{args.output_tag}.parquet")
    save(event_daily, ["meddra_pt", "report_date", "quarter"], f"nest_pv_event_daily_{args.output_tag}.parquet")
    save(drug_event_daily, ["drug_id", "meddra_pt", "report_date", "quarter"], f"nest_pv_drug_event_daily_{args.output_tag}.parquet")
    (DATA / f"nest_pv_sufficient_statistics_audit_{args.output_tag}.json").write_text(
        pd.Series(audit).to_json(indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
