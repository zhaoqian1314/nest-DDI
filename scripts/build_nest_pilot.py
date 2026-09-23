#!/usr/bin/env python3
"""Build a leakage-controlled four-quarter NEST-DDI pilot from FAERS ASCII files."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
DEFAULT_QUARTERS = ("2024q3", "2024q4", "2025q1", "2025q2")


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def ascii_file(quarter: str, prefix: str) -> Path:
    matches = list((DATA / f"faers_{quarter}" / "ASCII").glob(f"{prefix}*.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} file for {quarter}, found {matches}")
    return matches[0]


def load_latest_case_metadata(quarters: tuple[str, ...], version_policy: str = "global_latest") -> pd.DataFrame:
    frames = []
    for quarter in quarters:
        demo = pd.read_csv(
            ascii_file(quarter, "DEMO"),
            sep="$",
            usecols=["primaryid", "caseid", "caseversion", "event_dt", "fda_dt", "rept_dt",
                     "occp_cod", "reporter_country"],
            dtype=str,
        )
        demo["quarter"] = quarter
        frames.append(demo)
    meta = pd.concat(frames, ignore_index=True)
    meta["caseversion_num"] = pd.to_numeric(meta["caseversion"], errors="coerce").fillna(-1)
    if version_policy == "global_latest":
        meta = meta.sort_values(["caseid", "caseversion_num", "primaryid"]).drop_duplicates("caseid", keep="last")
    elif version_policy == "release_local":
        meta = meta.sort_values(["quarter", "caseid", "caseversion_num", "primaryid"]).drop_duplicates(
            ["quarter", "caseid"], keep="last"
        )
    else:
        raise ValueError(f"unknown version policy: {version_policy}")
    for column in ("event_dt", "fda_dt", "rept_dt"):
        meta[column + "_parsed"] = pd.to_datetime(meta[column], format="%Y%m%d", errors="coerce")
    # Report-process time must follow report receipt, not the clinical event date.
    meta["report_date"] = meta["rept_dt_parsed"].fillna(meta["fda_dt_parsed"])
    quarter_period = meta["quarter"].str.upper().str.replace("Q", "Q", regex=False)
    expected = pd.PeriodIndex(quarter_period, freq="Q")
    in_release_quarter = (meta["report_date"] >= expected.start_time) & (meta["report_date"] <= expected.end_time)
    return meta.loc[in_release_quarter].dropna(subset=["primaryid", "report_date"])


def build_quarter_events(
    quarter: str,
    valid_ids: set[str],
    metadata: pd.DataFrame,
    name_map: dict[str, str],
    max_drugs_per_case: int,
    pair_only: bool = False,
) -> tuple[Counter, Counter, dict[str, int]]:
    drug_roles: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for chunk in pd.read_csv(
        ascii_file(quarter, "DRUG"), sep="$", usecols=["primaryid", "drugname", "role_cod"],
        dtype=str, chunksize=150_000,
    ):
        chunk = chunk[chunk["primaryid"].isin(valid_ids)]
        for row in chunk.itertuples(index=False):
            drug_id = name_map.get(norm(row.drugname))
            if drug_id:
                drug_roles[row.primaryid][drug_id].add(str(row.role_cod or "UNK"))

    reactions: dict[str, set[str]] = defaultdict(set)
    if not pair_only:
        for chunk in pd.read_csv(
            ascii_file(quarter, "REAC"), sep="$", usecols=["primaryid", "pt"], dtype=str,
            chunksize=200_000,
        ):
            chunk = chunk[chunk["primaryid"].isin(valid_ids)]
            for row in chunk.itertuples(index=False):
                if row.pt and str(row.pt) != "nan":
                    reactions[row.primaryid].add(str(row.pt).strip())

    meta = metadata.set_index("primaryid")
    pair_daily: Counter = Counter()
    event_daily: Counter = Counter()
    audit = Counter()
    for primaryid, roles in drug_roles.items():
        drugs = sorted(roles)
        if len(drugs) < 2:
            continue
        if len(drugs) > max_drugs_per_case:
            audit["cases_skipped_polypharmacy_cap"] += 1
            continue
        if primaryid not in meta.index:
            continue
        date = pd.Timestamp(meta.at[primaryid, "report_date"]).date().isoformat()
        events = reactions.get(primaryid, set())
        for drug_a, drug_b in combinations(drugs, 2):
            pair_id = f"{drug_a}|{drug_b}"
            roles_a = ";".join(sorted(roles[drug_a]))
            roles_b = ";".join(sorted(roles[drug_b]))
            pair_daily[(pair_id, drug_a, drug_b, date, quarter)] += 1
            if not pair_only:
                for event in events:
                    event_daily[(pair_id, drug_a, drug_b, event, date, quarter, roles_a, roles_b)] += 1
        audit["mapped_cases_with_pairs"] += 1
        audit["mapped_case_reactions"] += len(events)
    audit["mapped_drug_case_ids"] = len(drug_roles)
    audit["reaction_case_ids"] = len(reactions)
    return pair_daily, event_daily, dict(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarters", nargs="+", default=list(DEFAULT_QUARTERS))
    parser.add_argument("--max-drugs-per-case", type=int, default=20)
    parser.add_argument("--top-events", type=int, default=200)
    parser.add_argument("--min-pair-event-reports", type=int, default=3)
    parser.add_argument("--output-tag", default="4q", help="Version tag used in output file names.")
    parser.add_argument("--pair-only", action="store_true", help="Skip reaction/event aggregation for pair-level tests.")
    parser.add_argument("--version-policy", choices=("global_latest", "release_local"), default="global_latest")
    args = parser.parse_args()
    quarters = tuple(args.quarters)

    drugs = pd.read_csv(DATA / "ddinter_drugs.csv", dtype=str)
    name_map = {norm(name): drug_id for drug_id, name in zip(drugs["id"], drugs["name"])}
    metadata = load_latest_case_metadata(quarters, args.version_policy)
    meta_by_quarter = {q: metadata.loc[metadata["quarter"].eq(q)].copy() for q in quarters}

    pair_counts: Counter = Counter()
    event_counts: Counter = Counter()
    quarter_audits = {}
    for quarter in quarters:
        quarter_meta = meta_by_quarter[quarter]
        valid_ids = set(quarter_meta["primaryid"])
        pairs, events, audit = build_quarter_events(
            quarter, valid_ids, quarter_meta, name_map, args.max_drugs_per_case, args.pair_only
        )
        pair_counts.update(pairs)
        event_counts.update(events)
        quarter_audits[quarter] = audit
        print(quarter, audit)

    pair_rows = [(*key, value) for key, value in pair_counts.items()]
    pair_daily = pd.DataFrame(
        pair_rows,
        columns=["pair_id", "drug1_id", "drug2_id", "report_date", "quarter", "report_count"],
    )
    pair_daily["report_date"] = pd.to_datetime(pair_daily["report_date"])
    pair_daily = pair_daily.sort_values(["report_date", "pair_id"])
    pair_daily.to_csv(DATA / f"nest_ddi_pair_daily_{args.output_tag}.csv", index=False)

    if args.pair_only:
        audit = {
            "quarters": list(quarters), "latest_case_versions": int(len(metadata)),
            "pair_daily_rows": int(len(pair_daily)), "unique_pairs": int(pair_daily["pair_id"].nunique()),
            "date_min": str(pair_daily["report_date"].min().date()), "date_max": str(pair_daily["report_date"].max().date()),
            "max_drugs_per_case": args.max_drugs_per_case, "pair_only": True,
            "version_policy": args.version_policy,
            "quarter_audits": quarter_audits,
        }
        (DATA / f"nest_ddi_pilot_audit_{args.output_tag}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return

    event_frequency = Counter()
    pair_event_frequency = Counter()
    for key, value in event_counts.items():
        pair_id, _, _, event, *_ = key
        event_frequency[event] += value
        pair_event_frequency[(pair_id, event)] += value
    keep_events = {event for event, _ in event_frequency.most_common(args.top_events)}
    event_rows = [
        (*key, value) for key, value in event_counts.items()
        if key[3] in keep_events and pair_event_frequency[(key[0], key[3])] >= args.min_pair_event_reports
    ]
    event_daily = pd.DataFrame(
        event_rows,
        columns=["pair_id", "drug1_id", "drug2_id", "meddra_pt", "report_date", "quarter",
                 "role1", "role2", "report_count"],
    )
    event_daily["report_date"] = pd.to_datetime(event_daily["report_date"])
    event_daily = event_daily.sort_values(["report_date", "pair_id", "meddra_pt"])
    event_daily.to_csv(DATA / f"nest_ddi_event_daily_{args.output_tag}.csv", index=False)

    audit = {
        "quarters": list(quarters),
        "latest_case_versions": int(len(metadata)),
        "pair_daily_rows": int(len(pair_daily)),
        "unique_pairs": int(pair_daily["pair_id"].nunique()),
        "event_daily_rows": int(len(event_daily)),
        "unique_pair_events": int(event_daily[["pair_id", "meddra_pt"]].drop_duplicates().shape[0]),
        "date_min": str(pair_daily["report_date"].min().date()),
        "date_max": str(pair_daily["report_date"].max().date()),
        "max_drugs_per_case": args.max_drugs_per_case,
        "top_events": args.top_events,
        "min_pair_event_reports": args.min_pair_event_reports,
        "version_policy": args.version_policy,
        "quarter_audits": quarter_audits,
    }
    (DATA / f"nest_ddi_pilot_audit_{args.output_tag}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
