#!/usr/bin/env python3
"""Audit FAERS role combinations and the NEST polypharmacy cap by quarter."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combinations: Counter[tuple[str, str, str]] = Counter()
    quarter_totals: Counter[str] = Counter()
    for chunk in pd.read_csv(
        DATA / "nest_ddi_event_daily_4q.csv",
        usecols=["quarter", "role1", "role2", "report_count"],
        chunksize=500_000,
        dtype={"quarter": str, "role1": str, "role2": str},
    ):
        chunk["role1"] = chunk["role1"].fillna("UNK")
        chunk["role2"] = chunk["role2"].fillna("UNK")
        grouped = chunk.groupby(["quarter", "role1", "role2"])["report_count"].sum()
        for (quarter, role1, role2), count in grouped.items():
            combinations[(quarter, role1, role2)] += int(count)
            quarter_totals[quarter] += int(count)

    pilot = json.loads((DATA / "nest_ddi_pilot_audit.json").read_text(encoding="utf-8"))
    output = {"role_combinations": {}, "polypharmacy_cap": {}, "scope_note":
              "Role audit uses the restricted pair-event pilot, not an exposure-confirmed cohort."}
    for quarter in pilot["quarters"]:
        rows = [
            {"role1": r1, "role2": r2, "reports": n,
             "share_of_pilot_reports": n / quarter_totals[quarter]}
            for (q, r1, r2), n in combinations.items() if q == quarter
        ]
        rows.sort(key=lambda x: x["reports"], reverse=True)
        output["role_combinations"][quarter] = {
            "pilot_reports": quarter_totals[quarter],
            "top_combinations": rows[:12],
        }
        q_audit = pilot["quarter_audits"][quarter]
        valid = int(q_audit["mapped_cases_with_pairs"] + q_audit["cases_skipped_polypharmacy_cap"])
        skipped = int(q_audit["cases_skipped_polypharmacy_cap"])
        output["polypharmacy_cap"][quarter] = {
            "mapped_pair_eligible_cases_before_cap": valid,
            "skipped_cases": skipped,
            "skipped_share": skipped / valid if valid else 0.0,
        }
    (OUT / "role_polypharmacy_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
