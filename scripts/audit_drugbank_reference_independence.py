#!/usr/bin/env python3
"""Audit whether bundled DrugBank labels can serve as an independent NEST reference.

The output is an independence/provenance audit, not a predictive validation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data"
OUT = ROOT / "results" / "nest_ddi"


def norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def main() -> None:
    ddinter = pd.read_csv(DATA / "processed" / "ddinter_drugs.csv", dtype=str)
    deepddi = pd.read_csv(DATA / "deepddi-master" / "database" / "drug_info_combined.csv", dtype=str)
    mapping = deepddi.dropna(subset=["Name", "Drug name"]).drop_duplicates("Name")
    by_name = dict(zip(mapping["Name"].map(norm), mapping["Drug name"]))
    ddinter["drugbank_id"] = ddinter["name"].map(norm).map(by_name)
    known = pd.read_csv(DATA / "deepddi-master" / "data" / "DrugBank_known_ddi.txt", sep="\t", dtype=str)
    known_pairs = {
        "|".join(sorted((row.drug1, row.drug2)))
        for row in known.itertuples(index=False) if str(row.Label) == "1"
    }
    pairs = pd.read_csv(DATA / "processed" / "ddinter_pairs.csv", dtype=str)
    id_to_db = dict(zip(ddinter["id"], ddinter["drugbank_id"]))
    pairs["db1"] = pairs["drug1_id"].map(id_to_db)
    pairs["db2"] = pairs["drug2_id"].map(id_to_db)
    mapped_pairs = pairs.dropna(subset=["db1", "db2"]).copy()
    mapped_pairs["db_pair"] = mapped_pairs.apply(lambda x: "|".join(sorted((x.db1, x.db2))), axis=1)
    overlap = int(mapped_pairs["db_pair"].isin(known_pairs).sum())
    result = {
        "reference_file": "data/deepddi-master/data/DrugBank_known_ddi.txt",
        "reference_type": "curated positive DDI labels; no report date or mechanism subtype retained in the bundled file",
        "ddinter_drugs": int(len(ddinter)),
        "ddinter_drugs_mapped_to_drugbank_by_normalized_name": int(ddinter["drugbank_id"].notna().sum()),
        "ddinter_pairs": int(len(pairs)),
        "ddinter_pairs_mapped_to_drugbank": int(len(mapped_pairs)),
        "mapped_ddinter_pairs_present_as_drugbank_positive": overlap,
        "independence_assessment": "not suitable as an independent external or mechanistic validation because candidate-pool/source overlap and label timing/provenance cannot be excluded; use only as a descriptive concordance resource after an independently versioned reference is obtained",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drugbank_reference_independence_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
