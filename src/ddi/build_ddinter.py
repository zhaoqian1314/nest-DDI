#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the canonical DDI dataset from raw DDInter CSVs.

Merges the 8 ATC-code CSV files, dedups pairs (order-normalized), and
writes a clean interaction table + a unique drug list.

Outputs (in data/processed/):
  ddinter_pairs.csv   : deduped, order-normalized drug pairs with severity
  ddinter_drugs.csv   : unique drugs with DDInterID and name
"""
import glob
import os
import pandas as pd

RAW = r"D:\BI\DDI\data\raw"
OUT = r"D:\BI\DDI\data\processed"
os.makedirs(OUT, exist_ok=True)

SEV_ORDER = {"Minor": 0, "Moderate": 1, "Major": 2, "Unknown": 3}


def main():
    files = sorted(glob.glob(os.path.join(RAW, "ddinter_code_*.csv")))
    assert files, "no DDInter CSV files found"
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["DDInterID_A", "DDInterID_B", "Drug_A", "Drug_B"])
    df = df.drop_duplicates(subset=["DDInterID_A", "DDInterID_B", "Drug_A", "Drug_B"])

    # severity as ordinal
    df["sev"] = df["Level"].map(SEV_ORDER).fillna(3).astype(int)

    # order-normalize pairs: canonical key = sorted pair of IDs
    a = df["DDInterID_A"].astype(str)
    b = df["DDInterID_B"].astype(str)
    swap = a > b
    aid = a.where(~swap, b)
    bid = b.where(~swap, a)
    aname = df["Drug_A"].where(~swap, df["Drug_B"])
    bname = df["Drug_B"].where(~swap, df["Drug_A"])
    asev = df["sev"].where(~swap, df["sev"])

    pairs = pd.DataFrame({
        "drug1_id": aid.values, "drug1": aname.values,
        "drug2_id": bid.values, "drug2": bname.values,
        "severity": asev.values, "severity_label": df["Level"].values,
    })
    pairs = pairs.drop_duplicates(subset=["drug1_id", "drug2_id"])
    pairs = pairs.sort_values("drug1_id").reset_index(drop=True)
    pairs.to_csv(os.path.join(OUT, "ddinter_pairs.csv"), index=False)

    # unique drugs
    d1 = pairs[["drug1_id", "drug1"]].rename(columns={"drug1_id": "id", "drug1": "name"})
    d2 = pairs[["drug2_id", "drug2"]].rename(columns={"drug2_id": "id", "drug2": "name"})
    drugs = pd.concat([d1, d2]).drop_duplicates("id").sort_values("id").reset_index(drop=True)
    drugs.to_csv(os.path.join(OUT, "ddinter_drugs.csv"), index=False)

    print(f"raw rows: {len(df)}")
    print(f"deduped pairs: {len(pairs)}")
    print(f"unique drugs: {len(drugs)}")
    print(f"severity distribution: {pairs['severity_label'].value_counts().to_dict()}")
    print(f"pairs with severity>=Moderate(1): {(pairs['severity']>=1).sum()}")
    print(f"pairs with severity==Major(2): {(pairs['severity']==2).sum()}")


if __name__ == "__main__":
    main()
