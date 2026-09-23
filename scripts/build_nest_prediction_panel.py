#!/usr/bin/env python3
"""Create rolling-quarter prediction rows with static and Hawkes-inspired history features."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
HALF_LIVES = (7, 30, 90)
CUTOFFS = (
    ("2024q3", pd.Timestamp("2024-09-30"), pd.Timestamp("2024-12-31")),
    ("2024q4", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31")),
    ("2025q1", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")),
)


def add_pair_history(base: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    grouped = history.groupby("pair_id")
    total = grouped.agg(pair_total=("report_count", "sum"), pair_active_days=("report_date", "nunique"),
                        pair_last_date=("report_date", "max"))
    base = base.join(total, on="pair_id")
    base["pair_days_since"] = (cutoff - base["pair_last_date"]).dt.days
    for days in HALF_LIVES:
        recent = history.loc[history["report_date"] > cutoff - pd.Timedelta(days=days)].groupby("pair_id")["report_count"].sum()
        base[f"pair_recent_{days}"] = base["pair_id"].map(recent)
        decay = np.exp(-np.log(2) * (cutoff - history["report_date"]).dt.days / days) * history["report_count"]
        decayed = decay.groupby(history["pair_id"]).sum()
        base[f"pair_decay_{days}"] = base["pair_id"].map(decayed)
    return base


def add_drug_history(base: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    left = history[["drug1_id", "drug2_id", "report_date", "report_count"]].rename(
        columns={"drug1_id": "drug_id", "drug2_id": "partner"}
    )
    right = history[["drug1_id", "drug2_id", "report_date", "report_count"]].rename(
        columns={"drug2_id": "drug_id", "drug1_id": "partner"}
    )
    drug = pd.concat([left, right], ignore_index=True)
    total = drug.groupby("drug_id").agg(drug_total=("report_count", "sum"),
                                         drug_partners=("partner", "nunique"),
                                         drug_last_date=("report_date", "max"))
    for side in (1, 2):
        identifier = base[f"drug{side}_id"]
        base[f"drug{side}_total"] = identifier.map(total["drug_total"])
        base[f"drug{side}_partners"] = identifier.map(total["drug_partners"])
        base[f"drug{side}_days_since"] = (cutoff - identifier.map(total["drug_last_date"])).dt.days
    for days in HALF_LIVES:
        window = drug.loc[drug["report_date"] > cutoff - pd.Timedelta(days=days)]
        recent = window.groupby("drug_id")["report_count"].sum()
        decay = np.exp(-np.log(2) * (cutoff - drug["report_date"]).dt.days / days) * drug["report_count"]
        decayed = decay.groupby(drug["drug_id"]).sum()
        for side in (1, 2):
            identifier = base[f"drug{side}_id"]
            base[f"drug{side}_recent_{days}"] = identifier.map(recent)
            base[f"drug{side}_decay_{days}"] = identifier.map(decayed)
        a = base[f"drug1_decay_{days}"].fillna(0)
        b = base[f"drug2_decay_{days}"].fillna(0)
        base[f"cross_excitation_{days}"] = np.sqrt(a * b)
        base[f"activity_asymmetry_{days}"] = np.abs(np.log1p(a) - np.log1p(b))
        base[f"pair_share_{days}"] = base[f"pair_decay_{days}"].fillna(0) / (1.0 + np.sqrt(a * b))
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="4q")
    parser.add_argument("--output-tag", default="4q")
    parser.add_argument("--include-2025q2-fold", action="store_true")
    args = parser.parse_args()
    candidates = pd.read_csv(DATA / "obspu_ddi_v2.csv")
    candidates = candidates.drop_duplicates("pair_id").copy()
    keep = ["pair_id", "drug1_id", "drug2_id", "logdeg1", "logdeg2", "logdeg_min",
            "logdeg_max", "logdeg_product", "tanimoto"]
    candidates = candidates[keep]
    daily = pd.read_csv(DATA / f"nest_ddi_pair_daily_{args.input_tag}.csv", parse_dates=["report_date"])
    daily = daily[daily["pair_id"].isin(set(candidates["pair_id"]))].copy()
    first_date = daily.groupby("pair_id")["report_date"].min()

    panels = []
    audit = {"candidate_pairs": int(len(candidates)), "folds": {}}
    cutoffs = CUTOFFS + (("2025q2", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-09-30")),) if args.include_2025q2_fold else CUTOFFS
    for fold, cutoff, horizon in cutoffs:
        history = daily[daily["report_date"] <= cutoff].copy()
        future = daily[(daily["report_date"] > cutoff) & (daily["report_date"] <= horizon)].copy()
        base = candidates.copy()
        base = add_pair_history(base, history, cutoff)
        base = add_drug_history(base, history, cutoff)
        future_pairs = set(future["pair_id"])
        base["target_any_next_quarter"] = base["pair_id"].isin(future_pairs).astype("int8")
        base["target_first_next_quarter"] = (
            base["pair_id"].map(first_date).gt(cutoff) & base["pair_id"].map(first_date).le(horizon)
        ).astype("int8")
        base["at_risk_first"] = base["pair_total"].fillna(0).eq(0).astype("int8")
        base["fold"] = fold
        base["cutoff"] = cutoff
        base["horizon"] = horizon
        base = base.drop(columns=["pair_last_date"], errors="ignore")
        for col in base.columns:
            if col.endswith("days_since"):
                base[col] = base[col].fillna(9999).clip(0, 9999)
        numeric = base.select_dtypes(include=["number"]).columns
        base[numeric] = base[numeric].fillna(0)
        panels.append(base)
        audit["folds"][fold] = {
            "history_reports": int(history["report_count"].sum()),
            "future_reports": int(future["report_count"].sum()),
            "future_any_pairs": int(base["target_any_next_quarter"].sum()),
            "future_first_pairs": int(base["target_first_next_quarter"].sum()),
            "at_risk_pairs": int(base["at_risk_first"].sum()),
        }
        print(fold, audit["folds"][fold])
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(DATA / f"nest_ddi_prediction_panel_{args.output_tag}.parquet", index=False)
    audit["rows"] = int(len(panel))
    audit["columns"] = list(panel.columns)
    (DATA / f"nest_ddi_prediction_panel_audit_{args.output_tag}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
