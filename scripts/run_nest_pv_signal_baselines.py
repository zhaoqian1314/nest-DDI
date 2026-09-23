#!/usr/bin/env python3
"""Prospective Ω/additive/multiplicative DDI-event signal baselines.

This is a pharmacovigilance signal-ranking task: among drug-pair--MedDRA
triplets with >=3 historical co-reports, rank triplets that recur in the next
quarter.  It is deliberately reported separately from the pair-emergence task.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
CUTOFFS = (
    ("2024q3", pd.Timestamp("2024-09-30"), pd.Timestamp("2024-12-31")),
    ("2024q4", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-03-31")),
    ("2025q1", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")),
)


def aggregate(frame: pd.DataFrame, cutoff: pd.Timestamp, by: list[str]) -> pd.Series:
    history = frame.loc[frame["report_date"].le(cutoff)]
    if not by:
        return pd.Series([history["report_count"].sum()])
    return history.groupby(by)["report_count"].sum()


def scores_at_cutoff(
    cutoff: pd.Timestamp,
    pair_daily: pd.DataFrame,
    pair_event_daily: pd.DataFrame,
    case_daily: pd.DataFrame,
    drug_daily: pd.DataFrame,
    event_daily: pd.DataFrame,
    drug_event_daily: pd.DataFrame,
) -> pd.DataFrame:
    pair = aggregate(pair_daily, cutoff, ["pair_id"])
    pair_event = aggregate(pair_event_daily, cutoff, ["pair_id", "meddra_pt"]).rename("n111")
    base = pair_event.loc[pair_event.ge(3)].reset_index()
    base["n11"] = base["pair_id"].map(pair).fillna(0.0)
    ids = base["pair_id"].str.split("|", n=1, expand=True)
    base["drug1_id"], base["drug2_id"] = ids[0], ids[1]

    total_cases = float(aggregate(case_daily, cutoff, []).sum())
    drug = aggregate(drug_daily, cutoff, ["drug_id"])
    event = aggregate(event_daily, cutoff, ["meddra_pt"])
    drug_event = aggregate(drug_event_daily, cutoff, ["drug_id", "meddra_pt"])
    for side in (1, 2):
        drug_id = base[f"drug{side}_id"]
        base[f"n{side}"] = drug_id.map(drug).fillna(0.0)
        keys = pd.MultiIndex.from_arrays([drug_id, base["meddra_pt"]])
        base[f"n{side}event"] = drug_event.reindex(keys, fill_value=0.0).to_numpy()
    base["nevent"] = base["meddra_pt"].map(event).fillna(0.0)

    # Four mutually exclusive exposure groups for each pair-event triplet.
    n11e = base["n111"].to_numpy(float)
    n11 = base["n11"].to_numpy(float)
    n1 = base["n1"].to_numpy(float)
    n2 = base["n2"].to_numpy(float)
    n1e = base["n1event"].to_numpy(float)
    n2e = base["n2event"].to_numpy(float)
    ne = base["nevent"].to_numpy(float)
    n10, n01 = n1 - n11, n2 - n11
    n10e, n01e = n1e - n11e, n2e - n11e
    n00 = total_cases - n1 - n2 + n11
    n00e = ne - n1e - n2e + n11e
    eps = 1e-12
    f11 = n11e / np.maximum(n11, eps)
    f10 = np.clip(n10e / np.maximum(n10, eps), 0, 1)
    f01 = np.clip(n01e / np.maximum(n01, eps), 0, 1)
    f00 = np.clip(n00e / np.maximum(n00, eps), 0, 1)

    def odds(p: np.ndarray) -> np.ndarray:
        return p / np.maximum(1 - p, eps)
    # Norén's additive baseline expressed on the odds scale, with the required
    # max restrictions that prevent a single drug appearing protective.
    g11 = 1 - 1 / (
        np.maximum(odds(f10), odds(f00))
        + np.maximum(odds(f01), odds(f00))
        - odds(f00) + 1
    )
    expected = g11 * n11
    omega = np.log2((n11e + 0.5) / (expected + 0.5))
    omega025 = omega - norm.ppf(0.975) / (np.log(2) * np.sqrt(np.maximum(n11e, eps)))
    additive = f11 - f10 - f01 + f00
    multiplicative = (f11 * f00) / np.maximum(f10 * f01, eps)
    base_rate = ne / total_cases
    prr_pair = f11 / np.maximum(base_rate, eps)
    prr_1 = (n1e / np.maximum(n1, eps)) / np.maximum(base_rate, eps)
    prr_2 = (n2e / np.maximum(n2, eps)) / np.maximum(base_rate, eps)
    combo_rr = prr_pair / np.maximum(np.maximum(prr_1, prr_2), eps)
    base["historical_count"] = n11e
    base["omega"] = omega
    base["omega025"] = omega025
    base["additive_risk_difference"] = additive
    base["multiplicative_interaction"] = multiplicative
    base["combination_risk_ratio"] = combo_rr
    return base.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    result = {
        "auroc": float(roc_auc_score(y, score)),
        "aupr": float(average_precision_score(y, score)),
        "n": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
    }
    for k in (100, 500, 1000):
        result[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair_daily = pd.read_csv(DATA / "nest_ddi_pair_daily_4q.csv", parse_dates=["report_date"])
    pair_event_daily = pd.read_csv(DATA / "nest_ddi_event_daily_4q.csv", parse_dates=["report_date"])
    case_daily = pd.read_parquet(DATA / "nest_pv_case_daily_4q.parquet")
    drug_daily = pd.read_parquet(DATA / "nest_pv_drug_daily_4q.parquet")
    event_daily = pd.read_parquet(DATA / "nest_pv_event_daily_4q.parquet")
    drug_event_daily = pd.read_parquet(DATA / "nest_pv_drug_event_daily_4q.parquet")
    all_results: dict[str, dict] = {}
    for fold, cutoff, horizon in CUTOFFS:
        frame = scores_at_cutoff(cutoff, pair_daily, pair_event_daily, case_daily, drug_daily, event_daily, drug_event_daily)
        future = pair_event_daily.loc[
            pair_event_daily["report_date"].gt(cutoff) & pair_event_daily["report_date"].le(horizon),
            ["pair_id", "meddra_pt"],
        ].drop_duplicates()
        future_index = pd.MultiIndex.from_frame(future)
        frame["target_recurrence_next_quarter"] = pd.MultiIndex.from_frame(frame[["pair_id", "meddra_pt"]]).isin(future_index).astype("int8")
        y = frame["target_recurrence_next_quarter"].to_numpy()
        results = {name: evaluate(y, frame[name].to_numpy()) for name in (
            "historical_count", "omega", "omega025", "additive_risk_difference",
            "multiplicative_interaction", "combination_risk_ratio",
        )}
        all_results[fold] = {"cutoff": str(cutoff.date()), "horizon": str(horizon.date()), "results": results}
        print(fold, {name: round(item["aupr"], 5) for name, item in results.items()}, flush=True)
        if fold == "2025q1":
            frame.to_parquet(OUT / "pv_signal_scores_2025q1.parquet", index=False)
    output = {
        "task": "prospective recurrence of historically observed pair--MedDRA triplets",
        "candidate_rule": "triplets with at least three historical pair-event reports",
        "warning": "This is a pharmacovigilance signal-ranking task, not clinical incidence or causal DDI confirmation.",
        "folds": all_results,
    }
    (OUT / "pv_signal_baseline_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
