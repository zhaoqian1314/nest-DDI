#!/usr/bin/env python3
"""Conservative external concordance audit against cutoff-dated openFDA labels.

This is not a supervised external test: absence from a label section is unknown,
not a negative DDI label.  A pair is counted only when a label dated on or before
the 2025Q2 cutoff explicitly names the partner in ``drug_interactions`` text.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
CACHE = DATA / "openfda_label_interactions_20250630.json"
CUTOFF = "20250630"


def text_for_drug(name: str) -> dict[str, object]:
    query = f'openfda.generic_name:"{name}" AND effective_time:[* TO {CUTOFF}]'
    url = "https://api.fda.gov/drug/label.json?" + urlencode({"search": query, "limit": 100})
    try:
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return {"records": 0, "text": ""}
        return {"records": 0, "text": "", "error": f"HTTP {exc.code}"}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"records": 0, "text": "", "error": type(exc).__name__}
    texts = []
    for record in payload.get("results", []):
        texts.extend(record.get("drug_interactions", []))
    return {"records": len(payload.get("results", [])), "text": "\n".join(texts)}


def mentions(text: str, name: str) -> bool:
    normalized = str(name).strip().lower()
    if len(normalized) < 4:
        return False
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(normalized) + r"(?![a-z0-9])", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=300)
    args = parser.parse_args()
    panel = pd.read_parquet(DATA / "nest_ddi_prediction_panel_5q_release.parquet")
    test = panel.loc[panel["fold"].eq("2025q2"), ["drug1_id", "drug2_id"]].reset_index(drop=True)
    drugs = pd.read_csv(DATA / "ddinter_drugs.csv", dtype=str).set_index("id")["name"].to_dict()
    scores = np.load(OUT / "later_temporal_release_local_scores.npz")
    ranks = {
        "hgb": np.argsort(-scores["baseline"])[:args.top_k],
        "nest": np.argsort(-scores["nest"])[:args.top_k],
    }
    all_idx = np.unique(np.concatenate(list(ranks.values())))
    selected = test.iloc[all_idx].copy()
    names = set(selected["drug1_id"].map(drugs)).union(selected["drug2_id"].map(drugs))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    missing = sorted(name for name in names if name and name not in cache)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(text_for_drug, name): name for name in missing}
        for future in as_completed(futures):
            name = futures[future]
            cache[name] = future.result()
            time.sleep(0.05)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    rows = []
    for method, idx in ranks.items():
        hits, eligible, examples = 0, 0, []
        for row in test.iloc[idx].itertuples(index=False):
            left, right = drugs.get(row.drug1_id, ""), drugs.get(row.drug2_id, "")
            left_record, right_record = cache.get(left, {}), cache.get(right, {})
            source_available = bool(left_record.get("records", 0) or right_record.get("records", 0))
            if source_available:
                eligible += 1
            hit = mentions(str(left_record.get("text", "")), right) or mentions(str(right_record.get("text", "")), left)
            if hit:
                hits += 1
                if len(examples) < 10:
                    examples.append({"drug1": left, "drug2": right})
        rows.append({
            "ranker": method, "top_k": args.top_k, "pairs_with_label_source": eligible,
            "explicit_label_mentions": hits,
            "mention_rate_among_source_available": (hits / eligible) if eligible else None,
            "examples": examples,
        })
    result = {
        "cutoff": CUTOFF,
        "reference": "openFDA drug label endpoint, drug_interactions field",
        "design": "Top-K post-hoc descriptive concordance; no label absence is treated as negative and no model was selected with this reference",
        "matching": "case-insensitive exact partner-name boundary within a cutoff-dated interaction section",
        "limitations": [
            "openFDA is a current service and does not supply a historical snapshot of all labels",
            "generic-name matching misses synonyms, classes and combination products",
            "an explicit label mention is not a patient-level or causal validation",
        ],
        "results": rows,
    }
    (OUT / "openfda_label_concordance_2025q3.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
