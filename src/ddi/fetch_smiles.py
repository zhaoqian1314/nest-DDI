#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch canonical SMILES for all DDInter drugs from PubChem.

Rate-limited (sleeps between requests). Resumable: writes incrementally and
skips CIDs already fetched. Run in background.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

DRUGS = r"D:\BI\DDI\data\processed\ddinter_drugs.csv"
OUT = r"D:\BI\DDI\data\processed\ddinter_smiles.json"

USER_AGENT = "DDI-Research/1.0 (drug-drug interaction research)"


def fetch_smiles(name):
    """Return (cid, smiles) or (None, None)."""
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{urllib.parse.quote(name)}/property/SMILES,CanonicalSMILES,ConnectivitySMILES/JSON")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        props = d["PropertyTable"]["Properties"][0]
        cid = props.get("CID")
        smi = (props.get("CanonicalSMILES") or props.get("SMILES")
               or props.get("ConnectivitySMILES"))
        return cid, smi
    except Exception:
        return None, None


def main():
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    drugs = pd.read_csv(DRUGS)
    names = drugs["name"].tolist()
    # load existing progress
    result = {}
    if os.path.exists(OUT):
        result = json.load(open(OUT, encoding="utf-8"))

    todo = [n for n in names[start:] if n not in result]
    print(f"start={start}, total={len(names)}, already done={len(result)}, todo={len(todo)}", flush=True)

    ok = 0
    for i, name in enumerate(todo):
        cid, smi = fetch_smiles(name)
        if smi:
            result[name] = {"cid": cid, "smiles": smi}
            ok += 1
        else:
            result[name] = {"cid": None, "smiles": None}
        if (i + 1) % 100 == 0:
            json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  progress {i+1}/{len(todo)} (ok={ok})", flush=True)
        time.sleep(0.35)  # ~3 req/s, under PubChem limit

    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    n_smiles = sum(1 for v in result.values() if v["smiles"])
    print(f"DONE. total={len(result)}, with_smiles={n_smiles} ({n_smiles/len(result)*100:.1f}%)")


if __name__ == "__main__":
    main()
