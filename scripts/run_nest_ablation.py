#!/usr/bin/env python3
"""Feature-group ablation for NEST-DDI on the frozen recurrence panel.

Reuses the exact frozen pipeline of run_nest_final_comparison.any_task():
same folds, same background model, same HistGradientBoosting setting, same seed.
Only the feature subset is altered.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nest_baselines import DRUG_HISTORY, PAIR_HISTORY, RAW_CROSS, STATIC  # noqa: E402
from run_nest_final_comparison import metrics  # noqa: E402
from run_nest_original import engineer, fit_background  # noqa: E402

ROOT = Path(r"D:\BI\DDI")
OUT = ROOT / "results" / "nest_ddi"

HGB_PARAMS = dict(
    learning_rate=0.06, max_iter=260, max_leaf_nodes=31, min_samples_leaf=35,
    l2_regularization=1.5, random_state=42,
)


def drop_static(name: str) -> bool:
    """Static structural background and every feature derived from it."""
    return (
        name in STATIC
        or name == "background_logit"
        or name.startswith("structure_activity_")
        or name.startswith("degree_activity_")
    )


def drop_decay(name: str) -> bool:
    """All decayed-history terms and every feature derived from decay columns."""
    return (
        "decay" in name
        or "accel" in name
        or name.startswith("cascade_excess_")
    )


def drop_drug(name: str) -> bool:
    """Drug-level analogue terms and every feature derived from drug-level columns."""
    return (
        name.startswith("drug")
        or name.startswith("cascade_excess_")
        or name in ("partner_geomean", "partner_min", "partner_asymmetry",
                    "observation_opportunity")
    )


def run_variant(train, test, y_train, y_test, drop=None, only=None, use_background=True):
    x_train, features = engineer(train)
    x_test, _ = engineer(test)
    background_score = None
    if use_background:
        base_test, background = fit_background(x_train, y_train, x_test)
        base_train = background.predict_proba(x_train[STATIC])[:, 1]
        x_train["background_logit"] = logit(np.clip(base_train, 1e-5, 1 - 1e-5))
        x_test["background_logit"] = logit(np.clip(base_test, 1e-5, 1 - 1e-5))
        features = features + ["background_logit"]
        background_score = base_test
    kept = [f for f in features if (only is None or only(f)) and not (drop is not None and drop(f))]
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(x_train[kept], y_train)
    score = model.predict_proba(x_test[kept])[:, 1]
    return kept, background_score, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tag", default="4q_release")
    ap.add_argument("--output-tag", default="ablation_release_local")
    args = ap.parse_args()

    panel = pd.read_parquet(ROOT / "data" / "processed" / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    train = panel[panel["fold"].isin(["2024q3", "2024q4"])]
    test = panel[panel["fold"].eq("2025q1")]
    y_train = train["target_any_next_quarter"].to_numpy()
    y_test = test["target_any_next_quarter"].to_numpy()
    print(f"train {len(train)} test {len(test)} positives {int(y_test.sum())}", flush=True)

    variants = {
        "full": dict(),
        "no_decay": dict(drop=drop_decay),
        "no_static": dict(drop=drop_static, use_background=False),
        "no_drug_analogues": dict(drop=drop_drug),
        "static_only": dict(only=lambda n: n in STATIC, use_background=False),
    }

    results, dropped = {}, {}
    for name, kwargs in variants.items():
        t0 = time.time()
        kept, background_score, score = run_variant(train, test, y_train, y_test, **kwargs)
        block = metrics(y_test, score)
        block["n_features"] = len(kept)
        block["elapsed_s"] = round(time.time() - t0, 1)
        if background_score is not None:
            block["background_aupr"] = float(metrics(y_test, background_score)["aupr"])
        results[name] = block
        _, all_features = engineer(train)
        universe = all_features + (["background_logit"] if kwargs.get("use_background", True) else [])
        dropped[name] = sorted(set(universe) - set(kept))
        print(f"{name:20s} aupr={block['aupr']:.5f} auroc={block['auroc']:.5f} "
              f"({block['n_features']} feats, {block['elapsed_s']}s)", flush=True)

    payload = {
        "protocol": "release-local; train 2024q3+2024q4; test 2025q1->2025q2 (frozen)",
        "panel": f"nest_ddi_prediction_panel_{args.input_tag}.parquet",
        "target": "target_any_next_quarter",
        "seed": 42,
        "hgb_params": HGB_PARAMS,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positives": int(y_test.sum()),
        "variants": results,
        "dropped_columns": dropped,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.output_tag}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
