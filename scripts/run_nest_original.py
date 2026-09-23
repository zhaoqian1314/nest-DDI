#!/usr/bin/env python3
"""Train and tune the NEST-DDI additive background/cascade ranking model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(r"D:\BI\DDI")
PANEL = ROOT / "data" / "processed" / "nest_ddi_prediction_panel_4q.parquet"
OUT = ROOT / "results" / "nest_ddi"
STATIC = ["logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]
RAW_DYNAMIC = [
    "pair_total", "pair_active_days", "pair_days_since",
    *[f"pair_{kind}_{d}" for d in (7, 30, 90) for kind in ("recent", "decay")],
    *[f"drug{s}_{name}" for s in (1, 2) for name in (
        "total", "partners", "days_since", "recent_7", "decay_7", "recent_30", "decay_30",
        "recent_90", "decay_90")],
    *[f"{name}_{d}" for d in (7, 30, 90) for name in (
        "cross_excitation", "activity_asymmetry", "pair_share")],
]


def engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame[STATIC + RAW_DYNAMIC].astype("float32").copy().fillna(0)
    eps = 1e-4
    for days, previous in ((7, 30), (30, 90)):
        for entity in ("pair", "drug1", "drug2"):
            x[f"{entity}_accel_{days}_{previous}"] = np.log1p(x[f"{entity}_decay_{days}"]) - np.log1p(x[f"{entity}_decay_{previous}"])
    for days in (7, 30, 90):
        a = np.log1p(x[f"drug1_decay_{days}"])
        b = np.log1p(x[f"drug2_decay_{days}"])
        pair = np.log1p(x[f"pair_decay_{days}"])
        x[f"cascade_excess_{days}"] = pair - 0.5 * (a + b)
        x[f"drug_min_decay_{days}"] = np.minimum(a, b)
        x[f"drug_max_decay_{days}"] = np.maximum(a, b)
        x[f"drug_harmonic_{days}"] = 2 * a * b / (a + b + eps)
        x[f"structure_activity_{days}"] = x["tanimoto"] * x[f"cross_excitation_{days}"]
        x[f"degree_activity_{days}"] = x["logdeg_min"] * np.log1p(x[f"cross_excitation_{days}"])
    x["pair_persistence"] = x["pair_active_days"] / (1.0 + x["pair_total"])
    x["partner_geomean"] = np.sqrt((1.0 + x["drug1_partners"]) * (1.0 + x["drug2_partners"]))
    x["partner_min"] = np.minimum(x["drug1_partners"], x["drug2_partners"])
    x["partner_asymmetry"] = np.abs(np.log1p(x["drug1_partners"]) - np.log1p(x["drug2_partners"]))
    x["observation_opportunity"] = np.sqrt(np.log1p(x["drug1_total"]) * np.log1p(x["drug2_total"]))
    return x, list(x.columns)


def score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    probability = np.clip(score, 0, 1)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "aupr": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def make_xgb(y: np.ndarray, params: dict, seed: int = 42) -> XGBClassifier:
    positives = max(int(y.sum()), 1)
    return XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=-1,
        scale_pos_weight=(len(y) - positives) / positives, random_state=seed,
        **params,
    )


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def fit_background(train_x: pd.DataFrame, y: np.ndarray, pred_x: pd.DataFrame) -> tuple[np.ndarray, XGBClassifier]:
    model = make_xgb(y, {
        "n_estimators": 260, "max_depth": 5, "learning_rate": 0.06, "subsample": 0.9,
        "colsample_bytree": 0.9, "min_child_weight": 12, "reg_lambda": 3.0, "reg_alpha": 0.1,
    })
    model.fit(train_x[STATIC], y)
    return model.predict_proba(pred_x[STATIC])[:, 1], model


def select_model(train: pd.DataFrame, valid: pd.DataFrame, target: str) -> dict:
    x_train, features = engineer(train)
    x_valid, _ = engineer(valid)
    y_train = train[target].to_numpy()
    y_valid = valid[target].to_numpy()
    base_valid, background = fit_background(x_train, y_train, x_valid)
    base_train = background.predict_proba(x_train[STATIC])[:, 1]
    x_train["background_logit"] = logit(np.clip(base_train, 1e-5, 1 - 1e-5))
    x_valid["background_logit"] = logit(np.clip(base_valid, 1e-5, 1 - 1e-5))
    features = features + ["background_logit"]

    xgb_grid = [
        {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 8, "reg_lambda": 3.0, "reg_alpha": 0.1},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.035, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10, "reg_lambda": 4.0, "reg_alpha": 0.15},
        {"n_estimators": 420, "max_depth": 7, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 14, "reg_lambda": 5.0, "reg_alpha": 0.2},
    ]
    candidates = []
    for params in xgb_grid:
        model = make_xgb(y_train, params)
        model.fit(x_train[features], y_train)
        pred = model.predict_proba(x_valid[features])[:, 1]
        candidates.append((average_precision_score(y_valid, pred), "xgb", params, pred))
        print(target, "validation xgb", candidates[-1][0], params)

    hist_grid = [
        {"learning_rate": 0.06, "max_iter": 260, "max_leaf_nodes": 31, "min_samples_leaf": 35, "l2_regularization": 1.5},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "min_samples_leaf": 45, "l2_regularization": 2.0},
    ]
    for params in hist_grid:
        model = HistGradientBoostingClassifier(random_state=42, **params)
        model.fit(x_train[features], y_train)
        pred = model.predict_proba(x_valid[features])[:, 1]
        candidates.append((average_precision_score(y_valid, pred), "hist", params, pred))
        print(target, "validation hist", candidates[-1][0], params)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    second = candidates[1]
    blend_candidates = []
    for mode in ("probability", "rank"):
        a = best[3] if mode == "probability" else rank01(best[3])
        b = second[3] if mode == "probability" else rank01(second[3])
        for weight in np.linspace(0, 1, 21):
            pred = weight * a + (1 - weight) * b
            blend_candidates.append((average_precision_score(y_valid, pred), mode, float(weight)))
    blend_candidates.sort(reverse=True)
    return {
        "features": features,
        "best": {"family": best[1], "params": best[2], "validation_aupr": float(best[0])},
        "second": {"family": second[1], "params": second[2], "validation_aupr": float(second[0])},
        "blend": {"validation_aupr": float(blend_candidates[0][0]), "mode": blend_candidates[0][1], "weight_best": blend_candidates[0][2]},
    }


def fit_family(family: str, params: dict, x: pd.DataFrame, y: np.ndarray):
    model = make_xgb(y, params) if family == "xgb" else HistGradientBoostingClassifier(random_state=42, **params)
    model.fit(x, y)
    return model


def final_evaluation(frame: pd.DataFrame, target: str, at_risk: bool) -> dict:
    data = frame.loc[frame["at_risk_first"].eq(1)].copy() if at_risk else frame.copy()
    tuning_train = data[data["fold"].eq("2024q3")]
    validation = data[data["fold"].eq("2024q4")]
    selection = select_model(tuning_train, validation, target)

    train = data[data["fold"].isin(["2024q3", "2024q4"])]
    test = data[data["fold"].eq("2025q1")]
    x_train, _ = engineer(train)
    x_test, _ = engineer(test)
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()
    base_test, background = fit_background(x_train, y_train, x_test)
    base_train = background.predict_proba(x_train[STATIC])[:, 1]
    x_train["background_logit"] = logit(np.clip(base_train, 1e-5, 1 - 1e-5))
    x_test["background_logit"] = logit(np.clip(base_test, 1e-5, 1 - 1e-5))
    features = selection["features"]
    best_model = fit_family(selection["best"]["family"], selection["best"]["params"], x_train[features], y_train)
    second_model = fit_family(selection["second"]["family"], selection["second"]["params"], x_train[features], y_train)
    best_score = best_model.predict_proba(x_test[features])[:, 1]
    second_score = second_model.predict_proba(x_test[features])[:, 1]
    if selection["blend"]["mode"] == "rank":
        best_component, second_component = rank01(best_score), rank01(second_score)
    else:
        best_component, second_component = best_score, second_score
    weight = selection["blend"]["weight_best"]
    nest_score = weight * best_component + (1 - weight) * second_component
    cascade_fraction = np.clip(1.0 - base_test / np.clip(best_score, 1e-5, 1), 0, 1)
    return {
        "selection": selection,
        "background": score_metrics(y_test, base_test),
        "best_component": score_metrics(y_test, best_score),
        "second_component": score_metrics(y_test, second_score),
        "nest_blend": score_metrics(y_test, nest_score),
        "cascade_fraction_mean": float(cascade_fraction.mean()),
        "cascade_fraction_positive_mean": float(cascade_fraction[y_test == 1].mean()),
        "cascade_fraction_negative_mean": float(cascade_fraction[y_test == 0].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    result = {
        "protocol": "tune on 2024q3->2024q4; retrain 2024q3+2024q4; test 2025q1->2025q2",
        "any_next_quarter": final_evaluation(panel, "target_any_next_quarter", False),
        "first_next_quarter": final_evaluation(panel, "target_first_next_quarter", True),
    }
    (OUT / "nest_original_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
