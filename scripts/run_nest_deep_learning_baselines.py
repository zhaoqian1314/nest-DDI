#!/usr/bin/env python3
"""Deep-learning baselines under the release-local temporal DDI protocol.

This script deliberately compares architecture families on the same frozen
pair panel.  It does not claim to reproduce a molecular-graph paper when the
input panel contains cutoff-available tabular pair/history features.  The
implemented families are: DeepDDI-style dense DNN, residual MLP, Wide&Deep,
DeepFM, a lightweight FT-Transformer for numerical tokens, and a 1-D CNN.
All models are trained on 2024Q3--2025Q1 history (the Q3/Q4 folds) and scored
once on the untouched 2025Q1->2025Q2 fold.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"
STATIC = ["logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto"]
PAIR_HISTORY = [
    "pair_total", "pair_active_days", "pair_days_since", "pair_recent_7", "pair_decay_7",
    "pair_recent_30", "pair_decay_30", "pair_recent_90", "pair_decay_90",
]
DRUG_HISTORY = [
    "drug1_total", "drug1_partners", "drug1_days_since", "drug2_total", "drug2_partners",
    "drug2_days_since", "drug1_recent_7", "drug1_decay_7", "drug2_recent_7", "drug2_decay_7",
    "drug1_recent_30", "drug1_decay_30", "drug2_recent_30", "drug2_decay_30",
    "drug1_recent_90", "drug1_decay_90", "drug2_recent_90", "drug2_decay_90",
]
RAW_CROSS = [
    "cross_excitation_7", "activity_asymmetry_7", "pair_share_7", "cross_excitation_30",
    "activity_asymmetry_30", "pair_share_30", "cross_excitation_90", "activity_asymmetry_90",
    "pair_share_90",
]
FEATURES = STATIC + PAIR_HISTORY + DRUG_HISTORY + RAW_CROSS


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


class DenseDNN(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128)),
            nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128)),
            nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 128)),
        ])
        self.out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = torch.relu(h + block(h))
        return self.out(h).squeeze(-1)


class WideDeep(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.wide = nn.Linear(n, 1)
        self.deep = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.wide(x) + self.deep(x)).squeeze(-1)


class DeepFM(nn.Module):
    def __init__(self, n: int, k: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(n, 1)
        self.emb = nn.Parameter(torch.randn(n, k) * 0.02)
        self.deep = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xv = x.unsqueeze(-1) * self.emb.unsqueeze(0)
        summed = xv.sum(dim=1)
        fm = 0.5 * (summed.pow(2) - xv.pow(2).sum(dim=1)).sum(dim=1, keepdim=True)
        return (self.linear(x) + fm + self.deep(x)).squeeze(-1)


class FTTransformerLite(nn.Module):
    def __init__(self, n: int, d: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=64, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        h = self.encoder(tokens).mean(dim=1)
        return self.out(self.norm(h)).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
                                  nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.out = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.out(h).squeeze(-1)


MODEL_BUILDERS = {
    "deepddi_dnn": DenseDNN,
    "residual_mlp": ResidualMLP,
    "wide_deep": WideDeep,
    "deepfm": DeepFM,
    "ft_transformer": FTTransformerLite,
    "cnn1d": CNN1D,
}


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "aupr": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "precision_at_100": float(y[order[:100]].mean()),
        "precision_at_500": float(y[order[:500]].mean()),
        "precision_at_1000": float(y[order[:1000]].mean()),
    }


def make_arrays(frame: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    return ((x - mean) / scale).astype(np.float32)


def fit_predict(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                epochs: int, batch_size: int, seed: int) -> np.ndarray:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_BUILDERS[name](x_train.shape[1]).to(device)
    positives = max(float(y_train.sum()), 1.0)
    weight = torch.tensor([(len(y_train) - positives) / positives], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
                        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(yb)
        print(f"{name} epoch={epoch + 1} loss={running / len(y_train):.5f}", flush=True)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size * 2):
            xb = torch.from_numpy(x_test[start:start + batch_size * 2]).to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


def run_task(panel: pd.DataFrame, target: str, models: list[str], epochs: int, batch_size: int, seed: int) -> dict:
    train = panel[panel["fold"].isin(["2024q3", "2024q4"])].copy()
    test = panel[panel["fold"].eq("2025q1")].copy()
    if target == "target_first_next_quarter":
        train = train[train["at_risk_first"].eq(1)].copy()
        test = test[test["at_risk_first"].eq(1)].copy()
    raw = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = ((raw - mean) / scale).astype(np.float32)
    x_test = make_arrays(test, mean, scale)
    y_train = train[target].to_numpy(dtype=np.int64)
    y_test = test[target].to_numpy(dtype=np.int64)
    result = {"target": target, "train_rows": int(len(train)), "test_rows": int(len(test)), "results": {}}
    scores = {}
    for name in models:
        scores[name] = fit_predict(name, x_train, y_train, x_test, epochs, batch_size, seed)
        result["results"][name] = metrics(y_test, scores[name])
        print(target, name, result["results"][name], flush=True)
    return result, y_test, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="4q_release")
    parser.add_argument("--output-tag", default="deep_baselines_release_local")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=list(MODEL_BUILDERS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    all_results = {"protocol": "train on 2024q3+2024q4; frozen test on 2025q1->2025q2; standardization fit on training folds only",
                   "features": FEATURES, "models": args.models, "epochs": args.epochs, "batch_size": args.batch_size,
                   "device": "cuda" if torch.cuda.is_available() else "cpu", "tasks": {}}
    score_payload = {}
    for target in ("target_any_next_quarter", "target_first_next_quarter"):
        result, y, scores = run_task(panel, target, args.models, args.epochs, args.batch_size, args.seed)
        all_results["tasks"][target] = result
        score_payload[target] = {"y": y}
        score_payload[target].update(scores)
    np.savez_compressed(OUT / f"{args.output_tag}_scores.npz", **{f"{task}__{name}": value for task, payload in score_payload.items() for name, value in payload.items()})
    (OUT / f"{args.output_tag}_metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
