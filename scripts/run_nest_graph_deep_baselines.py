#!/usr/bin/env python3
"""DPDDI-style historical graph baselines for the release-local panel.

The graph is built only from pair reports available through 2024Q4, before the
2025Q1->Q2 frozen test.  Three variants are evaluated: GCN+DNN, GCN with
feature fusion, and GraphSAGE with feature fusion.  This is a temporal
historical co-medication graph baseline, not a molecular-graph reproduction.
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
FEATURES = [
    "logdeg1", "logdeg2", "logdeg_min", "logdeg_max", "logdeg_product", "tanimoto",
    "pair_total", "pair_active_days", "pair_days_since", "pair_recent_7", "pair_decay_7",
    "pair_recent_30", "pair_decay_30", "pair_recent_90", "pair_decay_90", "drug1_total",
    "drug1_partners", "drug1_days_since", "drug2_total", "drug2_partners", "drug2_days_since",
    "drug1_recent_7", "drug1_decay_7", "drug2_recent_7", "drug2_decay_7", "drug1_recent_30",
    "drug1_decay_30", "drug2_recent_30", "drug2_decay_30", "drug1_recent_90", "drug1_decay_90",
    "drug2_recent_90", "drug2_decay_90", "cross_excitation_7", "activity_asymmetry_7", "pair_share_7",
    "cross_excitation_30", "activity_asymmetry_30", "pair_share_30", "cross_excitation_90",
    "activity_asymmetry_90", "pair_share_90",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


class GraphPairModel(nn.Module):
    def __init__(self, n_nodes: int, n_features: int, adjacency: torch.Tensor, variant: str) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.variant = variant
        self.node0 = nn.Parameter(torch.randn(n_nodes, 32) * 0.05)
        if variant.startswith("gcn"):
            self.w1 = nn.Linear(32, 32, bias=False)
            self.w2 = nn.Linear(32, 32, bias=False)
        else:
            self.s1 = nn.Linear(64, 32)
            self.s2 = nn.Linear(64, 32)
        pair_dim = 128
        if variant.endswith("fusion"):
            pair_dim += n_features
        self.head = nn.Sequential(nn.Linear(pair_dim, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))

    def encode(self) -> torch.Tensor:
        if self.variant.startswith("gcn"):
            h = torch.relu(torch.sparse.mm(self.adjacency, self.w1(self.node0)))
            return torch.sparse.mm(self.adjacency, self.w2(h))
        neigh = torch.sparse.mm(self.adjacency, self.node0)
        h = torch.relu(self.s1(torch.cat([self.node0, neigh], dim=1)))
        neigh2 = torch.sparse.mm(self.adjacency, h)
        return self.s2(torch.cat([h, neigh2], dim=1))

    def pair_logits(self, e: torch.Tensor, i: torch.Tensor, j: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
        ei, ej = e[i], e[j]
        pair = torch.cat([ei, ej, torch.abs(ei - ej), ei * ej], dim=1)
        if self.variant.endswith("fusion"):
            pair = torch.cat([pair, x], dim=1)
        return self.head(pair).squeeze(-1)


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    return {"auroc": float(roc_auc_score(y, score)), "aupr": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, np.clip(score, 0, 1))), "n": int(len(y)),
            "positives": int(y.sum()), "precision_at_100": float(y[order[:100]].mean()),
            "precision_at_500": float(y[order[:500]].mean()), "precision_at_1000": float(y[order[:1000]].mean())}


def graph_matrix(panel: pd.DataFrame, history_fold: str = "2024q4") -> tuple[torch.Tensor, dict[str, int]]:
    q4 = panel[panel["fold"].eq(history_fold)]
    ids = sorted(set(panel["drug1_id"]) | set(panel["drug2_id"]))
    mapping = {drug: k for k, drug in enumerate(ids)}
    edges = q4.loc[q4["pair_total"].gt(0), ["drug1_id", "drug2_id"]].drop_duplicates().to_numpy()
    src, dst = [], []
    for a, b in edges:
        i, j = mapping[a], mapping[b]
        src.extend([i, j]); dst.extend([j, i])
    src.extend(range(len(ids))); dst.extend(range(len(ids)))
    idx = torch.tensor([src, dst], dtype=torch.long)
    deg = torch.bincount(idx[0], minlength=len(ids)).float().clamp_min(1)
    val = 1.0 / torch.sqrt(deg[idx[0]] * deg[idx[1]])
    return torch.sparse_coo_tensor(idx, val, (len(ids), len(ids))).coalesce(), mapping


def arrays(frame: pd.DataFrame, mapping: dict[str, int], mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = frame["drug1_id"].map(mapping).to_numpy(dtype=np.int64)
    j = frame["drug2_id"].map(mapping).to_numpy(dtype=np.int64)
    x = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    return i, j, ((x - mean) / scale).astype(np.float32)


def fit_model(name: str, adjacency: torch.Tensor, mapping: dict[str, int], train: pd.DataFrame, test: pd.DataFrame,
              target: str, epochs: int, batch_size: int, seed: int) -> np.ndarray:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    ti, tj, tx = arrays(train, mapping, mean, scale)
    vi, vj, vx = arrays(test, mapping, mean, scale)
    y = train[target].to_numpy(dtype=np.float32)
    yt = test[target].to_numpy(dtype=np.int64)
    model = GraphPairModel(len(mapping), len(FEATURES), adjacency.to(device), name).to(device)
    pos = max(float(y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - pos) / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(ti), torch.from_numpy(tj), torch.from_numpy(tx), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train(); total = 0.0
        for bi, bj, bx, by in loader:
            bi, bj, bx, by = bi.to(device), bj.to(device), bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            # Rebuild the small graph encoder for each pair batch so that
            # autograd does not reuse a freed computation graph.
            embedding = model.encode()
            loss = loss_fn(model.pair_logits(embedding, bi, bj, bx if name.endswith("fusion") else None), by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); total += float(loss.detach()) * len(by)
        print(f"{target} {name} epoch={epoch + 1} loss={total / len(y):.5f}", flush=True)
    model.eval(); embedding = model.encode(); out = []
    with torch.no_grad():
        for start in range(0, len(vi), batch_size * 2):
            bi = torch.from_numpy(vi[start:start + batch_size * 2]).to(device)
            bj = torch.from_numpy(vj[start:start + batch_size * 2]).to(device)
            bx = torch.from_numpy(vx[start:start + batch_size * 2]).to(device)
            out.append(torch.sigmoid(model.pair_logits(embedding, bi, bj, bx if name.endswith("fusion") else None)).cpu().numpy())
    score = np.concatenate(out)
    print(target, name, metrics(yt, score), flush=True)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tag", default="4q_release"); ap.add_argument("--output-tag", default="graph_deep_baselines_release_local")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch-size", type=int, default=8192); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["gcn", "gcn_fusion", "sage_fusion"])
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    panel = pd.read_parquet(DATA / f"nest_ddi_prediction_panel_{args.input_tag}.parquet")
    adjacency, mapping = graph_matrix(panel); print("graph_nodes", len(mapping), "graph_nnz", adjacency._nnz(), flush=True)
    result = {"protocol": "graph built from pair_total>0 in 2024q4 only; train q3+q4; frozen test q1->q2", "models": args.models,
              "epochs": args.epochs, "batch_size": args.batch_size, "device": "cuda" if torch.cuda.is_available() else "cpu", "tasks": {}}
    payload = {}
    for target in ("target_any_next_quarter", "target_first_next_quarter"):
        train = panel[panel["fold"].isin(["2024q3", "2024q4"])].copy(); test = panel[panel["fold"].eq("2025q1")].copy()
        if target == "target_first_next_quarter":
            train = train[train.at_risk_first.eq(1)].copy(); test = test[test.at_risk_first.eq(1)].copy()
        task = {"train_rows": len(train), "test_rows": len(test), "results": {}}; payload[target] = {"y": test[target].to_numpy()}
        for name in args.models:
            score = fit_model(name, adjacency, mapping, train, test, target, args.epochs, args.batch_size, args.seed)
            task["results"][name] = metrics(test[target].to_numpy(), score); payload[target][name] = score
        result["tasks"][target] = task
    np.savez_compressed(OUT / f"{args.output_tag}_scores.npz", **{f"{t}__{n}": a for t, p in payload.items() for n, a in p.items()})
    (OUT / f"{args.output_tag}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
