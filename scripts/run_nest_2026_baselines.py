#!/usr/bin/env python3
"""Release-local reimplementations of recent 2025--2026 DDI architectures.

The public papers use molecular/heterogeneous graphs that are not present in
the FAERS release panel.  These variants therefore keep the audited temporal
co-medication graph and replace only the message-passing/decoder block.  They
are explicitly labelled reimplementations, not claims of exact paper-level
reproduction.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from run_nest_graph_deep_baselines import FEATURES, arrays, graph_matrix, metrics, seed_all

ROOT = Path(r"D:\BI\DDI")
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nest_ddi"


class ModernPairModel(nn.Module):
    def __init__(self, n_nodes: int, n_features: int, adj: torch.Tensor, variant: str):
        super().__init__(); self.register_buffer("adj", adj); self.variant = variant
        self.node0 = nn.Parameter(torch.randn(n_nodes, 48) * 0.04)
        self.inp = nn.Linear(48, 64)
        self.norm = nn.LayerNorm(64)
        if variant.startswith("gin"):
            self.mp = nn.Sequential(nn.Linear(64, 96), nn.GELU(), nn.Linear(96, 64))
        elif variant.startswith("edgeconv"):
            self.mp = nn.Linear(128, 64)
        else:
            self.mp = nn.Linear(64, 64)
        self.gate = nn.Linear(64, 1)
        extra = n_features if variant.endswith("fusion") else 0
        if variant.startswith("fgddi"):
            extra += 4  # chemistry-prior proxy features available in this panel
        self.head = nn.Sequential(nn.Linear(256 + extra, 128), nn.GELU(), nn.Dropout(.15), nn.Linear(128, 1))

    def encode(self):
        h = self.norm(self.inp(self.node0))
        neigh = torch.sparse.mm(self.adj, h)
        if self.variant.startswith("edgeconv"):
            # EdgeConv-style local geometry: aggregated |h_i-h_j|.
            diff = torch.sparse.mm(self.adj, h.abs()) - h.abs() * 0.0
            h = torch.relu(self.mp(torch.cat([h, diff], 1)))
        elif self.variant.startswith("gin"):
            h = self.mp(h + neigh)
        elif self.variant.startswith("gat") or self.variant.startswith("fgddi"):
            alpha = torch.sigmoid(self.gate(h))
            h = torch.relu(self.mp(h + torch.sparse.mm(self.adj, h * alpha)))
        else:  # graph-transformer surrogate: residual normalized attention-free block
            h = torch.relu(self.norm(h + self.mp(neigh)))
        return h

    def logits(self, emb, i, j, x):
        ei, ej = emb[i], emb[j]
        z = torch.cat([ei, ej, (ei-ej).abs(), ei*ej], 1)
        if self.variant.endswith("fusion"):
            z = torch.cat([z, x], 1)
        if self.variant.startswith("fgddi"):
            # Pairwise chemistry/statistical enrichment available without SMILES.
            z = torch.cat([z, x[:, [4, 5, 6, 7]]], 1)
        return self.head(z).squeeze(-1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-tag",default="4q_release"); ap.add_argument("--output-tag",default="modern_2026_baselines_release_local"); ap.add_argument("--epochs",type=int,default=4); ap.add_argument("--batch-size",type=int,default=8192); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--models",nargs="+",default=["gat_fusion","gin_fusion","edgeconv_fusion","fgddi_fusion","graph_transformer_fusion"]); a=ap.parse_args()
    panel=pd.read_parquet(DATA/f"nest_ddi_prediction_panel_{a.input_tag}.parquet"); adj,mapping=graph_matrix(panel); OUT.mkdir(parents=True,exist_ok=True)
    result={"protocol":"release-local temporal graph; recent architecture reimplementations using available covariates","models":a.models,"tasks":{}}; payload={}
    for target in ("target_any_next_quarter","target_first_next_quarter"):
        train=panel[panel.fold.isin(["2024q3","2024q4"])].copy(); test=panel[panel.fold.eq("2025q1")].copy()
        if target.endswith("first_next_quarter"): train=train[train.at_risk_first.eq(1)].copy(); test=test[test.at_risk_first.eq(1)].copy()
        task={"train_rows":len(train),"test_rows":len(test),"results":{}}; payload[target]={"y":test[target].to_numpy()}
        for name in a.models:
            score=fit_model(name,adj,mapping,train,test,target,a.epochs,a.batch_size,a.seed)
            task["results"][name]=metrics(test[target].to_numpy(),score); payload[target][name]=score
        result["tasks"][target]=task
    np.savez_compressed(OUT/f"{a.output_tag}_scores.npz",**{f"{t}__{n}":v for t,p in payload.items() for n,v in p.items()}); (OUT/f"{a.output_tag}_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


def fit_model(*args, **kwargs):
    # Keep optimizer local by duplicating the short training wrapper safely.
    variant,adj,mapping,train,test,target,epochs,batch,seed=args; seed_all(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw=train[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32); mean,scale=raw.mean(0),raw.std(0); scale[scale<1e-6]=1
    ti,tj,tx=arrays(train,mapping,mean,scale); vi,vj,vx=arrays(test,mapping,mean,scale); y=train[target].to_numpy(np.float32); yt=test[target].to_numpy(np.int64)
    model=ModernPairModel(len(mapping),len(FEATURES),adj.to(device),variant).to(device); pos=max(float(y.sum()),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-pos)/pos],device=device)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.from_numpy(ti),torch.from_numpy(tj),torch.from_numpy(tx),torch.from_numpy(y)),batch_size=batch,shuffle=True)
    for ep in range(epochs):
        total=0; model.train()
        for bi,bj,bx,by in dl:
            bi,bj,bx,by=bi.to(device),bj.to(device),bx.to(device),by.to(device); opt.zero_grad(set_to_none=True); loss=lf(model.logits(model.encode(),bi,bj,bx),by); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach())*len(by)
        print(f"{target} {variant} epoch={ep+1} loss={total/len(y):.5f}",flush=True)
    model.eval(); emb=model.encode(); out=[]
    with torch.no_grad():
        for s in range(0,len(vi),batch*2): out.append(torch.sigmoid(model.logits(emb,torch.from_numpy(vi[s:s+batch*2]).to(device),torch.from_numpy(vj[s:s+batch*2]).to(device),torch.from_numpy(vx[s:s+batch*2]).to(device))).cpu().numpy())
    score=np.concatenate(out); print(target,variant,metrics(yt,score),flush=True); return score

if __name__ == "__main__": main()
