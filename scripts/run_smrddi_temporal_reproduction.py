#!/usr/bin/env python3
"""SMRDDI (2024) temporal reproduction: self-supervised SMILES CNN + pair head."""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd,torch
from torch import nn
from run_nest_graph_deep_baselines import metrics,seed_all
ROOT=Path(r"D:\BI\DDI"); DATA=ROOT/'data/processed'; OUT=ROOT/'results/nest_ddi'

class Encoder(nn.Module):
    def __init__(self,vocab):
        super().__init__(); self.emb=nn.Embedding(vocab,64,padding_idx=0); self.conv=nn.Sequential(nn.Conv1d(64,96,5,padding=2),nn.GELU(),nn.Conv1d(96,128,5,padding=2),nn.GELU()); self.pool=nn.AdaptiveMaxPool1d(1)
    def forward(self,x): return self.pool(self.conv(self.emb(x).transpose(1,2))).squeeze(-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--ssl-epochs',type=int,default=4); ap.add_argument('--batch-size',type=int,default=16384); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); seed_all(a.seed); OUT.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(DATA/'nest_ddi_prediction_panel_4q_release.parquet'); sm=json.loads((DATA/'ddinter_smiles.json').read_text()); id2name=pd.read_csv(DATA/'ddinter_drug_identity_map.csv').set_index('id')['name'].to_dict(); names=sorted(sm); chars=sorted(set(''.join(sm[n].get('smiles') or '' for n in names))); vocab={c:i+1 for i,c in enumerate(chars)}; L=128; seq=np.zeros((len(names),L),np.int64)
    for k,n in enumerate(names): seq[k,:min(L,len(sm[n].get('smiles') or ''))]=[vocab.get(c,0) for c in (sm[n].get('smiles') or '')[:L]]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); S=torch.tensor(seq,device=device); enc=Encoder(len(vocab)+1).to(device)
    # SMRDDI-style self-supervised denoising pretraining on molecular strings.
    opt=torch.optim.AdamW(enc.parameters(),lr=2e-3)
    for ep in range(a.ssl_epochs):
        perm=torch.randperm(len(S),device=device); total=0
        for st in range(0,len(S),512):
            ids=perm[st:st+512]; x=S[ids].clone(); z=S[ids].clone(); x[(x>0)&(torch.rand_like(x.float())<.15)]=0; z[(z>0)&(torch.rand_like(z.float())<.15)]=0; h1=nn.functional.normalize(enc(x),dim=1); h2=nn.functional.normalize(enc(z),dim=1); loss=(1-(h1*h2).sum(1)).mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(ids)
        print('ssl',ep+1,total/len(S),flush=True)
    with torch.no_grad(): drug_emb=enc(S).cpu().numpy()
    idx={n:i for i,n in enumerate(names)}; result={'model':'SMRDDI-Temporal','protocol':'SMRDDI-style self-supervised SMILES CNN denoising + pair MLP; binary temporal adaptation','tasks':{}}
    for target in ('target_any_next_quarter','target_first_next_quarter'):
        tr=panel[panel.fold.isin(['2024q3','2024q4'])].copy(); te=panel[panel.fold.eq('2025q1')].copy();
        if target.endswith('first_next_quarter'): tr=tr[tr.at_risk_first.eq(1)]; te=te[te.at_risk_first.eq(1)]
        def ids(f): return np.array([idx.get(id2name.get(x,''),-1) for x in f],np.int64)
        i,j=ids(tr.drug1_id),ids(tr.drug2_id); ti,tj=ids(te.drug1_id),ids(te.drug2_id); keep=(i>=0)&(j>=0); tr=tr.iloc[np.where(keep)[0]]; i,j=i[keep],j[keep]; tk=(ti>=0)&(tj>=0); te=te.iloc[np.where(tk)[0]]; ti,tj=ti[tk],tj[tk]
        x=np.concatenate([drug_emb[i],drug_emb[j],np.abs(drug_emb[i]-drug_emb[j]),drug_emb[i]*drug_emb[j]],1).astype(np.float32); xt=np.concatenate([drug_emb[ti],drug_emb[tj],np.abs(drug_emb[ti]-drug_emb[tj]),drug_emb[ti]*drug_emb[tj]],1).astype(np.float32); y=tr[target].to_numpy(np.float32); yt=te[target].to_numpy(np.int64)
        head=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1)); opt=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-4); pos=max(float(y.sum()),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-pos)/pos])); dl=torch.utils.data.DataLoader(torch.arange(len(y)),batch_size=a.batch_size,shuffle=True)
        X=torch.from_numpy(x); Y=torch.from_numpy(y)
        for ep in range(a.epochs):
            for b in dl: opt.zero_grad(); loss=lf(head(X[b]).squeeze(1),Y[b]); loss.backward(); opt.step()
        with torch.no_grad(): score=torch.sigmoid(head(torch.from_numpy(xt)).squeeze(1)).numpy()
        result['tasks'][target]={'train_rows':len(y),'test_rows':len(yt),'metrics':metrics(yt,score)}
    (OUT/'smrddi_temporal_reproduction_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
