#!/usr/bin/env python3
"""DrugDAGT-style temporal reproduction with RDKit molecular graphs.

Uses the audited temporal labels, RDKit atom/bond graphs and a dual-attention
pair decoder. The original paper is multi-class; this adapts only the output
head to the current binary prospective labels.
"""
from pathlib import Path
import json, argparse, numpy as np, pandas as pd, torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import average_precision_score, roc_auc_score
from run_nest_graph_deep_baselines import metrics, seed_all

ROOT=Path(r"D:\BI\DDI"); DATA=ROOT/'data/processed'; OUT=ROOT/'results/nest_ddi'

def mol_graph(s):
    m=Chem.MolFromSmiles(s or '');
    if m is None: return Data(x=torch.zeros((1,8)),edge_index=torch.zeros((2,0),dtype=torch.long))
    x=[]
    for a in m.GetAtoms(): x.append([a.GetAtomicNum()/100,a.GetTotalDegree()/5,a.GetFormalCharge()/3,a.GetTotalNumHs()/4,a.GetIsAromatic(),a.IsInRing(),a.GetMass()/250,a.GetTotalValence()/8])
    e=[]
    for b in m.GetBonds(): i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); e += [[i,j],[j,i]]
    return Data(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor(e,dtype=torch.long).t().contiguous() if e else torch.zeros((2,0),dtype=torch.long))

class DrugDAGT(nn.Module):
    def __init__(self):
        super().__init__(); self.g1=GATConv(8,32,heads=2,concat=True); self.g2=GATConv(64,64,heads=1); self.q=nn.Linear(64,64); self.k=nn.Linear(64,64); self.v=nn.Linear(64,64); self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
    def encode(self,b):
        h=torch.relu(self.g1(b.x,b.edge_index)); h=self.g2(h,b.edge_index); return global_mean_pool(h,b.batch)
    def forward(self,e1,e2):
        a=torch.sigmoid((self.q(e1)*self.k(e2)).sum(1,keepdim=True)/8**0.5); z=torch.cat([e1,e2,(e1-e2).abs(),a*self.v(e2)+ (1-a)*self.v(e1)],1); return self.head(z).squeeze(1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--batch-size',type=int,default=4096); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); seed_all(a.seed); OUT.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(DATA/'nest_ddi_prediction_panel_4q_release.parquet'); sm=json.loads((DATA/'ddinter_smiles.json').read_text()); id2name=pd.read_csv(DATA/'ddinter_drug_identity_map.csv').set_index('id')['name'].to_dict(); names=sorted(sm); idx={n:i for i,n in enumerate(names)}; graphs=[mol_graph(sm[n].get('smiles')) for n in names]; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); b=Batch.from_data_list(graphs).to(device)
    out={'model':'DrugDAGT-Temporal','protocol':'RDKit molecular graphs + dual-attention pair decoder; binary temporal adaptation of multi-class DrugDAGT','tasks':{}}
    for target in ('target_any_next_quarter','target_first_next_quarter'):
        tr=panel[panel.fold.isin(['2024q3','2024q4'])].copy(); te=panel[panel.fold.eq('2025q1')].copy();
        if target.endswith('first_next_quarter'): tr=tr[tr.at_risk_first.eq(1)]; te=te[te.at_risk_first.eq(1)]
        def ids(f): return np.array([idx.get(id2name.get(x,''),-1) for x in f],dtype=np.int64)
        i,j=ids(tr.drug1_id),ids(tr.drug2_id); ti,tj=ids(te.drug1_id),ids(te.drug2_id); keep=(i>=0)&(j>=0); tr=tr.iloc[np.where(keep)[0]]; i,j=i[keep],j[keep]; tk=(ti>=0)&(tj>=0); te=te.iloc[np.where(tk)[0]]; ti,tj=ti[tk],tj[tk]; y=torch.tensor(tr[target].to_numpy(np.float32),device=device); yt=te[target].to_numpy(np.int64)
        model=DrugDAGT().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); pos=max(float(y.sum()),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-pos)/pos],device=device)); dl=torch.utils.data.DataLoader(torch.arange(len(y)),batch_size=a.batch_size,shuffle=True)
        for ep in range(a.epochs):
            model.train(); total=0
            for bi in dl:
                bi=bi.numpy(); emb=model.encode(b); loss=lf(model(emb[torch.tensor(i[bi],device=device)],emb[torch.tensor(j[bi],device=device)]),y[torch.tensor(bi,device=device)]); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(bi)
            print(target,ep+1,total/len(y),flush=True)
        model.eval(); emb=model.encode(b); scores=[]
        with torch.no_grad():
            for s in range(0,len(ti),a.batch_size): scores.append(torch.sigmoid(model(emb[torch.tensor(ti[s:s+a.batch_size],device=device)],emb[torch.tensor(tj[s:s+a.batch_size],device=device)])).cpu().numpy())
        score=np.concatenate(scores); out['tasks'][target]={'train_rows':len(y),'test_rows':len(yt),'metrics':metrics(yt,score)}
    (OUT/'drugdagt_temporal_reproduction_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
