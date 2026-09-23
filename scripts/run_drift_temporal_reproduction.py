#!/usr/bin/env python3
"""DRIFT-style temporal reproduction: ChemBERTa SMILES + Morgan FP + pair MLP.

The released environment uses the locally cached ChemBERTa checkpoint when
ChemBERTa-2 is unavailable; the checkpoint id is recorded in the output.
"""
from pathlib import Path
import json, argparse, random
import numpy as np, pandas as pd, torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score
from run_nest_graph_deep_baselines import metrics, seed_all

ROOT=Path(r"D:\BI\DDI"); DATA=ROOT/'data/processed'; OUT=ROOT/'results/nest_ddi'
# This checkpoint is cached with a complete tokenizer in the release runtime.
# It is the 77M ChemBERTa family checkpoint used when ChemBERTa-2 weights are
# not locally available; the exact checkpoint id is recorded in the output.
MODEL_ID='DeepChem/ChemBERTa-77M-MLM'

def embed(smiles_map, cache):
    if cache.exists(): return np.load(cache)['emb']
    tok=AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    # Transformers blocks loading legacy .bin checkpoints with torch<2.6.
    # Load the cached state dict explicitly with weights_only=True instead.
    snap=next(p for p in (Path.home()/'.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MLM/snapshots').glob('*') if (p/'pytorch_model.bin').exists())
    mod=AutoModel.from_config(AutoConfig.from_pretrained(MODEL_ID, local_files_only=True))
    state=torch.load(snap/'pytorch_model.bin', map_location='cpu', weights_only=True); mod.load_state_dict(state, strict=False); mod.eval()
    names=sorted(smiles_map); out=[]
    with torch.no_grad():
        for s in range(0,len(names),32):
            ss=[smiles_map[n] or '' for n in names]; z=tok(ss[s:s+32],padding=True,truncation=True,return_tensors='pt'); h=mod(**z).last_hidden_state; mask=z['attention_mask'].unsqueeze(-1); out.append(((h*mask).sum(1)/mask.sum(1).clamp_min(1)).numpy())
    emb=np.concatenate(out); np.savez_compressed(cache,names=np.array(names),emb=emb); return emb

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--batch-size',type=int,default=4096); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); seed_all(a.seed); OUT.mkdir(exist_ok=True,parents=True)
    panel=pd.read_parquet(DATA/'nest_ddi_prediction_panel_4q_release.parquet'); sm=json.loads((DATA/'ddinter_smiles.json').read_text()); names=sorted(sm); emb=embed({n:sm[n].get('smiles') for n in names},OUT/'drift_chemberta_embeddings.npz'); id_to_name=pd.read_csv(DATA/'ddinter_drug_identity_map.csv').set_index('id')['name'].to_dict(); idx={k:i for i,k in enumerate(names)}
    fps=[]
    for n in names:
        m=Chem.MolFromSmiles(sm[n].get('smiles') or ''); bit=np.zeros((2048,),dtype=np.float32)
        if m is not None: DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),bit)
        fps.append(bit)
    fps=np.asarray(fps); feat=np.concatenate([emb,fps],1).astype(np.float32); result={'model':'DRIFT-Temporal','checkpoint':MODEL_ID,'protocol':'temporal FAERS panel; ChemBERTa embedding + Morgan fingerprint + pair MLP','tasks':{}}
    for target in ('target_any_next_quarter','target_first_next_quarter'):
        tr=panel[panel.fold.isin(['2024q3','2024q4'])].copy(); te=panel[panel.fold.eq('2025q1')].copy()
        if target.endswith('first_next_quarter'): tr=tr[tr.at_risk_first.eq(1)]; te=te[te.at_risk_first.eq(1)]
        i=np.array([idx.get(id_to_name.get(x,''),-1) for x in tr.drug1_id]); j=np.array([idx.get(id_to_name.get(x,''),-1) for x in tr.drug2_id]); ti=np.array([idx.get(id_to_name.get(x,''),-1) for x in te.drug1_id]); tj=np.array([idx.get(id_to_name.get(x,''),-1) for x in te.drug2_id]); keep=(i>=0)&(j>=0); tr=tr.iloc[np.where(keep)[0]]; i=i[keep]; j=j[keep]; test_keep=(ti>=0)&(tj>=0); te=te.iloc[np.where(test_keep)[0]]; ti=ti[test_keep]; tj=tj[test_keep]
        y=tr[target].to_numpy(np.float32); yt=te[target].to_numpy(np.int64); dim=feat.shape[1]*4
        model=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1)); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); pos=max(float(y.sum()),1); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-pos)/pos])); dl=DataLoader(TensorDataset(torch.from_numpy(i),torch.from_numpy(j),torch.from_numpy(y)),batch_size=a.batch_size,shuffle=True)
        for _ in range(a.epochs):
            for bi,bj,by in dl:
                bx=torch.cat([torch.from_numpy(feat[bi]),torch.from_numpy(feat[bj]),torch.from_numpy(np.abs(feat[bi]-feat[bj])),torch.from_numpy(feat[bi]*feat[bj])],1); opt.zero_grad(); loss=lf(model(bx).squeeze(1),by); loss.backward(); opt.step()
        score=[]
        with torch.no_grad():
            for s in range(0,len(ti),a.batch_size):
                bi,bj=ti[s:s+a.batch_size],tj[s:s+a.batch_size]; bx=torch.from_numpy(np.concatenate([feat[bi],feat[bj],np.abs(feat[bi]-feat[bj]),feat[bi]*feat[bj]],1)); score.append(torch.sigmoid(model(bx).squeeze(1)).numpy())
        score=np.concatenate(score)
        result['tasks'][target]={'train_rows':len(y),'test_rows':len(yt),'metrics':metrics(yt,score)}
    (OUT/'drift_temporal_reproduction_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
