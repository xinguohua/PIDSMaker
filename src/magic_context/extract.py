"""Frozen MAGIC node-window contexts, objective diagnostics and unmodified original scores."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import negative_sampling
from yacs.config import CfgNode
import yaml

def intern(mapping,key):
    return mapping.setdefault(key,len(mapping))
def grouped(n,dest,values):
    order=np.argsort(dest,kind='stable')
    return np.split(values[order],np.cumsum(np.bincount(dest,minlength=n))[:-1])
def context(b,labels,ops,codebooks,semantics):
    x=b.x.cpu().numpy()
    ids=b.original_n_id.cpu().numpy().reshape(-1)
    u,v=b.edge_index.cpu().numpy()
    n=len(x)
    assert len(ids)==len(set(ids))==n
    assert not hasattr(b,'edge_feats') or b.edge_feats is None or b.edge_feats.numel()==0
    raw=np.array([intern(codebooks[0],t.tobytes()) for t in x])
    idx=raw.copy()
    incoming=grouped(n,v,u)
    # Three directed message-passing rounds: preserve multiplicity and self/residual features.
    for book in codebooks[1:]:
        idx=np.array([intern(book,(int(idx[i]),tuple(sorted(idx[neighbors].tolist())))) for i,neighbors in enumerate(incoming)])
    ns=[labels[int(i)] for i in ids]
    op=b.edge_type.argmax(1).cpu().tolist()
    hops=[[] for _ in ids]
    for a,z,r in zip(u,v,op):
        hops[a].append((1,ops[r],ns[z]));hops[z].append((0,ops[r],ns[a]))
    sem=np.array([intern(semantics,(ns[i],hashlib.sha256(repr(sorted(e)).encode()).digest())) for i,e in enumerate(hops)])
    ins=np.bincount(v,minlength=n);outs=np.bincount(u,minlength=n)
    # Only pre-encoder information: raw root x and means/totals of incoming walks.
    signals=np.c_[x,np.ones(n)]
    fields=[x]
    for depth in range(3):
        nxt=np.zeros_like(signals)
        np.add.at(nxt,v,signals[u])
        fields.extend([nxt[:,:-1]/np.maximum(1,nxt[:,-1:]),np.log1p(nxt[:,-1:])])
        signals=nxt
    return ids,idx,sem,ins,outs,np.hstack(fields).astype('float32'),x
def diagnostics(model,b,feat,struct,seed):
    import torch.nn.functional as F
    n=len(b.x);u,v=b.edge_index;full=b.x
    ordered=torch.randperm(n,device=full.device);loss=torch.full((n,),float('nan'),device=full.device)
    bce=torch.zeros(n,device=full.device);weight=torch.zeros_like(bce)
    # Complementary masks cover every node. For odd n the second mask has one extra node.
    for mask in (ordered[:n//2],ordered[n//2:]):
        x=full.clone();x[mask]=feat.mask_token
        b.x=x
        h=model.embed(b,inference=True)[0]
        b.x=full
        reconstructed=feat.decoder(h,b.edge_index,edge_feats=getattr(b,'edge_feats',None))
        loss[mask]=feat.loss_fn(reconstructed[mask],full[mask],inference=True)
        neg=negative_sampling(b.edge_index,num_nodes=n,num_neg_samples=len(u))
        for (a,d),y in ((b.edge_index,1.),(neg,0.)):
            logits=struct.edge_recon_fc(torch.cat([h[a],h[d]],dim=-1)).reshape(-1)
            values=F.binary_cross_entropy_with_logits(logits,torch.full_like(logits,y),reduction='none')
            for end in (a,d):
                bce.index_add_(0,end,values);weight.index_add_(0,end,torch.ones_like(values))
    return loss.cpu().numpy(),(bce/weight.clamp_min(1)).cpu().numpy()
def main(cell,out):
    run=cell.parents[1];sys.path[:0]=[str(run/'code'),str(run/'code/src')]
    from baseline_matrix import digest,load_evaluation_checkpoint
    from pidsmaker.utils.replay_state import restore_replay_state
    from pidsmaker.tasks.batching import get_preprocessed_graphs
    from pidsmaker.factory import build_model
    from pidsmaker.utils.utils import get_device,set_seed,calculate_average_from_file
    from pidsmaker.utils.dataset_utils import get_rel2id
    cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    manifest=json.loads((cell/'manifest.json').read_text());ck=json.loads((cell/'checkpoint.json').read_text())
    report=json.loads((cell/'knee/report.json').read_text());threshold=report['policy']['threshold']
    assert cfg.training.num_epochs==12 and cfg.training.seed==0 and report['scope']=='retrospective_test'
    assert digest(cell/'resolved_config.yml')==manifest['config_sha256'] and digest(ck['path'])==report['identity']['checkpoint_sha256']
    pred=pd.read_json(cell/'knee/objects.jsonl',lines=True,precise_float=True).set_index('node_id')
    mapping=json.loads((cell/'uuid_mapping.json').read_text())
    assert pred.index.is_unique and all(mapping[str(i)]==u for i,u in pred.uuid.items())
    assert (pred.prediction==(pred.score>threshold)).all() and (pred.threshold==threshold).all()
    assert Counter(zip(pred['gt'],pred.prediction))=={(0,0):286815,(0,1):43,(1,0):71,(1,1):1}
    pred[['gt','score','prediction']].to_pickle(out/'pred.pkl')
    labels={int(i):r[1] for i,r in torch.load(Path(cfg.construction._dicts_dir)/'indexid2msg.pkl').items()}
    types={int(i):r[0] for i,r in torch.load(Path(cfg.construction._dicts_dir)/'indexid2msg.pkl').items()}
    sets=get_preprocessed_graphs(cfg);train,val,test,maxnodes=sets
    dev=get_device(cfg);set_seed(cfg,seed=0)
    model=build_model(train[0][0],device=dev,cfg=cfg,max_node_num=maxnodes,all_data=(val,test))
    load_evaluation_checkpoint(model,ck['path']);restore_replay_state(model,Path(ck['path']).parent/'replay_state.pt',required=True)
    feat=next(m for m in model.modules() if type(m).__name__=='GMAEFeatReconstruction')
    struct=next(m for m in model.modules() if type(m).__name__=='GMAEStructPrediction')
    matrix=Path(cfg.training._edge_losses_dir)/'test/model_epoch_11'
    files=sorted(matrix.glob('*.csv'),key=lambda p:int(p.stem.rsplit('_b',1)[1]))
    scale=calculate_average_from_file(Path(cfg.training._magic_dir)/'train_distance.txt')
    assert scale>0
    books=[{} for _ in range(4)];semantics={};op=get_rel2id(cfg,from_zero=True)
    audit={};peak={}
    for split,loader in [('train',train),('test',test)]:
        assert len(loader)==1
        base=out/split;base.mkdir()
        edges=0;nnodes=0
        for w,b in enumerate(loader[0]):
            ids,ct,sem,ins,outs,desc,x=context(b,labels,op,books,semantics)
            raw=b.original_edge_index.cpu().numpy()
            assert np.array_equal(raw,ids[b.edge_index.cpu().numpy()])
            edges+=raw.shape[1];nnodes+=len(ids)
            b=b.to(dev)
            with torch.no_grad():
                h=model.embed(b,inference=True)[0].cpu().numpy()
                sce,bce=diagnostics(model,b,feat,struct,w)
            if split=='test':
                f=pd.read_csv(files[w],float_precision='round_trip').set_index('node')
                assert f.index.is_unique and set(ids)==set(f.index)
                scores=f.loc[ids,'magic_score'].to_numpy()
                for i,s in zip(ids,scores):peak[i]=max(peak.get(i,-np.inf),s)
            else:
                rng=np.random.RandomState(20261126+w)
                ref=h[rng.choice(len(h),min(5000,len(h)),replace=False)]
                mean=ref.mean(0);sd=ref.std(0)
                scores=np.full(len(h),np.nan)
                if len(h)>=10 and (sd>0).all():
                    idx=NearestNeighbors(n_neighbors=10).fit((ref-mean)/sd)
                    scores=idx.kneighbors((h-mean)/sd)[0].mean(1)/scale
            np.savez(base/f'{w:04d}.npz',id=ids,context=ct,semantic=sem,degree=ins+outs,ins=ins,outs=outs,kind=np.array([types[int(i)] for i in ids]),window_nodes=np.full(len(ids),len(ids)),x=x,desc=desc,h=h,sce=sce,bce=bce,score=scores)
            if w%100==0:print(split,w,edges,flush=True)
        audit[split]=dict(windows=len(loader[0]),edges=edges,node_windows=nnodes)
    assert audit['train']['edges']==3426589 and audit['test']['edges']==2065670
    assert set(peak)==set(pred.index)
    err=max(abs(s-pred.at[i,'score']) for i,s in peak.items());assert err<1e-8
    audit.update(scale=scale,original_peak_max_error=err,checkpoint=ck['sha256'],threshold=threshold,raw_x_variants=len(books[0]),run=run.name,config=manifest['config_sha256'],knee_sha256=digest(cell/'knee/objects.jsonl'))
    (out/'summary.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]),Path(os.environ['RUN_OUTPUT_DIR']))
