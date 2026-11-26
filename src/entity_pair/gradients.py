"""Frozen epoch-11 gradients on native training batches, with train-only neighbors."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from neighbors import loader


def main(root, out):
    run = root / "cadets-velox-firstpaths-rebuild-20261126b"
    first = root / "pair-interference-s1-20261126a/output"
    second = root / "pair-interference-s2-20261126a/output"
    cell, cfg, labels, ops, (train, val, test, maxnodes) = loader(run)
    from baseline_matrix import digest, load_evaluation_checkpoint
    from pidsmaker.factory import build_model
    from pidsmaker.utils.utils import get_device, set_seed

    set_seed(cfg, seed=0)
    device = get_device(cfg)
    model = build_model(train[0][0], device=device, cfg=cfg, max_node_num=maxnodes, all_data=(val,test))
    ck = json.loads((cell / "checkpoint.json").read_text())
    assert digest(ck["path"]) == ck["sha256"] == "ad2345d526222df0a0848f27eb4ca7c432ec1b77e854f2f21245cc4a172da789"
    load_evaluation_checkpoint(model, ck["path"])
    model.train()
    assert len(model.objectives)==1 and cfg.training.grad_accumulation==1
    obj=next(m for m in model.modules() if type(m).__name__=="EdgeTypePrediction")
    assert obj.loss_fn.__name__=="cross_entropy" and not obj.balanced_loss
    keys, variants=pd.read_pickle(second/"training_variants.pkl")
    lookup={k:i for i,k in enumerate(keys)}
    queries=pd.read_pickle(first/"queries.pkl").query("canonical").drop_duplicates("node")
    targets=sorted({lookup[k] for k in zip(queries.src,queries.dst) if k in lookup})
    target_set=set(targets)
    vectors={}
    hist=defaultdict(Counter)
    anchors=defaultdict(Counter)
    batch_ids=[]
    for b in train[0]:
        ids=np.array([vectors.setdefault(x.tobytes(),len(vectors)) for x in b.x.numpy()])
        u,v=b.original_edge_index.tolist()
        pairs=np.array([lookup[labels[s],labels[d]] for s,d in zip(u,v)])
        batch_ids.append(pairs)
        a,z=ids[b.edge_index.numpy()]
        for i,r,x,y in zip(pairs,b.edge_type.argmax(1).tolist(),a,z):
            hist[i][r]+=1
            if i in target_set:
                anchors[i][x,y]+=1
    vec=np.array([np.frombuffer(x,np.float32) for x in vectors],dtype=float)
    triples=np.array([(i,a,z) for i in range(len(keys)) for a,z in sorted(variants[i])])
    p,a,z=triples.T
    starts=np.r_[0,1+np.flatnonzero(np.diff(p))]
    norm=np.einsum("ij,ij->i",vec,vec)
    denom=np.sqrt(norm[a]+norm[z])
    groups={}
    rng=np.random.default_rng(0)
    for i in targets:
        x,y=min(anchors[i],key=lambda k:(-anchors[i][k],k))
        assert (x,y) in variants[i]
        scores=np.maximum.reduceat(((vec@vec[x])[a]+(vec@vec[y])[z])/(denom*np.sqrt(norm[x]+norm[y])),starts)
        ranking=np.argsort(-scores.round(12),kind="stable")
        near=ranking[ranking!=i][:10]
        def modes(j):
            return {k for k,n in hist[j].items() if n==max(hist[j].values())}
        groups[i]={"T":[i],"C":[j for j in near if not modes(i)&modes(j)],
                   "S":[j for j in near if modes(i)==modes(j)],
                   "R":list(rng.choice(ranking[len(ranking)*3//4:],10,replace=False))}
    selected=set(j for g in groups.values() for js in g.values() for j in js)
    params=[p for p in model.parameters() if p.requires_grad]
    sums={}; effective={}; mass=Counter(); batchsum={}
    def grad(loss):
        return torch.cat([v.flatten() for v in torch.autograd.grad(loss,params,retain_graph=True)]).detach()
    # Capture the unreduced native loss from precisely the logits used for training.
    native=obj.loss_fn
    def capture(x,y,**kw):
        nonlocal edges
        edges=native(x,y,inference=True,weight=kw["weight"])
        return native(x,y,**kw)
    edges=None
    obj.loss_fn=capture
    for ix,(b,ids) in enumerate(zip(train[0],batch_ids)):
        positions=defaultdict(list)
        for row,j in enumerate(ids):
            if j in selected: positions[j].append(row)
        if not positions:continue
        b=b.to(device)
        loss=model(b)["loss"]
        assert torch.allclose(edges.mean(),loss)
        full=grad(loss)
        for j,rows in positions.items():
            v=grad(edges[rows].sum())
            for table,contribution in ((sums,v),(effective,v/len(ids))):
                if j not in table:table[j]=contribution.clone()
                else:table[j]+=contribution
            mass[j]+=len(rows)/len(ids)
            if j in target_set:
                batchsum[j]=batchsum.get(j,0)+full
        b.to("cpu")
    pd.to_pickle(dict(keys=keys,groups=groups, batch_ids=batch_ids),out/"plan.pkl")
    torch.save({j:v.cpu() for j,v in effective.items()},out/"effective.pt")
    rows=[]
    for i,g in groups.items():
        t=sums[i]/sum(hist[i].values())
        for group,js in g.items():
            if not js:continue
            n=sum(sum(hist[j].values()) for j in js)
            v=sum(sums[j] for j in js)/n
            direction=sum(effective[j] for j in js)
            dot=lambda x,y:torch.dot(x.double(),y.double()).item()
            rows.append(dict(src=keys[i][0],dst=keys[i][1],group=group,target_support=sum(hist[i].values()),
                support=n,pairs=len(js),cosine=dot(t,v)/(dot(t,t)*dot(v,v))**.5,norm=dot(v,v)**.5,
                projection= -dot(t,direction)/dot(t,t)**.5,
                dominance=-dot(direction,t)/dot(effective[i],t),batch_dot=dot(batchsum[i],t)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"gradients.csv",index=False)
    print(frame.groupby("group").cosine.describe(),flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(os.environ["RUN_OUTPUT_DIR"]))
