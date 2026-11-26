"""Stage 2. Exact capped cosine retrieval from verified reindexed input vectors."""
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from yacs.config import CfgNode

def loader(run):
    sys.path[:0]=[str(run/'code'),str(run/'code/src')]
    from pidsmaker.tasks.batching import get_preprocessed_graphs
    from pidsmaker.utils.dataset_utils import get_rel2id
    cell=run/'output/CADETS_E3-velox-full'
    cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    labels={int(n):v[1] for n,v in torch.load(Path(cfg.construction._dicts_dir)/'indexid2msg.pkl').items()}
    return cell,cfg,labels,get_rel2id(cfg,from_zero=True),get_preprocessed_graphs(cfg)

def main(run,source,first,out):
    cell,cfg,labels,ops,(train,val,test,maxnodes)=loader(run)
    train_table=pd.read_csv(source/'condition_operations.csv',keep_default_na=False)
    h={k:dict(zip(g.op,g.n.astype(int))) for k,g in train_table.groupby(['src','dst'])}
    keys=sorted(h); cid={k:i for i,k in enumerate(keys)}
    counts=Counter(); vectors={}; variants=defaultdict(set); exact=Counter()
    for bi,b in enumerate(train[0]):
        ids=[vectors.setdefault(x.tobytes(),len(vectors)) for x in b.x.cpu().numpy()]
        u,v=b.original_edge_index.cpu().tolist()
        a,z=b.edge_index.cpu().tolist()
        rel=b.edge_type.argmax(1).cpu().tolist()
        for s,d,i,j,r in zip(u,v,a,z,rel):
            c=cid[labels[s],labels[d]]
            counts[c,ops[r]]+=1
            variants[c].add((ids[i],ids[j]))
            exact[ids[i],ids[j],ops[r]]+=1
    assert counts=={(cid[k],o):n for k,v in h.items() for o,n in v.items()}
    baseline=pd.read_csv(source/'input_operations.csv')
    assert exact=={(s,d,o):n for s,d,o,n in baseline.itertuples(index=False,name=None)}
    vec=np.array([np.frombuffer(x,np.float32) for x in vectors],dtype='float64')
    assert np.isfinite(vec).all()
    variants_table=[(i,a,b) for i in range(len(keys)) for a,b in sorted(variants[i])]
    p,si,di=np.array(variants_table).T
    starts=np.r_[0,np.flatnonzero(np.diff(p))+1]
    norms=np.einsum('ij,ij->i',vec,vec)
    denominator=np.sqrt(norms[si]+norms[di])
    assert np.all(denominator>0)
    qs=pd.read_pickle(first/'queries.pkl')
    origins={}
    for r in qs.itertuples():
        key=(r.src,r.dst,r.vec_src,r.vec_dst)
        if key in origins:continue
        b=test[0][r.batch]
        pos=b.edge_index[:,r.row].cpu().tolist()
        u,v=b.original_edge_index[:,r.row].cpu().tolist()
        assert (u,v,labels[u],labels[v])==(r.u,r.v,r.src,r.dst)
        vs=b.x[pos].cpu().numpy()
        for x,idx in zip(vs,(r.vec_src,r.vec_dst)):
            assert vectors.get(x.tobytes(),idx)==idx and (idx<len(vec))==(x.tobytes() in vectors)
        origins[key]=vs
    nvec=len(vec)
    same_src=defaultdict(list);same_dst=defaultdict(list)
    for i,(s,d) in enumerate(keys):same_src[s].append(i);same_dst[d].append(i)
    rows=[];lookup={}
    query_keys=list(origins)
    for start in range(0,len(query_keys),24):
        batchkeys=query_keys[start:start+24]
        x=np.array([origins[k] for k in batchkeys],dtype='float64')
        dots_src=x[:,0]@vec.T; dots_dst=x[:,1]@vec.T
        for j,key in enumerate(batchkeys):
            s,d,vs,vd=key
            length=np.linalg.norm(x[j])
            assert length>0
            sim=(dots_src[j,si]+dots_dst[j,di])/(length*denominator)
            scores=np.maximum.reduceat(sim,starts)
            own=cid.get((s,d),-1)
            for method,candidates in [('feature',np.arange(len(keys))),('same_src',np.array(same_src[s],dtype=int)),('same_dst',np.array(same_dst[d],dtype=int))]:
                other=candidates[candidates!=own]
                # Stable secondary sort by lexical directed pair, independent of outcomes and operation.
                selected=other[np.lexsort((other,-np.round(scores[other],12)))[:10]]
                lookup[key,method]=list(selected)
                for rank,i in enumerate(selected,1):
                    nh=h[keys[i]]
                    rows.append((s,d,vs,vd,method,rank,*keys[i],float(scores[i]),sum(nh.values()),json.dumps(nh,sort_keys=True)))
    result=pd.DataFrame(rows,columns='src dst vec_src vec_dst method rank neighbor_src neighbor_dst cosine train_count train_operations'.split())
    result.to_csv(out/'SIMILAR_CONTEXT_ANALYSIS.csv',index=False)
    pd.to_pickle((keys,variants),out/'training_variants.pkl')
    rows=[]
    for r in qs.itertuples():
        key=(r.src,r.dst,r.vec_src,r.vec_dst)
        for method in ('feature','same_src','same_dst'):
            nh=Counter();high=0
            for i in lookup[key,method]:
                other=h[keys[i]];nh.update(other)
                if sum(other.values())>=1000 and max(other.values())==other.get(r.guess,0) and other.get(r.guess,0)>other.get(r.op,0):high+=1
            rows.append((r.node,r.batch,r.row,method,r.op,r.guess,nh[r.op],nh[r.guess],sum(nh.values()),high,json.dumps(nh)))
    pd.DataFrame(rows,columns='node batch row method op guess true_n guess_n total hf_other_mode hist'.split()).to_pickle(out/'query_neighbors.pkl')
    (out/'summary.json').write_text(json.dumps(dict(training_pairs=len(keys),training_variants=len(p),exact_vectors=nvec,unique_queries=len(origins),train_events=sum(counts.values()),neighbors=len(result)),indent=2))
if __name__=='__main__':main(*map(Path,sys.argv[1:]),Path(os.environ['RUN_OUTPUT_DIR']))
