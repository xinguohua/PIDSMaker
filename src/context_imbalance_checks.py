"""Audit actual loader inputs and perform operation-stratified sensitivity checks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import torch
import yaml
from yacs.config import CfgNode

KEY = ['op','src_label','dst_label']


def check(cell, source, out, semantic):
    sys.path.insert(0,str(cell.parents[1]/'code'))
    from pidsmaker.tasks.batching import get_preprocessed_graphs
    from pidsmaker.utils.dataset_utils import get_rel2id
    cfg = CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    # Check each training batch against the exact row sequence in the native graph.
    train,_,_,_ = get_preprocessed_graphs(cfg)
    assert len(train)==1
    rel = get_rel2id(cfg,from_zero=True)
    counts = Counter()
    for w in pd.read_csv(source/'train_windows.csv').itertuples():
        graph = torch.load(Path(cfg.transformation._graphs_dir)/('graph_'+w.window[:10])/w.window,map_location='cpu')
        rows = iter(graph.edges(data=True))
        n = 0
        for b in train[0][w.first_batch:w.first_batch+w.batches]:
            src,dst = b.original_edge_index.cpu().tolist()
            types = b.edge_type.argmax(1).cpu().tolist()
            for u,v,o,t,y in zip(src,dst,types,b.t.cpu().tolist(),b.y.cpu().reshape(-1).tolist()):
                a,z,e = next(rows)
                assert (int(a),int(z),e['label'],int(e['time']),int(e['y']))==(u,v,rel[o],t,y)
                counts[rel[o],graph.nodes[a]['label'],graph.nodes[z]['label']]+=1
                n+=1
        assert n==w.edges and next(rows,None) is None
    tr = pd.read_csv(source/'training.csv.gz',keep_default_na=False)
    assert tr.set_index(KEY).N.to_dict()==counts
    (out/'loader_check.json').write_text(json.dumps(dict(batches=len(train[0]),edges=sum(counts.values()),triples=len(counts))))
    # Independent check for a nonuniform FPR across the five prespecified support bins.
    # Simulate allocating each operation's fixed FP count among its observed bin sizes.
    rng = np.random.default_rng(20261126)
    rates = pd.read_csv(source/'peak_rates.csv')
    results = []
    for op,g in rates.groupby('op'):
        g=g[g.nodes>0]
        n=g.nodes.to_numpy()
        fp=g.FP.to_numpy()
        reps=20000
        p=np.nan
        if fp.sum()>0 and len(g)>1 and fp.sum()<n.sum():
            expected=n*fp.sum()/n.sum()
            def stat(x):
                return (((x-expected)**2)/(expected*(1-fp.sum()/n.sum()))).sum(axis=-1)
            samples=rng.multivariate_hypergeometric(n,int(fp.sum()),size=reps)
            p=(1+(stat(samples)>=stat(fp)-1e-10).sum())/(1+reps)
        lo=g[g.bin.isin(['0','1-9'])][['FP','TN']].sum().astype(int)
        hi=g[g.bin=='1000+'][['FP','TN']].sum().astype(int)
        test=fisher_exact([lo,hi]) if lo.sum()*hi.sum()>0 else (np.nan,np.nan)
        results.append(dict(op=op,fp=int(fp.sum()),heterogeneity_p=p,low_fp=lo.FP,low_tn=lo.TN,
                            high_fp=hi.FP,high_tn=hi.TN,low_vs_high_OR=test[0],low_vs_high_p=test[1]))
    tests=pd.DataFrame(results)
    for k in ['heterogeneity_p','low_vs_high_p']:
        ordered=tests[k].dropna().sort_values()
        tests.loc[ordered.index,k+'_holm']=(ordered*np.arange(len(tests),len(tests)-len(ordered),-1)).cummax().clip(upper=1)
    tests.to_csv(out/'peak_tests.csv',index=False)
    peaks=pd.read_csv(source/'normal_node_peaks.csv.gz',keep_default_na=False)
    peaks['relative_support']=pd.cut(peaks.Qc,[-1,0,.01,.1,.5,1],labels=['0','(0,.01]','(.01,.1]','(.1,.5]','(.5,1]'])
    relstats=peaks.groupby(['op','bin','relative_support'],observed=True).fp.agg(['sum','size'])
    relstats.to_csv(out/'relative_absolute.csv')
    sem=sqlite3.connect(f'file:{semantic}?mode=ro',uri=True)
    checked=[]
    cases=pd.read_csv(source/'cases.csv')
    windows=pd.read_csv(source/'test_windows.csv').set_index('w')
    sem.execute('attach database ? as native',(f'file:{semantic.parent}/native.sqlite?mode=ro',))
    for r in cases.itertuples():
        # The historical source-offset map contributes only raw provenance locations.
        name=windows.loc[r.w,'window']
        row=sem.execute('select file,offset from source s join native.windows w using(w) where w.name=? and w.split=? and s.row=?',(name,'test',r.row)).fetchone()
        assert row is not None
        with open(row[0],'rb') as stream:
            stream.seek(row[1])
            ev=next(iter(json.loads(stream.readline())['datum'].values()))
        peak=peaks[peaks.node==r.node].iloc[0]
        assert ev['type']==r.op and ev['uuid']==r.uuid and int(ev['timestampNanos'])==peak.ns
        checked.append(dict(node=r.node,role=r.role,op=r.op,w=r.w,row=r.row,file=row[0],offset=row[1],event_uuid=r.uuid))
    pd.DataFrame(checked).to_csv(out/'raw_witnesses.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cell','source','semantic']:
        p.add_argument(name,type=Path)
    a=p.parse_args()
    check(a.cell,a.source,Path(os.environ['RUN_OUTPUT_DIR']),a.semantic)
