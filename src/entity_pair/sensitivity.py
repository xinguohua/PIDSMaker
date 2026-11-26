"""Reweight fixed neighbors without changing retrieval; quantify matching balance."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
source,first,second=map(Path,sys.argv[1:])
out=Path(os.environ['RUN_OUTPUT_DIR'])
qs=pd.read_pickle(first/'queries.pkl')
canon=qs[qs.canonical].set_index('node')
tr=pd.read_csv(source/'condition_operations.csv',keep_default_na=False).groupby(['src','dst']).total.first().to_dict()
nn=pd.read_csv(second/'SIMILAR_CONTEXT_ANALYSIS.csv',keep_default_na=False)
f=nn[nn.method=='feature']
groups={key:[(r.train_count,json.loads(r.train_operations)) for r in g.itertuples()] for key,g in f.groupby(['src','dst','vec_src','vec_dst'])}
def measure(r,alternative):
    n=tr.get((r.src,r.dst),0)
    neighbors=groups[r.src,r.dst,r.vec_src,r.vec_dst]
    scores=[(size,h.get(alternative,0),h.get(r.op,0),max(h.values())) for size,h in neighbors]
    return dict(pooled=sum(a-b for _,a,b,_ in scores)>0,more_frequent=any(size>n and a==m and a>b for size,a,b,m in scores),
                at_least_100=any(size>=100 and a==m and a>b for size,a,b,m in scores),at_least_1000=any(size>=1000 and a==m and a>b for size,a,b,m in scores))
rows=[]
for a,b in pd.read_csv(first/'matched.csv').itertuples(index=False,name=None):
    x,y=canon.loc[a],canon.loc[b]
    n=tr.get((x.src,x.dst),0);m=tr.get((y.src,y.dst),0)
    for k,v in measure(x,x.guess).items():
        rows.append(dict(node=a,control=b,rule=k,fp=float(v),tn=float(measure(y,x.guess)[k]),op=x.op,
                         same_kinds=x.src.split()[0]==y.src.split()[0] and x.dst.split()[0]==y.dst.split()[0],
                         freq_ratio=max(n+1,m+1)/min(n+1,m+1),degree_ratio=max(x.degree,y.degree)/min(x.degree,y.degree)))
df=pd.DataFrame(rows)
summary=[]
for label,sub in [('all',df),('both_types_and_2x',df[df.same_kinds&(df.freq_ratio<=2)&(df.degree_ratio<=2)])]:
    for k,g in sub.groupby('rule'):
        v=g.groupby('node')[['fp','tn']].mean()
        summary.append(dict(controls=label,rule=k,fp_nodes=len(v),distinct_tn=g.control.nunique(),FP=v.fp.mean(),TN=v.tn.mean()))
pd.DataFrame(summary).to_csv(out/'neighbor_sensitivity.csv',index=False)
g=df[df.rule=='at_least_1000']
(out/'summary.json').write_text(json.dumps(dict(comparisons=len(g),same_types=int(g.same_kinds.sum()),median_frequency_ratio=g.freq_ratio.median(),median_degree_ratio=g.degree_ratio.median())))
