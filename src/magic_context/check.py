"""Independent type denominators and matched subgroup checks on saved frozen outputs."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, analysis = map(Path, sys.argv[1:])
out = Path(os.environ['RUN_OUTPUT_DIR'])
cols = ['context', 'id', 'degree', 'window_nodes', 'kind', 'score', 'semantic']
def collect(split):
    pieces = []
    for window, path in enumerate(sorted((src/split).glob('*.npz'))):
        with np.load(path) as d:
            pieces.append(pd.DataFrame({c:d[c] for c in cols}).assign(window=window))
    return pd.concat(pieces, ignore_index=True)

train = collect('train')
test = collect('test')
counts = train.groupby('context').agg(instances=('id','size'), nodes=('id','nunique'), windows=('window','nunique'))
counts.join(test.groupby('context').id.size().rename('test_instances'), how='outer').fillna(0).astype(int).to_csv(out/'CONTEXT_TYPE_COUNTS.csv.gz')
nodes = test.sort_values(['score','window'], ascending=[False,True]).drop_duplicates('id').set_index('id')
nodes = nodes.join(pd.read_pickle(src/'pred.pkl')[['gt','prediction']])
nodes['N'] = nodes.context.map(counts.instances).fillna(0)
nodes['activity'] = test.groupby('id').degree.sum()
tn = nodes[(nodes['gt']==0)&(nodes.prediction==0)]
fp = nodes[(nodes['gt']==0)&(nodes.prediction==1)]
matches = []
for i,r in fp.iterrows():
    candidates = tn[tn.kind==r.kind]
    fields = ['activity','degree','window_nodes']
    delta = (np.log1p(candidates[fields])-np.log1p(r[fields].astype(float))).abs().sum(axis=1)
    for j in delta.sort_index().nsmallest(3).index:
        matches.append(dict(node=i,control=j,N=r.N,tn_N=candidates.at[j,'N'],distance=delta[j]))
pd.DataFrame(matches).to_csv(out/'SUPPORT_UNMATCHED_ON_EXPOSURE.csv',index=False)

pairs = pd.read_csv(analysis/'MATCHED_FP_TN_ANALYSIS.csv')
ev = pd.read_csv(analysis/'ALL_FP_CONTEXT_EVIDENCE.csv').set_index('id')
near = pd.read_csv(analysis/'SIMILAR_CONTEXT_ANALYSIS.csv')
parts=[]
for (space,k),g in near.groupby(['space','K']):
    g = g.set_index('node')
    for category,m in pairs.groupby(pairs.case.map(ev.category)):
        for col in ['structural_delta','diff_target_mass','neighbor_N','target_delta']:
            paired = pd.DataFrame({'case':m.case,'fp':m.case.map(g[col]),'tn':m.control.map(g[col])}).groupby('case').mean()
            parts.append(dict(space=space,K=k,category=category,metric=col,n=len(paired),fp_mean=paired.fp.mean(),tn_mean=paired.tn.mean(),higher=int((paired.fp>paired.tn+1e-8).sum()),lower=int((paired.fp<paired.tn-1e-8).sum())))
pd.DataFrame(parts).to_csv(out/'FIT_NEIGHBOR_CONTRASTS.csv',index=False)
mal = nodes[nodes['gt']==1].copy()
mal['semantic_N']=mal.semantic.map(train.semantic.value_counts()).fillna(0)
mal[['prediction','context','N','semantic','semantic_N','score']].to_csv(out/'MALICIOUS_SUPPORT.csv')
