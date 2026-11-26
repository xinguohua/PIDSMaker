"""Reporting checks on cached per-object tables; no additional model replay."""
from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from masked_analysis import unit
d,raw=map(Path,sys.argv[1:])
out=Path(os.environ['RUN_OUTPUT_DIR'])
g=pd.read_csv(d/'G3_groups.csv').dropna(subset=['minor_bias']).set_index('g0')
freq=pd.read_csv(d/'LOCAL_TARGET_DISTRIBUTION.csv')
counts=pd.read_csv(d/'TARGET_DISCREPANCY_ANALYSIS.csv').set_index('g0')
w=freq.groupby('g0').batch_weight.transform('max')
weighted=freq[(freq.batch_weight==w)].groupby('g0').target.agg(set)
g['weighted_major_ok']=[int(m in weighted[k]) for k,m in g.major.items()]
obj=pd.read_csv(d/'DIRECTIONAL_RECONSTRUCTION_BIAS.csv')
pairs=pd.read_csv(d/'MATCHES.csv')
obs=pd.read_csv(d/'FP_TN_LEARNING_BIAS_ANALYSIS.csv').set_index('id')
hs=obs.filter(regex='^h[0-9]').columns
p=pairs.copy()
for side,col in [('case','case'),('control','control')]:
    p[side+'_eligible']=p[col].map(obs.eligible)
p['cosine_embedding']=np.sum(unit(obs.loc[p.case,hs].to_numpy(float))*unit(obs.loc[p.control,hs].to_numpy(float)),axis=1)
p['embedding_L2']=np.linalg.norm(obs.loc[p.case,hs].to_numpy()-obs.loc[p.control,hs].to_numpy(),axis=1)
p.to_csv(out/'PAIRED_REPRESENTATIONS.csv',index=False)
records=[]
for name,sub in [('all',g),('nonzero_indegree',g[g.indeg>0]),('batch_weight_agrees',g[g.weighted_major_ok==1]),('at_least_ten_minor',g[g.minor_n>=10]),('at_least_two_nodes',g[g.unique_nodes>=2])]:
    records.append(dict(subgroup=name,n=len(sub),major_n=sub.major_n.sum(),minor_n=sub.minor_n.sum(),bias=np.average(sub.minor_bias,weights=sub.minor_n),reverse=np.average(sub.reverse,weights=sub.major_n),macro_bias=sub.minor_bias.mean(),rho=spearmanr(sub.ratio,sub.minor_bias)[0]))
pd.DataFrame(records).to_csv(out/'CORE_ROBUSTNESS.csv',index=False)
# Independently verify exact-cell reconstruction agreement and supply target dictionary for every test control.
cols=list(obj.filter(regex='^p[0-9]').columns)
sub=obj[obj.g0.isin(g.index)]
spread=(sub.groupby('g0')[cols].max()-sub.groupby('g0')[cols].min()).abs().max(axis=1)
examples=[]
for tag,k in [('largest',g.instances.idxmax()),('positive_nonempty',g[g.indeg.gt(0)&g.minor_bias.gt(.9)].instances.idxmax()),('counterexample',g[g.minor_bias.eq(0)].instances.idxmax())]:
    for target,rows in sub[sub.g0==k].groupby('target'):
        r=rows.iloc[0]
        examples.append(dict(case=tag,context=k,target=target,n=len(rows),node=r.id,window=r.window,mean_own=rows.loss.mean(),mean_major=rows.to_major.mean(),positive=int((rows.margin>1e-9).sum()),output=json.dumps([float(x) for x in r[cols]]),major=g.at[k,'major']))
pd.DataFrame(examples).to_csv(out/'CASE_TARGETS.csv',index=False)
extra=[]
for win,t in obs.reset_index().groupby('window'):
    for file in (raw/'test0').glob(f'{win:04d}-*.npz'):
        with np.load(file) as f:
            positions=pd.Index(f['id']).get_indexer(t.id)
            for rec,pos in zip(t.itertuples(),positions):
                if pos<0:continue
                extra.append(dict(id=rec.id,target=rec.target,true_vector=json.dumps(f['target'][pos].tolist()),window=win,draw=int(file.stem.split('-')[1])))
assert len(extra)==len(obs)
pd.DataFrame(extra).to_csv(out/'TEST_TARGET_VECTORS.csv',index=False)
r=dict(max_exact_group_output_spread=spread.max(),median_exact_group_output_spread=spread.median(),eligible_both_matches=int((p.case_eligible&p.control_eligible).sum()),cosine_embedding_median=p.cosine_embedding.median(),unique_major_weight_agrees=int(g.weighted_major_ok.sum()),mixed_multi_node=int(((counts.target_types>1)&(counts.unique_nodes>1)).sum()))
(out/'audit.json').write_text(json.dumps(r,indent=2))
