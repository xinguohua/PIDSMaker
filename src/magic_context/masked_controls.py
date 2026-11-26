"""Outcome-blind, dissimilar input pairs plus support and normal-node cross-checks."""
from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from masked_analysis import read,dist,unit

folder,replay=map(Path,sys.argv[1:3])
out=Path(os.environ['RUN_OUTPUT_DIR'])
data=read(replay/'train0')
stats=pd.read_csv(folder/'TARGET_DISCREPANCY_ANALYSIS.csv').set_index('g0')
freq=pd.read_csv(folder/'LOCAL_TARGET_DISTRIBUTION.csv')
vec=pd.read_csv(folder/'TARGET_VECTORS.csv')
ds=unit(vec.filter(regex='^y[0-9]').to_numpy(float))
data=data.join(stats[['major']],on='g0')
mom=data.groupby('g0')[list(data.filter(regex='^m[0-9]').columns)].mean()
pure=stats[stats.target_types==1].index.to_numpy()
rng=np.random.default_rng(1203)
pairs=rng.permutation(pure)[:2*(len(pure)//2)].reshape(-1,2)
a,b=pairs.T
difference=np.linalg.norm(mom.loc[a].to_numpy()-mom.loc[b].to_numpy(),axis=1)
cut=np.quantile(difference,.75)
ok=(difference>=cut)&(difference>1e-9)&(stats.loc[a].indeg.values!=stats.loc[b].indeg.values)
lookup=data.groupby('g0').indices
def loss(sub,target):
    return dist(unit(sub.filter(regex='^p[0-9]').to_numpy(float)),ds[target])
base_cut=json.loads((folder/'summary.json').read_text())['cutoff']
rows=[]
for (a,b),delta in zip(pairs[ok],difference[ok]):
    sa,sb=stats.loc[a],stats.loc[b]
    if sa.instances<sb.instances:a,b,sa,sb=b,a,sb,sa
    ma,mi=int(sa.major),int(sb.major)
    if sa.instances<4*sb.instances or dist(ds[ma],ds[mi])<base_cut-1e-9:continue
    am=data.iloc[lookup[a]];bm=data.iloc[lookup[b]]
    l1,l2=loss(am,ma),loss(bm,mi)
    rows.append(dict(major_context=a,minor_context=b,major=ma,minor=mi,major_n=len(am),minor_n=len(bm),input_distance=delta,major_loss=l1.mean(),minor_loss=l2.mean(),bias=(l2-loss(bm,ma)>1e-9).mean(),reverse=(l1-loss(am,mi)>1e-9).mean(),kind=sa.modal_kind,window_size=sa.mean_window_size))
g4=pd.DataFrame(rows)
g4.to_csv(out/'G4_RANDOM_PAIRS.csv',index=False)
print('low cutoff',cut,'eligible',len(rows),flush=True)
matches=[]
g3=pd.read_csv(folder/'G3_groups.csv').dropna(subset=['minor_bias'])
for r in g3.itertuples():
    if not len(g4):break
    total=g4.major_n+g4.minor_n
    near=g4[(g4.kind==r.modal_kind)&(np.abs(np.log(total/r.instances))<=np.log(2))]
    if not len(near):continue
    distance=np.abs(np.log((near.major_n+near.minor_n)/r.instances))+np.abs(np.log(near.window_size/r.mean_window_size))+np.abs(np.log((near.major_n/near.minor_n)/r.ratio))
    best=near.loc[distance.idxmin()]
    matches.append(dict(g3=r.g0,major_control=best.major_context,minor_control=best.minor_context,g3_bias=r.minor_bias,g4_bias=best.bias))
pd.DataFrame(matches).to_csv(out/'G3_G4_MATCHES.csv',index=False)
test=pd.read_csv(folder/'NORMAL_POPULATION.csv.gz')
matched=pd.read_csv(folder/'MATCHES.csv')
obs=test.set_index('id')
records=[]
for k in ['N','eligible','biased','loss','density_distance','score','margin']:
    joined=pd.DataFrame(dict(case=matched.case,fp=matched.case.map(obs[k]),tn=matched.control.map(obs[k])))
    if k=='margin':
        joined=joined[matched.case.map(obs.eligible)&matched.control.map(obs.eligible)]
    j=joined.groupby('case')[['fp','tn']].mean().dropna()
    records.append(dict(metric=k,n=len(j),fp_mean=j.fp.mean(),tn_mean=j.tn.mean(),fp_median=j.fp.median(),tn_median=j.tn.median(),fp_higher=(j.fp>j.tn+1e-9).sum(),fp_lower=(j.fp<j.tn-1e-9).sum()))
pd.DataFrame(records).to_csv(out/'MATCHED_DETECTION_SUMMARY.csv',index=False)
# Exact matching on SCE directions collapses collinear targets. Re-evaluate core subgroup.
from masked_analysis import summarize,pooled
import masked_analysis
masked_analysis.pcols=list(data.filter(regex='^p[0-9]').columns)
axes,axis=np.unique(np.round(ds,10),axis=0,return_inverse=True)
tidx=pd.MultiIndex.from_frame(vec.filter(regex='^y[0-9]'))
# Use original target indices from the identical per-object row order.
obj=pd.read_csv(folder/'SIMILAR_INPUT_GROUPS.csv',usecols=['window','id','target','g0'])
assert np.array_equal(obj.id,data.id) and (obj.g0==data.g0).all()
data['axis']=axis[obj.target]
data['loss']=dist(unit(data[masked_analysis.pcols].to_numpy(float)),ds[obj.target])
gs,_,_=summarize(data,'g0','axis',axes)
rows=[dict(estimated_using='direction',ratio=ratio,**pooled(gs[(gs.ratio>=ratio)&(gs.max_major_SCE>=base_cut-1e-9)])) for ratio in [2,4,8]]
pd.DataFrame(rows).to_csv(out/'DIRECTION_SUPPORT_SENSITIVITY.csv',index=False)
# Link directional bias to natural counts; remove high-leverage cases as a sensitivity check.
for i in [1,5,10]:
    s=g3.sort_values('instances',ascending=False).iloc[i:]
    print('drop',i,len(s),np.average(s.minor_bias,weights=s.minor_n),flush=True)
fp=test[test.prediction==1]
summary=dict(moment_cutoff=cut,eligible_low_pairs=len(g4),eligible_low_pair_candidates=int(ok.sum()),pairs_sampled=len(pairs),g3_g4_matches=len(matches),fp_peak_exact_seen=int(fp.N.gt(0).sum()),tn_peak_exact_seen=int(test.loc[test.prediction==0,'N'].gt(0).sum()),bias_score_rho=float(spearmanr(test.loc[test.eligible,'margin'],test.loc[test.eligible,'score'])[0]))
(out/'control_summary.json').write_text(json.dumps(summary,indent=2))
