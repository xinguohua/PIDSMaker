"""Compute complete denominators from the verified native event and input tables."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source = Path(sys.argv[1])
out = Path(os.environ['RUN_OUTPUT_DIR'])
protocol = json.loads((source/'protocol.json').read_text())
threshold = protocol['threshold']
nodes = pd.read_csv(source/'nodes.csv',float_precision='round_trip').set_index('node')
normal = nodes[nodes['gt']==0]
train = pd.read_csv(source/'condition_operations.csv',keep_default_na=False)
exact = pd.read_csv(source/'input_operations.csv')
keys = ['src','dst']
conditions = train.drop_duplicates(keys)[keys+['maximum','modes','operations','exact_pairs']]
counts = train[keys+['op','n']]
bins = ['unseen_condition','unseen_operation','lt01','lt10','lt50','lt100','mode']
types = ['low_to_high','high_to_low','equal_frequency','unsupported','correct','other']
def support_bin(df):
    n,m = df.n,df.maximum
    return np.select([m==0,n==0,n*100<m,n*10<m,n*2<m,n<m],bins[:-1],default='mode')
def events():
    return pd.read_csv(source/'test_events.csv.gz',chunksize=100000,float_precision='round_trip',keep_default_na=False)
def endpoints(df):
    return pd.concat([df.assign(node=df.u),df[df.u!=df.v].assign(node=lambda x:x.v)],ignore_index=True)
max_id = int(nodes.index.max())+1
maximum = np.full(max_id,-np.inf)
degree = np.zeros(max_id,dtype=np.int64)
for frame in events():
    e = endpoints(frame[['u','v','loss']])
    np.maximum.at(maximum,e.node,e.loss)
    np.add.at(degree,e.node,1)
max_error = float(np.abs(maximum[normal.index]-normal.score).max())
assert max_error < 1e-6
rates = Counter()
peak_chunks = []
for frame in events():
    df = frame.merge(conditions,on=keys,how='left',validate='m:1').merge(counts,on=keys+['op'],how='left',validate='m:1')
    df[['maximum','n']] = df[['maximum','n']].fillna(0).astype(int)
    df['bin'] = support_bin(df)
    y0 = df.y.eq(0)
    both = df.u.isin(normal.index)&df.v.isin(normal.index)
    for scope,mask in [('native_y0',y0),('both_normal_y0',y0&both)]:
        part = df[mask].assign(high=lambda x:x.loss>threshold)
        rates.update({(scope,*k):int(v) for k,v in part.groupby(['bin','high']).size().items()})
    e = endpoints(df)
    peak_chunks.append(e[e.node.isin(normal.index)&(e.loss==maximum[e.node])].copy())
peaks = pd.concat(peak_chunks,ignore_index=True)
peaks = peaks.merge(counts.rename(columns={'op':'guess','n':'guess_n'}),how='left',on=keys+['guess'],validate='m:1')
peaks.guess_n = peaks.guess_n.fillna(0).astype(int)
peaks['confusion'] = np.select([peaks.op==peaks.guess,(peaks.n==0)|(peaks.maximum==0),peaks.guess_n>peaks.n,
                                peaks.guess_n<peaks.n,peaks.guess_n==peaks.n],['correct','unsupported','low_to_high','high_to_low','equal_frequency'],'other')
peaks['guess_mode'] = (peaks.maximum>0)&(peaks.guess_n==peaks.maximum)
peaks['q'] = peaks.n/peaks.maximum.replace(0,np.nan)
pair = ['vec_src','vec_dst']
estate = exact.groupby(pair).agg(input_total=('n','sum'),input_ops=('op','size'),input_max=('n','max')).reset_index()
peaks = peaks.merge(estate,on=pair,how='left',validate='m:1').merge(exact.rename(columns={'n':'input_true_n'}),on=pair+['op'],how='left',validate='m:1')
peaks = peaks.join(normal[['uuid','score','prediction']],on='node')
peaks['degree']=degree[peaks.node]
peaks['group']=np.where(peaks.prediction==1,'FP','TN')
peaks.sort_values(['node','batch','row'],inplace=True)
groups=peaks.groupby('node')
peaks['peak_count']=groups.batch.transform('size')
peaks['mixed_bin']=groups.bin.transform('nunique')>1
peaks['mixed_confusion']=groups.confusion.transform('nunique')>1
peaks['canonical']=~peaks.node.duplicated()
peaks.to_csv(out/'all_tied_peaks.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
main=peaks[peaks.canonical].copy()
assert len(main)==len(normal) and main.prediction.sum()==protocol['FP']
main.to_csv(out/'node_peaks.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
peaks[peaks.prediction==1].to_csv(out/'fp_peak_confusions.csv',index=False)
def cross(col,order):
    t=pd.crosstab(main[col],main.group).reindex(index=order,columns=['FP','TN'],fill_value=0)
    t['FPR']=t.FP/(t.FP+t.TN)
    return t
tab=cross('bin',bins)
tab.to_csv(out/'node_support.csv')
cross('confusion',types).to_csv(out/'confusion.csv')
main['tie_bin']=np.where(main.mixed_bin,'Mixed',main.bin)
cross('tie_bin',bins+['Mixed']).to_csv(out/'tie_sensitivity.csv')
main['tie_confusion']=np.where(main.mixed_confusion,'Mixed',main.confusion)
cross('tie_confusion',types+['Mixed']).to_csv(out/'confusion_sensitivity.csv')
rows=[]
for scope in ['native_y0','both_normal_y0']:
    for b in bins:
        high=rates[scope,b,True]
        n=high+rates[scope,b,False]
        rows.append((scope,b,n,high,high/n if n else None))
pd.DataFrame(rows,columns=['scope','bin','events','high','rate']).to_csv(out/'event_rates.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(range(7),100*tab.FPR,color='#276a83')
ax.set_xticks(range(7))
ax.set_xticklabels(['New pair','New op','(0,.01)','[.01,.1)','[.1,.5)','[.5,1)','1'])
ax.set_xlabel('Peak event: local training support')
ax.set_ylabel('Node false-positive rate (%)')
fig.tight_layout()
fig.savefig(out/'support_fpr.png',dpi=180)
plt.close(fig)
# Exact controls; compare mode with seen nonmode across conditions with the same true count.
for side in keys:
    main[side+'_kind']=main[side].str.split().str[0]
    main[side+'_missing']=main[side].str.contains(r'(?:^|\s)(?:None|null|unknown)(?:\s|$)',case=False,regex=True)
strata=['op','src_kind','dst_kind','src_missing','dst_missing','n','degree']
eligible=main[main.n>0].copy()
eligible['nonmode']=eligible.n<eligible.maximum
g=eligible.groupby(strata+['nonmode']).prediction.agg(['size','sum']).unstack('nonmode').dropna()
g.columns=['mode_nodes','nonmode_nodes','mode_fp','nonmode_fp']
g=g.reset_index()
g.to_csv(out/'matched_strata.csv',index=False)
covered=eligible.merge(g[strata],on=strata,validate='m:1')
cases=[]
for _,part in covered.groupby(strata):
    for fp in part[part.prediction==1].sort_values('node').head(1).itertuples():
        other=part[(part.prediction==0)&(part.nonmode!=fp.nonmode)].sort_values('node')
        if len(other):
            tn=other.iloc[0]
            cases.append(dict(fp_uuid=fp.uuid,tn_uuid=tn.uuid,op=fp.op,n=fp.n,degree=fp.degree,fp_src=fp.src,fp_dst=fp.dst,
                              tn_src=tn.src,tn_dst=tn.dst,fp_q=fp.q,tn_q=tn.q,fp_score=fp.score,tn_score=tn.score))
pd.DataFrame(cases,columns='fp_uuid tn_uuid op n degree fp_src fp_dst tn_src tn_dst fp_q tn_q fp_score tn_score'.split()).to_csv(out/'matched_cases.csv',index=False)
main[main.prediction==1].groupby(['src','dst','op','guess','bin'],dropna=False).size().sort_values(ascending=False).head(12).to_csv(out/'frequent_fp.csv')
details={}
for group,x in main.groupby('group'):
    details[group]=dict(nodes=len(x),ties=int((x.peak_count>1).sum()),mixed=int(x.mixed_bin.sum()),correct=int((x.op==x.guess).sum()),
                        mode_predictions=int(x.guess_mode.sum()),errors_to_mode=int(((x.op!=x.guess)&x.guess_mode).sum()),
                        known_condition=int((x.maximum>0).sum()),exact_seen=int(x.input_total.notna().sum()),
                        exact_multi=int((x.input_ops>1).sum()),exact_single_true=int(((x.input_ops==1)&x.input_true_n.notna()).sum()),
                        exact_single_true_wrong=int(((x.input_ops==1)&x.input_true_n.notna()&(x.op!=x.guess)).sum()),
                        native_multiple_inputs=int((x.exact_pairs>1).sum()))
rank=train.groupby('op').agg(global_events=('n','sum'))
rank['local_nonmode_events']=train[train.n<train.maximum].groupby('op').n.sum()
rank.fillna(0).sort_values('global_events',ascending=False).to_csv(out/'global_operations.csv')
totals=g[['mode_nodes','nonmode_nodes','mode_fp','nonmode_fp']].sum().astype(int).to_dict()
weights=np.minimum(g.mode_nodes,g.nonmode_nodes)
standardized={k:float(np.average(g[k+'_fp']/g[k+'_nodes'],weights=weights)) if len(g) else None for k in ['mode','nonmode']}
summary=dict(peaks=details,matched=dict(strata=len(g),**totals,standardized_FPR=standardized,case_pairs=len(cases)),
             max_node_score_error=max_error,train=json.loads((source/'training.json').read_text()),replay=json.loads((source/'replay.json').read_text()))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
