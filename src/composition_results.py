"""Summarize overlapping participation, disjoint peak contexts, and common-support comparisons."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from action_composition import ro
from composition_compatibility import THRESHOLD

LABELS={1:'Seen Composition',2:'Novel Composition',4:'Novel Action Composition',8:'Incomplete'}
FREQ={2:'0',16:'1–10',32:'11–100',64:'>100'}
MASK=(1<<21)-1

def primary(mask):
    m=mask&15
    if m==0:return 'No path'
    if m in LABELS:return LABELS[m]
    return 'Multiple'  # Includes mixed incomplete and complete evidence, detailed mask retained.

def result_rows(ind,fp,length,scope,group):
    yes=int(fp[ind].sum())
    n=int(ind.sum())
    return dict(length=length,scope=scope,group=group,nodes=n,FP=yes,TN=n-yes,FPR=yes/n if n else None)

def band(n):
    return 0 if n==0 else 1 if n<=10 else 2 if n<=100 else 3 if n<=1000 else 4

def adjusted(controls,k):
    from scipy.stats import norm
    rows=[]
    for alternative in ('0','1–10','11–100'):
        sample=controls[controls.bin.isin((alternative,'>100'))]
        groups=sample.groupby(['kind','op','action_bands','rel_bin','qc_bin','context_band','degree_band'])
        a_sum=b_sum=score=variance=0.
        overlap=Counter()
        for _,df in groups:
            flags=df.bin==alternative
            e=int(flags.sum()); h=len(df)-e
            if not e or not h:continue
            a=int(df.loc[flags,'FP'].sum());c=int(df.loc[~flags,'FP'].sum())
            n=e+h
            a_sum+=a*(h-c)/n
            b_sum+=(e-a)*c/n
            score+=a-e*(a+c)/n
            variance+=e*h*(a+c)*(n-a-c)/(n*n*(n-1))
            overlap.update(dict(strata=1,exposed=e,high=h,exposed_FP=a,high_FP=c))
        rows.append(dict(length=k,comparison=alternative,**dict(overlap),
                         MH_OR=a_sum/b_sum if b_sum else None,
                         CMH_p=2*norm.sf(abs(score)/np.sqrt(variance)) if variance else None,
                         eligible=len(sample)))
    return rows

def summarize(root,src,out):
    with (src/'peak_input.pkl').open('rb') as f: nodes,ties,conditions=pickle.load(f)
    peaks=pd.DataFrame(ties,columns=['node','w','row','u','v','ns','seq','uuid','action','loss','condition'])
    del ties
    acts=pd.read_pickle(src/'actions.pkl').set_index('action')
    action_n=acts.train.to_dict()
    ops=acts.op.to_dict()
    peaks['context_n']=[sum(conditions.get(c,{}).values()) for c in peaks.condition]
    support_cache={c:(max(hist.values()),hist) for c,hist in conditions.items()}
    def relative(c,a):
        maximum,hist=support_cache.get(c,(0,{}))
        return hist.get(ops[a],0)/maximum if maximum else -1
    peaks['relative']=[relative(c,a) for c,a in zip(peaks.condition,peaks.action)]
    op_max=Counter()
    for hist in conditions.values():
        for op,n in hist.items():op_max[op]=max(op_max[op],n)
    peaks['op']=peaks.action.map(ops)
    peaks['context_op_n']=[conditions.get(c,{}).get(op,0) for c,op in zip(peaks.condition,peaks.op)]
    peaks['Qc']=peaks.context_op_n/peaks.op.map(op_max)
    normal=pd.DataFrame.from_dict(nodes,orient='index',columns=['kind','degree','score']).sort_index()
    normal.index.name='node'
    normal['FP']=normal.score>THRESHOLD
    fp=normal.FP.to_numpy()
    ids=normal.index.to_numpy()
    size=int(max(ids))+1
    peak_ids=peaks.node.to_numpy()
    groups=pd.read_csv(src/'event_groups.tsv',sep=' ',names=['w','row','mask2','mask3','key2','key3']).set_index(['w','row'])
    assert groups.index.is_unique
    pq=groups.reindex(pd.MultiIndex.from_frame(peaks[['w','row']]),fill_value=0)
    for c in pq: peaks[c]=pq[c].to_numpy()
    native=ro(root/'velox_action_composition/native.sqlite')
    edges=pd.read_sql_query("select e.w,e.row,e.u,e.v from edges e join windows w using(w) where split='test'",native).set_index(['w','row']).loc[groups.index]
    assert len(edges)==1906174
    types=np.memmap(src/'types.bin',np.uint64,'r').reshape((-1,3))
    # Only look up exact frequencies for the outcome-blind single-path control witnesses.
    wanted=np.unique(peaks[['key2','key3']].to_numpy().ravel())
    keys={}
    totals=Counter()
    complete=dict(acts.complete)
    train=acts.train.to_dict()
    complete_arr=np.zeros(max(train)+1,bool); support_arr=np.zeros(max(train)+1,np.int64)
    complete_arr[acts.index]=acts.complete; support_arr[acts.index]=acts.train
    for start in range(0,len(types),500000):
        key,tr,te=types[start:start+500000].T
        for key0,n in zip(key[np.isin(key,wanted)],tr[np.isin(key,wanted)]):keys[int(key0)]=int(n)
        for k,pick in ((2,key<(1<<42)),(3,key>=(1<<42))):
            val=key[pick];t=tr[pick];z=te[pick]
            parts=[(val>>(21*j))&MASK for j in range(k)]
            ok=np.logical_and.reduce([complete_arr[x] for x in parts])
            known=np.logical_and.reduce([support_arr[x]>0 for x in parts])
            for label,mask in [('Seen Composition',ok&(t>0)),('Novel Composition',ok&known&(t==0)),('Novel Action Composition',ok&~known),('Incomplete',~ok)]:
                for tag,v in [('train',t),('test',z)]:
                    totals[k,label,tag,'types']+=int(np.count_nonzero(v[mask]))
                    totals[k,label,tag,'instances']+=int(v[mask].sum())
    type_table=pd.DataFrame([(*key,n) for key,n in totals.items()],columns=['length','group','split','unit','n'])
    type_table.to_csv(out/'composition_counts.csv',index=False)
    for k,expected in [(2,(29434179,14119305)),(3,(64722817,31680780))]:
        for split,n in zip(('train','test'),expected):
            assert type_table.query("length==@k and split==@split and unit=='instances'").n.sum()==n
    rows=[]; controls=[]
    for k in (2,3):
        name=f'mask{k}'
        all_bits=np.zeros(size,np.uint8);peak_bits=np.zeros(size,np.uint8)
        for column in ('u','v'):np.bitwise_or.at(all_bits,edges[column].to_numpy(),groups[name].to_numpy(dtype='uint8'))
        np.bitwise_or.at(peak_bits,peak_ids,peaks[name].to_numpy(dtype='uint8'))
        for scope,b in [('participation',all_bits[ids]),('peak',peak_bits[ids])]:
            normal[f'{scope}_{k}']=b
            for bit,l in LABELS.items():rows.append(result_rows((b&bit)>0,fp,k,scope,l))
            for bit,l in FREQ.items():rows.append(result_rows((b&bit)>0,fp,k,scope+'_frequency',l))
            for m in range(16):rows.append(result_rows((b&15)==m,fp,k,scope+'_mask',str(m)))
            for l in ['No path','Multiple',*LABELS.values()]:
                values=np.array([primary(i)==l for i in range(128)])
                rows.append(result_rows(values[b],fp,k,scope+'_exclusive',l))
        canon=peaks.sort_values(['node','w','row']).drop_duplicates('node').copy()
        witnesses=canon[f'key{k}'].map(int)
        selected=[]
        for key in witnesses.unique():
            if not key: continue
            seq=tuple((int(key)>>(21*j))&MASK for j in reversed(range(k)))
            if all(complete[a] and train[a]>0 for a in seq):
                n=keys[key]
                selected.append((key,'0' if not n else '1–10' if n<=10 else '11–100' if n<=100 else '>100',
                                 ','.join(str(band(train[a])) for a in seq),','.join(str(train[a]) for a in seq)))
        ref=pd.DataFrame(selected,columns=[f'key{k}','bin','action_bands','action_counts'])
        control=canon.merge(ref,on=f'key{k}').set_index('node').join(normal[['degree','kind','FP']])
        control['rel_bin']=pd.cut(control.relative,[-2,-.5,0,.01,.1,.5,.9999999999,1],include_lowest=True).astype(str)
        control['qc_bin']=pd.cut(control.Qc,[-1,0,.01,.1,.5,.9999999999,1],include_lowest=True).astype(str)
        control['context_band']=control.context_n.map(band);control['degree_band']=control.degree.map(band)
        control.to_csv(out/f'control_nodes_L{k}.csv.gz')
        controls.extend(adjusted(control,k))
    pd.DataFrame(controls).to_csv(out/'adjusted.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'node_results.csv',index=False)
    normal.to_csv(out/'normal_nodes.csv.gz')
    peaks.drop(columns='condition').to_csv(out/'all_tied_peak_events.csv.gz',index=False)
    peak_fp=peaks[peaks.node.map(normal.FP)].drop(columns='condition')
    peak_fp.to_csv(out/'FP_peak_events.csv',index=False)
    union=(normal.peak_2.astype(int)|normal.peak_3.astype(int))&2
    summary=dict(normal=len(normal),FP=int(fp.sum()),tied_nodes=int((peaks.groupby('node').size()>1).sum()),
                 all_peaks=len(peaks),FP_peaks=len(peak_fp),novel_peak_FP_either=int(normal.FP[union>0].sum()),
                 actions=[dict(split=s,complete=bool(c),types=int((df[s]>0).sum()),instances=int(df[s].sum()))
                          for c,df in acts.groupby('complete') for s in ('train','test')])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('root','source'):p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args()
    summarize(a.root,a.source,Path(os.environ['RUN_OUTPUT_DIR']))
