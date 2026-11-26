"""Cross-check two real test windows with the original interval membership algorithm."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from action_index import ActionIndex

def check(source,results,out):
    chosen={581,919}
    acts=pd.read_pickle(source/'actions.pkl').set_index('action')
    expected=pd.read_csv(source/'event_groups.tsv',sep=' ',names=['w','row','m2','m3','key2','key3'])
    windows={}
    with (source/'events.tsv').open() as f:
        for line in f:
            w,test,n=map(int,line.split());groups=[]
            for _ in range(n):
                raw=next(f)
                if w not in chosen:continue
                row,u,v,g,a,complete=map(int,raw.split())
                while len(groups)<=g:groups.append([])
                groups[g].append((row,u,v,a))
            if w in chosen:windows[w]=ActionIndex(groups)
    rows=[]
    lookup={}
    wanted={}
    for w,index in windows.items():
        for k in (2,3):
            seqs={}
            for seq,count in index.sequences(k):
                key=0
                for a in seq:key=key<<21|a
                seqs[key]=(seq,count)
            wanted[w,k]=seqs
            lookup.update(dict.fromkeys(seqs))
    keys=np.array(sorted(lookup),dtype='uint64')
    types=np.memmap(source/'types.bin',np.uint64,'r').reshape(-1,3)
    for i in range(0,len(types),500000):
        block=types[i:i+500000]
        for key,n,z in block[np.isin(block[:,0],keys)]:
            lookup[int(key)]=int(n),int(z)
    for (w,k),seqs in wanted.items():
        index=windows[w]
        flags=Counter()
        for key,(seq,count) in seqs.items():
            n,z=lookup[key]
            assert count<=z
            bit=8 if not all(acts.at[a,'complete'] for a in seq) else 4 if not all(acts.at[a,'train'] for a in seq) else 2 if n==0 else 1|(16 if n<=10 else 32 if n<=100 else 64)
            for variant,low,high in index.bounds(seq):
                for records in index.events[variant][low:high]:
                    for row in records:flags[row]|=bit
        ref=expected[expected.w==w]
        assert all(flags[r]==v for r,v in zip(ref.row,ref[f'm{k}']))
        rows.append(dict(window=w,length=k,event_masks_checked=len(ref),window_types=len(seqs),instances=sum(z for _,z in seqs.values())))
    evidence=[]
    for k in (2,3):
        df=pd.read_csv(results/f'control_nodes_L{k}.csv.gz',usecols=['FP','bin','action_counts'],dtype={'bin':str})
        strong=df.action_counts.map(lambda s:all(int(n)>100 for n in s.split(',')))
        for b in ('0','1–10','11–100','>100'):
            subset=df[strong & (df.bin==b)]
            evidence.append(dict(length=k,train_bin=b,nodes=len(subset),FP=int(subset.FP.sum()),TN=int((~subset.FP).sum())))
    (out/'summary.json').write_text(json.dumps(dict(window_checks=rows,all_actions_over_100=evidence),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','results'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    check(a.source,a.results,Path(os.environ['RUN_OUTPUT_DIR']))
