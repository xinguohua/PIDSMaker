"""Check sparse adjusted strata by conditional simulation and vary normal-event definition."""
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from context_imbalance_summary import KEY,BINS,T,bucket

def run(source,out):
    ct=pd.read_csv(source/'context_performance.csv.gz',keep_default_na=False)
    pred=pd.read_pickle(source/'pred.pkl')
    normal=pred.index[pred['gt']==0]
    blocks=[]
    allrates=[]
    for f in sorted((source/'events').glob('*.gz')):
        e=pd.read_pickle(f).merge(ct[KEY+['cid','N']],on=KEY,how='left',sort=False,validate='m:1')
        assert e.cid.notna().all()
        e['bin']=bucket(e.N)
        e['high']=e.loss>T
        allrates.append(e[e.y==0].groupby(['op','bin'],observed=True).high.agg(['sum','size']))
        for side,mask in [('u',e.u.isin(normal)),('v',e.v.isin(normal)&e.u.ne(e.v))]:
            # Preserve the original eid ordering for ties in minimum support.
            a=e.loc[mask,[side,'op','N','cid']].rename(columns={side:'node'})
            a['seq']=int(f.stem.split('.')[0])*10**8+a.index
            a['side']=side
            a['degree']=a.groupby(['node','op']).node.transform('size')
            blocks.append(a.sort_values(['N','seq']).drop_duplicates(['node','op']))
    ar=pd.concat(allrates).groupby(level=[0,1],observed=True).sum()
    ar['rate']=ar['sum']/ar['size']
    ar.to_csv(out/'all_y0_event_rates.csv')
    x=pd.concat(blocks,ignore_index=True)
    degree=x.groupby('node').degree.sum()
    x=x.sort_values(['node','op','N','seq']).drop_duplicates(['node','op']).join(ct.set_index('cid')[['src_type','dst_type','competition','missing']],on='cid')
    x['activity_bin']=np.floor(np.log2(x.node.map(degree))).astype(int)
    x['node_type']=np.where(x.side=='u',x.src_type,x.dst_type)
    x['fp']=x.node.map(pred.prediction)
    ref=pd.read_csv(source/'adjusted.csv').set_index('op')
    rng=np.random.default_rng(20261126)
    rows=[]
    for op,g in x.groupby('op'):
        g=g[(g.N<10)|(g.N>=1000)].copy()
        g['low']=g.N<10
        null=np.zeros(100000)
        expected=observed=eligible=0
        low_n=high_n=0
        for _,s in g.groupby(['node_type','src_type','dst_type','activity_bin','missing','competition']):
            n=len(s)
            l=int(s.low.sum())
            h=n-l
            if min(l,h)==0:continue
            low_n+=l
            high_n+=h
            f=int(s.fp.sum())
            observed+=int(s.loc[s.low,'fp'].sum())
            expected+=l*f/n
            if 0<f<n:
                null+=rng.hypergeometric(l,h,f,size=len(null))
                eligible+=1
            elif f==n:null+=l
        r=ref.loc[op]
        assert (observed,low_n,high_n)==(r.low_fp,r.low_nodes,r.high_nodes)
        p=(1+(np.abs(null-expected)>=abs(observed-expected)-1e-10).sum())/(len(null)+1) if eligible else np.nan
        rows.append(dict(op=op,informative_strata=eligible,observed=observed,expected=expected,mc_p=p))
    outp=pd.DataFrame(rows)
    s=outp.mc_p.dropna().sort_values()
    outp.loc[s.index,'holm']=(s*np.arange(len(rows),len(rows)-len(s),-1)).cummax().clip(upper=1)
    outp.to_csv(out/'adjusted_sparse_tests.csv',index=False)
    g=pd.read_csv(source/'peak_rates.csv')
    fig,axes=plt.subplots(4,3,figsize=(13,12),constrained_layout=True)
    for (op,b),ax in zip(g.groupby('op'),axes.flat):
        ax.plot(BINS,b.FPR*100,'o-',ms=4)
        ax.set(title=op.replace('EVENT_',''),ylabel='Node FPR (%)')
        ax.tick_params(axis='x',labelrotation=25)
    for ax in axes.flat[g.op.nunique():]:ax.set_visible(False)
    fig.savefig(out/'peak_rates.png',dpi=140)
    summary={'y0_edges':int(ar['size'].sum()),'y0_high':int(ar['sum'].sum()),
             'normal_high':int(pd.read_csv(source/'event_rates.csv').high.sum())}
    (out/'sensitivity.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':
    run(Path(sys.argv[1]),Path(os.environ['RUN_OUTPUT_DIR']))
