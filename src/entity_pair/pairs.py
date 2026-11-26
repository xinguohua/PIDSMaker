"""Stage 1. Bind canonical/all tied peaks and count pair-level event and node populations."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from composition_compatibility import predictions

def main(source,cell,out):
    labels,proof=predictions(cell)
    proto=json.loads((source/'protocol.json').read_text())
    assert proto['prediction_sha256']==proof['sha256'] and proto['seed']==0
    peaks=pd.read_csv(source/'all_tied_peaks.csv.gz',keep_default_na=False,float_precision='round_trip')
    main=peaks[peaks.canonical].copy()
    assert set(main.node)=={n for n,g in labels.items() if g==0} and len(main)==286858
    assert main.prediction.sum()==172 and (main.prediction==(main.score>proof['threshold'])).all()
    train=pd.read_csv(source/'condition_operations.csv',keep_default_na=False)
    pair=['src','dst']
    tr=train.pivot(index=pair,columns='op',values='n').fillna(0).astype(int)
    assert tr.values.sum()==3426589
    main['N']=tr.sum(axis=1).reindex(pd.MultiIndex.from_frame(main[pair]),fill_value=0).values
    main['N_bin']=pd.cut(main.N,[-1,0,9,99,999,np.inf],labels=['0','1–9','10–99','100–999','≥1000'])
    main['category']=np.select([main.N==0,main.n==0,main.n<main.maximum,main.operations==1],['unseen_pair','unseen_operation','nonmode','sole_mode'],default='multiple_ops_mode')
    def rates(frame,col):
        df=frame.groupby(col,observed=True).prediction.agg(['size','sum']).rename(columns={'size':'nodes','sum':'FP'})
        df['TN']=df.nodes-df.FP
        df['FPR']=df.FP/df.nodes
        return df
    rates(main,'N_bin').to_csv(out/'frequency.csv')
    rates(main,'category').to_csv(out/'category.csv')
    stats=rates(main,pair)
    stats['N']=tr.sum(axis=1).reindex(stats.index,fill_value=0)
    stats.reset_index().to_pickle(out/'pair_stats.pkl')
    fp=peaks[peaks.prediction==1].copy()
    fp.to_csv(out/'all_fp_ties.csv',index=False)
    target=set(zip(fp.src,fp.dst))
    def distribution(frame,cols):
        return {tuple(k):dict(zip(g[cols[-1]],g['n'].astype(int))) for k,g in frame.groupby(cols[:-1])}
    train_dist=distribution(train,pair+['op'])
    event=Counter()
    total=0
    for df in pd.read_csv(source/'test_events.csv.gz',chunksize=150000,keep_default_na=False):
        event.update(df.groupby(pair+['op','guess']).size().to_dict());total+=len(df)
    assert total==2065670
    hist=pd.Series(event).rename('n').rename_axis(pair+['op','guess']).reset_index()
    test_dist=distribution(hist.groupby(pair+['op'],as_index=False).n.sum(),pair+['op'])
    pred_dist=distribution(hist.groupby(pair+['guess'],as_index=False).n.sum(),pair+['guess'])
    rows=[]
    for s,d in sorted(target):
        k=s,d
        h=train_dist.get(k,{})
        record=stats.loc[k] if k in stats.index else None
        rows.append(dict(src=s,dst=d,N=sum(h.values()),train_counts=json.dumps(h),P_train=json.dumps({o:n/sum(h.values()) for o,n in h.items()}),test_truth=json.dumps(test_dist[k]),test_prediction=json.dumps(pred_dist[k]),FP=int(record.FP) if record is not None else 0,TN=int(record.TN) if record is not None else 0,FPR=record.FPR if record is not None else np.nan))
    outpairs=pd.DataFrame(rows).sort_values('FP',ascending=False)
    assert outpairs.FP.sum()==172
    outpairs.to_csv(out/'ENTITY_PAIR_FP_TN.csv',index=False)
    hist[hist.set_index(pair).index.isin(target)].to_csv(out/'ENTITY_PAIR_OPERATION_DISTRIBUTION.csv',index=False)
    # Preselect distinct comparison pairs without using predictions of operation or neighborhood distributions.
    examples=main[main.prediction==1]
    pool=main[main.prediction==0].copy()
    samples=[]
    for row in examples.itertuples():
        df=pool[(pool.op==row.op)&((pool.src!=row.src)|(pool.dst!=row.dst))].copy()
        if df.empty:continue
        df['kind_diff']=(df.src.str.split().str[0]!=row.src.split()[0]).astype(int)+(df.dst.str.split().str[0]!=row.dst.split()[0]).astype(int)
        df['delta']=(np.log1p(df.N)-np.log1p(row.N)).abs()+(np.log1p(df.degree)-np.log1p(row.degree)).abs()
        chosen=df.sort_values(['kind_diff','delta','node']).drop_duplicates(pair).head(3)
        samples.extend((row.node,int(n)) for n in chosen.node)
    matches=pd.DataFrame(samples,columns=['fp_node','tn_node'])
    matches.to_csv(out/'matched.csv',index=False)
    queries=pd.concat([fp,main[main.node.isin(matches.tn_node)]],ignore_index=True)
    queries.to_pickle(out/'queries.pkl')
    assert len(fp)==204
    print(outpairs.head(8).to_string(index=False),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(predictions=proof,canonical_fp_pairs=len(examples.groupby(pair)),all_fp_pairs=len(target),ties=len(fp),matched=len(matches),unique_tn=matches.tn_node.nunique(),test_events=total,train_events=int(tr.values.sum())),indent=2))
if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(os.environ['RUN_OUTPUT_DIR']))
