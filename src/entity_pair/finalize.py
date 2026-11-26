"""Combine committed stages without reloading native inputs or executing the model."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
def main(source,first,second,third,out):
    peaks=pd.read_csv(source/'node_peaks.csv.gz',low_memory=False)
    qs=pd.read_pickle(first/'queries.pkl')
    pair=['src','dst']; vk=pair+['vec_src','vec_dst']
    train=pd.read_csv(source/'condition_operations.csv',keep_default_na=False)
    lens=train.groupby(pair).size()
    index=pd.MultiIndex.from_frame(peaks[pair])
    operations=lens.reindex(index,fill_value=0).to_numpy()
    peaks['category']=np.select([operations==0,peaks.n==0,peaks.n<peaks.maximum,operations==1],['unseen_pair','unseen_operation','nonmode','sole_mode'],default='multiple_ops_mode')
    def rates(df,keys):
        g=df.groupby(keys).prediction.agg(['size','sum']).rename(columns={'size':'nodes','sum':'FP'})
        g['TN']=g.nodes-g.FP;g['FPR']=g.FP/g.nodes
        return g
    rates(peaks,'category').to_csv(out/'category.csv')
    nn=pd.read_csv(second/'SIMILAR_CONTEXT_ANALYSIS.csv',keep_default_na=False)
    fps=qs[qs.prediction==1]
    config=fps[vk+['op','guess']].drop_duplicates()
    tested=nn.merge(config,on=vk)
    tested['true_operation_train_n']=[json.loads(h).get(o,0) for h,o in zip(tested.train_operations,tested.op)]
    tested['predicted_operation_train_n']=[json.loads(h).get(o,0) for h,o in zip(tested.train_operations,tested.guess)]
    tested.to_csv(out/'fp_neighbor_directions.csv',index=False)
    relevant=nn.groupby(vk+['method'])
    canonical=qs[qs.canonical].set_index('node')
    contrast=pd.read_csv(third/'matched_contrasts.csv')
    # Compare exactly the FP's alternative on both sides. Report outcome-independent spectral matches too.
    details=[]
    for r in contrast.itertuples():
        fp=canonical.loc[r.fp_node];tn=canonical.loc[r.tn_node]
        other=[]
        for rec in (fp,tn):
            key=tuple(rec[k] for k in vk)+(r.method,)
            best=high_n=0
            if key in relevant.groups:
                for e in relevant.get_group(key).itertuples():
                    h=json.loads(e.train_operations)
                    if e.train_count>=1000:
                        high_n+=1
                        if h.get(r.competitor,0)==max(h.values()) and h.get(r.competitor,0)>h.get(r.real,0):
                            best=max(best,e.cosine)
            other.extend([best,high_n])
        details.append(other)
    contrast[['fp_best_signal_cos','fp_high_count','tn_best_signal_cos','tn_high_count']]=details
    contrast.to_csv(out/'matched_evidence.csv',index=False)
    ev=pd.read_csv(third/'ALL_FP_EVIDENCE.csv')
    stats=pd.read_csv(first/'ENTITY_PAIR_FP_TN.csv')
    ev=ev.merge(qs[qs.canonical & (qs.prediction==1)][['node','uuid','event_uuid','score','loss','vec_src','vec_dst','degree','p_true','p_guess']],on='node',validate='1:1').merge(stats[pair+['train_counts','P_train']],on=pair,validate='m:1')
    ev['strong_train_failure_same_input']=ev.same_input_unambiguous_wrong & (ev.train_accuracy==0)
    ev['perfect_train_wrong_new_input']=(ev.train_accuracy==1)&ev.seen_pair_new_input
    ev['local_mode_competing_hf_neighbor']=ev.pair_mode & ev.feature_hf_mode
    ev.to_csv(out/'ALL_FP_EVIDENCE.csv',index=False)
    true_preds=rates(peaks[peaks.set_index(pair).index.isin(stats.set_index(pair).index)],pair+['op','guess']).reset_index()
    true_preds.to_csv(out/'peak_true_predicted.csv',index=False)
    summary=[]
    for method,g in contrast.groupby('method'):
        g=g[(g.fp_neighbors>0)&(g.tn_neighbors>0)].copy()
        g['fp_near']=g.fp_best_signal_cos>=.99;g['tn_near']=g.tn_best_signal_cos>=.99
        for subset,ids in [('all',ev.node),('true_is_local_mode',ev.loc[ev.pair_mode,'node']),('sole_true',ev.loc[ev.sole_true,'node'])]:
            sub=g[g.fp_node.isin(ids)]
            if sub.empty:continue
            m=sub.groupby('fp_node')[['fp_high','tn_high','fp_near','tn_near']].mean().join(ev.set_index('node')[pair])
            p=m.groupby(pair)[['fp_high','tn_high']].mean()
            summary.append(dict(method=method,subset=subset,fp_nodes=len(m),fp_pairs=len(p),unique_tn=sub.tn_node.nunique(),fp=m.fp_high.mean(),tn=m.tn_high.mean(),fp_by_pair=p.fp_high.mean(),tn_by_pair=p.tn_high.mean(),fp_cos99=m.fp_near.mean(),tn_cos99=m.tn_near.mean()))
    pd.DataFrame(summary).to_csv(out/'matched_summary.csv',index=False)
    checks=dict(sole_wrong_TN=int(((operations==1)&(peaks.n>0)&(peaks.guess!=peaks.op)&(peaks.prediction==0)).sum()),fp=int(ev.FP.sum()),canonical_pairs=len(stats),strong_train_failure=int(ev.strong_train_failure_same_input.sum()),perfect_train_wrong_new_input=int(ev.perfect_train_wrong_new_input.sum()),local_mode_hf=int(ev.local_mode_competing_hf_neighbor.sum()))
    (out/'summary.json').write_text(json.dumps(checks,indent=2))
if __name__=='__main__':main(*map(Path,sys.argv[1:]),Path(os.environ['RUN_OUTPUT_DIR']))
