"""Fixed-operation comparisons for the verified first-path context extraction."""
import argparse
import json
import os
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

KEY = ['op','src_label','dst_label']
BINS = ['0','1-9','10-99','100-999','1000+']
T = 5.9490790367126465


def bucket(n):
    return pd.cut(n,[-1,0,9,99,999,np.inf],labels=BINS)


def rates(data, keys):
    v = data.groupby(keys,observed=True).fp.agg(['sum','size']).rename(columns={'sum':'FP','size':'nodes'})
    v['TN'] = v.nodes-v.FP
    v['FPR'] = v.FP/v.nodes
    return v


def analyze(source,out,raw):
    train = pd.read_csv(source/'training.csv.gz',keep_default_na=False)
    predictions = pd.read_pickle(source/'pred.pkl')
    a = json.loads((source/'audit.json').read_text())
    print('source',a['splits'],flush=True)
    records = []
    for f in sorted((source/'events').glob('*.pkl.gz')):
        e = pd.read_pickle(f)
        e['w'] = int(f.name.split('.')[0])
        e['row'] = np.arange(len(e))
        records.append(e)
    e = pd.concat(records,ignore_index=True)
    del records
    assert len(e) == a['splits']['test']['edges']
    e['eid'] = e.index
    contexts = e[KEY+['src_type','dst_type']].drop_duplicates().merge(train,on=KEY,how='outer',validate='1:1')
    contexts['N'] = contexts.N.fillna(0).astype(int)
    contexts[['P','Qc']] = contexts[['P','Qc']].fillna(0)
    contexts['cid'] = contexts.index
    contexts['bin'] = bucket(contexts.N)
    # Conditional competition is measured within an entity pair, across operations.
    totals = train.groupby(KEY[1:]).N.sum()
    contexts = contexts.join(totals.rename('pair_total'),on=KEY[1:])
    contexts['share'] = contexts.N/contexts.pair_total
    contexts['competition'] = pd.cut(contexts.share,[-1,0,.1,.5,.9,1],labels=['absent','weak','minor','major','dominant']).astype(object).fillna('unseen_pair')
    fields = ['src_label','dst_label']
    for k in fields:
        contexts[k+'_missing'] = contexts[k].str.contains(r'(?:^|\s)(?:None|null|nan|\[NO_[A-Z]+\])(?:\s|$)',case=False,regex=True) | contexts[k].str.strip().isin(['','file','subject','netflow'])
    contexts['missing'] = contexts.src_label_missing.astype(int)+contexts.dst_label_missing.astype(int)
    join = e[KEY].merge(contexts[KEY+['cid']],on=KEY,how='left',sort=False,validate='m:1')
    e['cid'] = join.cid.to_numpy()
    del join
    e.drop(columns=KEY+['src_type','dst_type'],inplace=True)
    trstats = []
    fig,axes = plt.subplots(4,3,figsize=(13,13),constrained_layout=True)
    for (op,group),ax in zip(train.groupby('op'),axes.flat):
        g = group.sort_values('N',ascending=False)
        trstats.append(dict(op=op,events=g.N.sum(),contexts=len(g),n_min=g.N.min(),n_median=g.N.median(),n_max=g.N.max(),
                            q10=g.Qc.quantile(.1),q50=g.Qc.median(),q90=g.Qc.quantile(.9),
                            top1=g.P.iloc[:1].sum(),top5=g.P.iloc[:5].sum(),top10=g.P.iloc[:10].sum(),
                            contexts_1_9=(g.N<10).sum(),low_events=g[g.N<10].N.sum()))
        ax.loglog(np.arange(1,len(g)+1),g.N,'.',ms=2)
        ax.set(title=op.replace('EVENT_',''),xlabel='Context rank',ylabel='Training events')
    for ax in axes.flat[len(trstats):]: ax.set_visible(False)
    fig.savefig(out/'training_rank.png',dpi=140)
    plt.close(fig)
    pd.DataFrame(trstats).to_csv(out/'operation_distribution.csv',index=False)
    normal = predictions.index[predictions['gt']==0]
    good = (e.y==0)&e.u.isin(normal)&e.v.isin(normal)
    bench = e.loc[good,['cid','loss']].join(contexts[['op','bin']],on='cid')
    bench['high'] = bench.loss>T
    ev = bench.groupby(['op','bin'],observed=True).agg(events=('high','size'),high=('high','sum'),mean_loss=('loss','mean'))
    for q in [.5,.9,.99]:
        ev['loss_p'+str(int(q*100))] = bench.groupby(['op','bin'],observed=True).loss.quantile(q)
    grid = pd.MultiIndex.from_product([sorted(contexts.op.unique()),BINS],names=['op','bin'])
    ev = ev.reindex(grid)
    ev[['events','high']] = ev[['events','high']].fillna(0).astype(int)
    ev['rate'] = ev.high/ev.events
    ev.to_csv(out/'event_rates.csv')
    fig,axes = plt.subplots(4,3,figsize=(14,12),constrained_layout=True)
    for (op,g),ax in zip(ev.groupby(level=0),axes.flat):
        ax.plot(BINS,100*g.rate,'o-',ms=4)
        ax.set(title=op.replace('EVENT_',''),ylabel='High loss (%)')
        ax.tick_params(axis='x',labelrotation=25)
    for ax in axes.flat[len(trstats):]: ax.set_visible(False)
    fig.savefig(out/'event_rates.png',dpi=140)
    plt.close(fig)
    # One row per incident edge per endpoint. Self-loops count once.
    ends = []
    for side,mask in [('u',e.u.isin(normal)),('v',e.v.isin(normal)&e.v.ne(e.u))]:
        z = e.loc[mask,[side,'eid','cid','loss']].rename(columns={side:'node'})
        z['side'] = side
        ends.append(z)
    inc = pd.concat(ends,ignore_index=True)
    degree = inc.groupby('node').size()
    maxima = inc.groupby('node').loss.transform('max')
    ties = inc[inc.loss.eq(maxima)].sort_values(['node','eid'])
    ties = ties.join(contexts,on='cid',rsuffix='_context').join(e[['w','row','ns','uuid','u','v','y']],on='eid')
    ties['fp'] = ties.node.map(predictions.prediction)
    ties['score'] = ties.node.map(predictions.score)
    ties['activity'] = ties.node.map(degree)
    ties['ties'] = ties.groupby('node').eid.transform('size')
    ties['node_type'] = np.where(ties.side=='u',ties.src_type,ties.dst_type)
    assert np.allclose(ties.loss,ties.score,rtol=0,atol=1e-7) and (ties.fp==(ties.loss>T)).all()
    peaks = ties.drop_duplicates('node').copy()
    assert len(peaks)==286858 and peaks.fp.sum()==172
    ties.to_csv(out/'all_peak_ties.csv.gz',index=False)
    peaks.to_csv(out/'normal_node_peaks.csv.gz',index=False)
    perf = contexts.join(rates(peaks,['cid']),on='cid')
    perf[['FP','TN','nodes']] = perf[['FP','TN','nodes']].fillna(0).astype(int)
    perf.to_csv(out/'context_performance.csv.gz',index=False)
    per = rates(peaks,['op','bin']).reindex(grid)
    per[['FP','TN','nodes']] = per[['FP','TN','nodes']].fillna(0).astype(int)
    per.to_csv(out/'peak_rates.csv')
    rates(peaks,['bin']).to_csv(out/'pooled_peak_rates.csv')
    # Define exposure independently of loss: least-supported incident context per node and operation.
    x = inc[['node','cid','side','eid']].join(contexts[['op','N','competition','missing','src_type','dst_type']],on='cid')
    rep = x.sort_values(['node','op','N','eid']).drop_duplicates(['node','op']).copy()
    rep['activity'] = rep.node.map(degree)
    rep['node_type'] = np.where(rep.side=='u',rep.src_type,rep.dst_type)
    rep['activity_bin'] = np.floor(np.log2(rep.activity)).astype(int)
    rep['bin'] = bucket(rep.N)
    rep['fp'] = rep.node.map(predictions.prediction)
    rates(rep,['op','bin']).to_csv(out/'participation_rates.csv')
    controls = ['node_type','src_type','dst_type','activity_bin','missing','competition']
    effects = []
    for op,g in rep.groupby('op'):
        z = g[(g.N<10)|(g.N>=1000)].copy()
        z['low'] = z.N<10
        table = z.groupby(controls+['low'],observed=True).fp.agg(['sum','size']).unstack(fill_value=0)
        for k in ['size','sum']:
            for side in [False,True]:
                if (k,side) not in table: table[k,side]=0
        t = table[(table['size',True]>0)&(table['size',False]>0)]
        lo,hi = t['size',True],t['size',False]
        aa,cc = t['sum',True],t['sum',False]
        n = lo+hi
        num,den = (aa*(hi-cc)/n).sum(),((lo-aa)*cc/n).sum()
        fp = aa+cc
        delta = (aa-lo*fp/n).sum()
        var = (lo*hi*fp*(n-fp)/(n*n*(n-1))).sum()
        effects.append(dict(op=op,low_total=int(z.low.sum()),high_total=int((~z.low).sum()),overlap_strata=len(t),
                            low_nodes=lo.sum(),high_nodes=hi.sum(),low_fp=aa.sum(),high_fp=cc.sum(),
                            MH_OR=num/den if den else (np.inf if num else np.nan),
                            CMH_p=2*norm.sf(abs(delta)/np.sqrt(var)) if var>0 else np.nan))
    pd.DataFrame(effects).to_csv(out/'adjusted.csv',index=False)
    # Show low-support FP versus a higher-support TN; separately inspect well-supported FP.
    db = sqlite3.connect(f'file:{raw}?mode=ro',uri=True)
    examples = []
    for op,g in peaks.groupby('op'):
        for name,block in [('low',g[(g.fp==1)&(g.N<10)]),('supported',g[(g.fp==1)&(g.N>=1000)]),('intermediate',g[(g.fp==1)&g.N.between(10,999)])]:
            if block.empty: continue
            a = block.sort_values(['N','node']).iloc[0]
            pool = g[(g.fp==0)&(g.cid!=a.cid)].copy()
            if pool.empty: continue
            pool['outside_target'] = pool.N.lt(1000) if name!='supported' else pool.N.ge(10)
            pool['type_diff'] = (pool.node_type!=a.node_type).astype(int)+(pool.src_type!=a.src_type).astype(int)+(pool.dst_type!=a.dst_type).astype(int)
            pool['competition_diff'] = (pool.share.fillna(-1)-(a.share if pd.notna(a.share) else -1)).abs()
            pool['activity_diff'] = np.abs(np.log2(pool.activity/a.activity))
            b = pool.sort_values(['type_diff','outside_target','activity_diff','competition_diff','node']).iloc[0]
            for role,r in [('FP',a),('TN',b)]:
                rawrows = db.execute('select rid from events where ((src=? and dst=?) or (src=? and dst=?)) and op=? and ns=? and uuid=?',
                                    (int(r.u),int(r.v),int(r.v),int(r.u),op.replace('EVENT_',''),int(r.ns),r.uuid)).fetchall()
                examples.append(dict(pair=op+'_'+name,role=role,raw_ids='|'.join(str(k[0]) for k in rawrows),
                                     **{k:r[k] for k in ['node','op','src_label','dst_label','node_type','N','Qc','share','activity','score','w','row','uuid']}))
    pd.DataFrame(examples).to_csv(out/'cases.csv',index=False)
    evidence = dict(normal_events=int(good.sum()),test_events=len(e),edge_y0=int((e.y==0).sum()),nodes=len(peaks),
                    fp_low=int(((peaks.fp==1)&(peaks.N<10)).sum()),fp_unseen=int(((peaks.fp==1)&(peaks.N==0)).sum()),
                    fp_high=int(((peaks.fp==1)&(peaks.N>=1000)).sum()),fp_tied=int(((peaks.fp==1)&(peaks.ties>1)).sum()),
                    fp_cross_bin=int((ties[ties.fp==1].groupby('node').bin.nunique()>1).sum()),
                    ties=len(ties),max_score_error=float((ties.loss-ties.score).abs().max()))
    (out/'summary.json').write_text(json.dumps(evidence,indent=2))
    print(evidence,flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('raw',type=Path)
    args = p.parse_args()
    analyze(args.source,Path(os.environ['RUN_OUTPUT_DIR']),args.raw)
