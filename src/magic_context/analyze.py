"""Summarize MAGIC contexts and compare fixed neighbors of original decision groups."""
from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

BINS=['0','1–9','10–99','100–999','≥1000']
def bins(x):
    return pd.cut(x,[-1,0,9,99,999,np.inf],labels=BINS)

def read(base):
    # Align each matrix row with the metadata from its own frozen batch.
    meta=[]; matrices={k:[] for k in ['x','desc','h']}
    for w,file in enumerate(sorted(base.glob('*.npz'))):
        with np.load(file) as d:
            meta.append(pd.DataFrame({k:d[k] for k in d if k not in matrices}).assign(window=w))
            for k in matrices:matrices[k].append(d[k])
    return pd.concat(meta,ignore_index=True),{k:np.concatenate(v) for k,v in matrices.items()}

def means(matrix,ctx,count):
    out=np.zeros((len(count),matrix.shape[1]),dtype='float64')
    np.add.at(out,ctx,matrix)
    return out/count[:,None]

def main(src,out):
    train,tr=read(src/'train');test,te=read(src/'test')
    pred=pd.read_pickle(src/'pred.pkl')
    ct,ic,count=np.unique(train.context,return_inverse=True,return_counts=True)
    stats=train.groupby('context').agg(N=('id','size'),train_ids=('id','nunique'),windows=('window','nunique'),train_sce=('sce','mean'),train_bce=('bce','mean'),train_score=('score','mean'),train_score_max=('score','max'),valid_train_scores=('score','count'))
    assert np.array_equal(stats.N,count)
    peak=test.sort_values(['score','window'],ascending=[False,True]).drop_duplicates('id').copy()
    first=test.drop_duplicates('id').copy()
    activity=test.groupby('id').degree.sum()
    node=peak.join(pred[['gt','prediction']],on='id',validate='1:1').join(stats,on='context')
    node['N']=node.N.fillna(0).astype(int)
    node['semantic_N']=node.semantic.map(train.semantic.value_counts()).fillna(0).astype(int)
    node['activity']=node.id.map(activity)
    node['bin']=bins(node.N)
    node['first_N']=node.id.map(first.set_index('id').context.map(stats.N)).fillna(0).astype(int)
    node['row_id']=node.index
    node=node.set_index('id',drop=False)
    assert (node.score-pred.loc[node.index,'score']).abs().max()<1e-8
    threshold=json.loads((src/'summary.json').read_text())['threshold']
    node['outcome']=np.select([node['gt'].eq(0)&node.prediction.eq(1),node['gt'].eq(0),node.prediction.eq(1)],['FP','TN','TP'],default='FN')
    assert node.outcome.value_counts().to_dict()=={'FN':71,'TP':1,'FP':43,'TN':286815}
    q=train.groupby('kind')[['sce','bce']].quantile(.95)
    node['ref_sce95']=node.kind.map(q.sce);node['ref_bce95']=node.kind.map(q.bce)
    node['poor']=node.N.gt(0)&((node.train_sce>node.ref_sce95)|(node.train_bce>node.ref_bce95))
    node['category']=np.select([node.N.eq(0)&node.semantic_N.eq(0),node.N.eq(0),node.poor,node.train_score.gt(threshold),node.train_score.isna()],['A_unseen','B_semantic_seen_input_unseen','C_high_train_reconstruction','seen_high_train_reference_score','seen_train_score_missing'],default='D_seen_typical_train_diagnostics')
    test['N']=test.context.map(stats.N).fillna(0).astype(int)
    distribution=[]
    for tag in BINS:
        group=train[bins(train.context.map(stats.N))==tag]
        testg=test[bins(test.N)==tag]
        n=node[(node['gt']==0)&(node.bin==tag)]
        a=node[(node['gt']==0)&(bins(node.first_N)==tag)]
        fp=int(n.prediction.sum());tn=len(n)-fp
        distribution.append(dict(support=tag,train_types=group.context.nunique(),test_types=testg.context.nunique(),train_instances=len(group),unique_train_nodes=group.id.nunique(),test_instances=len(testg),FP=fp,TN=tn,FPR=fp/len(n) if len(n) else None,first_FP=int(a.prediction.sum()),first_TN=int((1-a.prediction).sum())))
    pd.DataFrame(distribution).to_csv(out/'CONTEXT_TRAINING_DISTRIBUTION.csv',index=False)
    rates=node[node['gt']==0].groupby(['category','outcome']).size().unstack(fill_value=0)
    rates['FPR']=rates.FP/(rates.FP+rates.TN)
    rates.to_csv(out/'learning_groups.csv')
    arr_support=np.log1p(node.N.to_numpy());rows=node.row_id.to_numpy()
    control=node.outcome.eq('TN').to_numpy();nids=node.id.to_numpy()
    desc=te['desc'][rows]
    std=np.maximum(np.std(tr['desc'],axis=0),1e-6)
    # Match by type, same support bin, then joint distance, allowing reuse and reporting all discrepancies.
    matches=[]
    for p in np.flatnonzero(~control):
        candidates=np.flatnonzero(control & (node.kind.to_numpy()==node.kind.iloc[p]))
        same=candidates[node.bin.to_numpy()[candidates]==node.bin.iloc[p]]
        if not len(same):continue
        cand=same
        av=np.log1p(node.activity.to_numpy()[cand])-np.log1p(node.activity.iloc[p])
        size=np.log1p(node.window_nodes.to_numpy()[cand])-np.log1p(node.window_nodes.iloc[p])
        deg=np.log1p(node.degree.to_numpy()[cand])-np.log1p(node.degree.iloc[p])
        support=arr_support[cand]-arr_support[p]
        diff=np.sqrt(np.mean(((desc[cand]-desc[p])/std)**2,axis=1))
        distance=abs(av)+abs(size)+abs(deg)+abs(support)+diff
        best=np.lexsort((nids[cand],distance))[:3]
        for rank,j in enumerate(best,1):
            z=cand[j]
            matches.append(dict(case=int(nids[p]),control=int(nids[z]),outcome=node.outcome.iloc[p],rank=rank,kind=node.kind.iloc[p],support=node.N.iloc[p],tn_support=node.N.iloc[z],activity=node.activity.iloc[p],tn_activity=node.activity.iloc[z],nodes=node.window_nodes.iloc[p],tn_nodes=node.window_nodes.iloc[z],degree=node.degree.iloc[p],tn_degree=node.degree.iloc[z],input_rms=diff[j],distance=distance[j]))
    match=pd.DataFrame(matches)
    match[match.outcome=='FP'].to_csv(out/'MATCHED_FP_TN_ANALYSIS.csv',index=False)
    all_ids=np.unique(np.r_[node.loc[~control,'id'].values,match.control.values])
    obs=node.loc[all_ids].copy()
    # Type centroids: prevent repeated occurrences occupying every neighbor slot.
    own=obs.context.to_numpy(); queryrows=obs.row_id.to_numpy()
    centers={k:means(tr[k],ic,count) for k in ('x','desc','h')}
    target=centers['x']
    # Positive link targets described by the distribution of endpoint types and directional degree.
    degree_target=np.log1p(train[['ins','outs']].to_numpy())
    link=means(np.c_[tr['desc'][:,tr['x'].shape[1]:2*tr['x'].shape[1]],degree_target],ic,count)
    qlink=np.c_[te['desc'][queryrows,tr['x'].shape[1]:2*tr['x'].shape[1]],np.log1p(obs[['ins','outs']].to_numpy())]
    near=[];fn_measures={};mal_refs=[]
    semantic_support=train.groupby(['context','semantic']).size().to_dict()
    for space,k in [('input','desc'),('embedding','h')]:
        scale=tr[k].std(axis=0);scale[scale<1e-6]=1
        center=centers[k]/scale;query=te[k][queryrows]/scale
        d,j=NearestNeighbors(n_neighbors=min(51,len(ct)),algorithm='brute').fit(center).kneighbors(query)
        # Actual individual reference in addition to centroid distances.
        sample_index=NearestNeighbors(n_neighbors=1,algorithm='brute').fit(tr[k]/scale)
        obs['nearest_train_'+space]=sample_index.kneighbors(query)[0][:,0]
        for i,r in enumerate(obs.itertuples()):
            if r.outcome in ('FN','TP'):
                z=j[i,0]
                mal_refs.append(dict(node=r.id,outcome=r.outcome,space=space,nearest_context=int(ct[z]),distance=d[i,0],train_instances=int(count[z]),same_semantic_instances=semantic_support.get((ct[z],r.semantic),0),own_input_support=r.N,own_semantic_support=r.semantic_N,original_score=r.score))
            sel=np.flatnonzero(ct[j[i]]!=own[i])
            for K in (5,10,20,50):
                positions=sel[:K];neighbors=j[i,positions]
                weights=count[neighbors];diff=np.sqrt(np.mean((target[neighbors]-te['x'][r.row_id])**2,axis=1))
                linkdiff=np.sqrt(np.mean((link[neighbors]-qlink[i])**2,axis=1))
                ne=dict(node=r.id,outcome=r.outcome,space=space,K=K,available=len(weights),own_N=r.N,
                    distance=float(d[i,positions].mean()),neighbor_N=int(weights.sum()),max_N=int(weights.max()),
                    absolute_high=bool((weights>=1000).any()),relative_high=bool((weights>r.N).any()),
                    diff_target_mass=int(weights[diff>1e-6].sum()),target_delta=float(np.average(diff,weights=weights)),structural_delta=float(np.average(linkdiff,weights=weights)),
                    diff_high=bool(((weights>=1000)&(diff>1e-6)).any()),diff_relative=bool(((weights>r.N)&(diff>1e-6)).any()))
                near.append(ne)
                if r.outcome=='FP' and K==50:
                    fn_measures[r.id,space]=[(int(ct[z]),int(count[z]),float(d[i,pos]),float(a),float(b)) for pos,z,a,b in zip(positions,neighbors,diff,linkdiff)]
    obs['score_rank_desc']=obs.score.map(node.score.rank(ascending=False,method='min').groupby(node.score).first())
    cols=['id','outcome','kind','window','context','semantic','semantic_N','N','train_ids','windows','train_sce','train_bce','train_score','train_score_max','sce','bce','score','score_rank_desc','category','activity','degree','window_nodes','nearest_train_input','nearest_train_embedding']
    obs[obs.outcome=='FP'][cols].to_csv(out/'ALL_FP_CONTEXT_EVIDENCE.csv',index=False)
    # FN evidence in aggregate feature/distance units; no paired attack-to-benign behavior recipes.
    mal=obs[obs.outcome.isin(['FN','TP'])][['id','outcome','score','score_rank_desc','semantic_N','N','nearest_train_input','nearest_train_embedding','category']].copy()
    mal[mal.outcome=='FN'].to_csv(out/'ALL_FN_CONTEXT_EVIDENCE.csv',index=False)
    obs[obs.outcome.isin(['FP','TN'])][cols].to_csv(out/'TRAIN_TEST_LEARNING_COMPARISON.csv',index=False)
    pd.DataFrame(mal_refs).to_csv(out/'MALICIOUS_REFERENCE_CONTEXTS.csv',index=False)
    near=pd.DataFrame(near)
    near[near.outcome.isin(['FP','TN'])].to_csv(out/'SIMILAR_CONTEXT_ANALYSIS.csv',index=False)
    details=[dict(node=n,space=s,rank=i,neighbor_context=c,N=f,distance=d,root_target_rms=a,structural_target_rms=b) for (n,s),entries in fn_measures.items() for i,(c,f,d,a,b) in enumerate(entries,1)]
    pd.DataFrame(details).to_csv(out/'FP_neighbor_details.csv',index=False)
    paired=[]
    for (space,K),ng in near.groupby(['space','K']):
        ng=ng.set_index('node')
        for group,mg in match.groupby('outcome'):
            for metric in ['neighbor_N','max_N','target_delta','structural_delta','diff_target_mass','diff_high','diff_relative','distance']:
                a=mg.case.map(ng[metric]).astype(float);b=mg.control.map(ng[metric]).astype(float)
                pair=pd.DataFrame(dict(a=a,b=b,id=mg.case)).groupby('id')[['a','b']].mean()
                paired.append(dict(outcome=group,space=space,K=K,metric=metric,matched=len(pair),case_mean=pair.a.mean(),tn_mean=pair.b.mean(),case_median=pair.a.median(),tn_median=pair.b.median(),cases_larger=int((pair.a>pair.b+1e-8).sum()),cases_smaller=int((pair.a+1e-8<pair.b).sum())))
    pd.DataFrame(paired).to_csv(out/'neighbor_contrasts.csv',index=False)
    fp=obs[obs.outcome=='FP'];m=match[match.outcome=='FP'];g=obs[obs.outcome.isin(['FN','TP'])]
    score_assoc={k:float(spearmanr(g.score,g[k])[0]) for k in ['N','nearest_train_input','nearest_train_embedding']}
    matched_summary={}
    for cgroup,mg in match.groupby('outcome'):
        matched_summary[cgroup]={}
        for k in ['nearest_train_input','nearest_train_embedding','train_sce','train_bce','sce','bce','train_score','score']:
            pairs=pd.DataFrame({'case':mg.case,'a':mg.case.map(obs[k]),'b':mg.control.map(obs[k])}).groupby('case')[['a','b']].mean().dropna()
            matched_summary[cgroup][k]=dict(pairs=len(pairs),case_median=pairs.a.median(),control_median=pairs.b.median(),above=int((pairs.a>pairs.b).sum()))
    n_top=max(1,int(np.ceil(len(count)*.01)))
    summary=dict(train_nodes=train.id.nunique(),train_instances=len(train),train_types=len(ct),top1pct_types=n_top,top1pct_mass=np.sort(count)[-n_top:].sum()/len(train),test_instances=len(test),test_types=test.context.nunique(),unseen_test_types=test.loc[test.N==0,'context'].nunique(),fp_categories=fp.category.value_counts().to_dict(),matched_FP=m.case.nunique(),unique_TN=m.control.nunique(),reuse=len(m)-m.control.nunique(),matched_links=len(m),malicious=len(mal),malicious_score_association=score_assoc,malicious_score_summary=mal.groupby('outcome').score.agg(['min','median','max']).to_dict('index'),match_diagnostics=matched_summary,train_score_missing=int(train.score.isna().sum()),sce_bce_cutoffs=q.to_dict('index'),train_score_scale=json.loads((src/'summary.json').read_text())['scale'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main(Path(sys.argv[1]),Path(os.environ['RUN_OUTPUT_DIR']))
