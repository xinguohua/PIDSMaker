"""Analyze exact targets, objective-equivalent directions, and original normal-node outcomes."""
from pathlib import Path
import json,os,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def dist(x,y):
    return (1-np.clip((x*y).sum(axis=-1),-1,1))**3


def read(folder,prior=None):
    pieces=[]
    for path in sorted(folder.glob('*.npz')):
        w,draw=map(int,path.stem.split('-'))
        with np.load(path) as f:
            t=pd.DataFrame({name:f[name] for name in ['id','kind','loss','indeg']})
            for key in ['g0','g1','g2']: t[key]=[x.tobytes().hex() for x in f[key]]
            for name,prefix in [('target','y'),('output','p'),('moment','m')]:
                for j,column in enumerate(f[name].T):t[prefix+str(j)]=column
            t['window']=w;t['draw']=draw;t['size']=int(f['size'])
            if prior:
                with np.load(prior/'test'/f'{w:04d}.npz') as ref:
                    rows=f['row'];t['score']=ref['score'][rows]
                    hs=ref['h'][rows]
                    for j,col in enumerate(hs.T):t['h'+str(j)]=col
            pieces.append(t)
    return pd.concat(pieces,ignore_index=True)


def ranks(a):
    return spearmanr(a.iloc[:,0],a.iloc[:,1])[0] if len(a)>2 and a.nunique().min()>1 else np.nan


def fit(t,key,target,dirs,write=False):
    freq=t.groupby([key,target]).size().rename('n').reset_index()
    freq=freq.sort_values([key,'n',target],ascending=[True,False,True])
    grp=freq.groupby(key,sort=False)
    groups=grp.n.agg(['sum','max','min','size']).rename(columns={'sum':'instances','max':'majority_N','min':'min_N','size':'target_types'})
    groups['major']=grp[target].first();groups['ratio']=groups.majority_N/groups.min_N
    freq['share']=freq.n/freq[key].map(groups.instances)
    freq['entropy_part']=-freq.share*np.log2(freq.share)
    groups['entropy_bits']=freq.groupby(key).entropy_part.sum()
    freq['distance_to_major']=dist(dirs[freq[target]],dirs[freq[key].map(groups.major)])
    groups['max_major_SCE']=freq.groupby(key).distance_to_major.max()
    groups['tied']=freq.n.eq(freq[key].map(groups.majority_N)).groupby(freq[key]).sum()>1
    groups['unique_nodes']=t.groupby(key).id.nunique()
    groups['windows']=t.groupby(key).window.nunique()
    groups['indeg']=t.groupby(key).indeg.first()
    groups['modal_kind']=t.groupby([key,'kind']).size().reset_index(name='n').sort_values('n').drop_duplicates(key,keep='last').set_index(key).kind
    groups['mean_window_size']=t.groupby(key)['size'].mean()
    if write:
        obs=t.groupby([key,target]).agg(nodes=('id','nunique'),windows=('window','nunique'))
        freq=freq.join(obs,on=[key,target])
        # Uniform weighting of batches, mean of masked loss within each batch.
        freq['batch_weight']=t.assign(weight=1/(t['size']//2)).groupby([key,target]).weight.sum().reindex(pd.MultiIndex.from_frame(freq[[key,target]])).values
    return groups,freq


def bias(t,key,target,groups,dirs):
    z=t.copy()
    z['major']=z[key].map(groups.major)
    hit=z.major.notna()
    z['to_major']=np.nan
    z.loc[hit,'to_major']=dist(unit(z.loc[hit,pcols].to_numpy(float)),dirs[z.loc[hit,'major'].astype(int)])
    z['margin']=z.loss-z.to_major
    z['distinct']=False
    z.loc[hit,'distinct']=dist(dirs[z.loc[hit,target]],dirs[z.loc[hit,'major'].astype(int)])>1e-9
    return z


def summarize(t,key,target,dirs,write=False):
    groups,freq=fit(t,key,target,dirs,write)
    z=bias(t,key,target,groups,dirs)
    # Tied maxima have no unique majority. Retain them in distributions, omit from directional rates.
    valid=z[key].map(groups.target_types).gt(1)&~z[key].map(groups.tied)
    minor=z[valid & z[target].ne(z.major)]
    maj=z[valid & z[target].eq(z.major)].copy()
    groups['minor_loss']=minor.groupby(key).loss.mean()
    groups['major_loss']=maj.groupby(key).loss.mean()
    groups['gap']=groups.minor_loss-groups.major_loss
    groups['minor_n']=minor.groupby(key).size()
    groups['major_n']=maj.groupby(key).size()
    groups['minor_bias']=minor.margin.gt(1e-9).groupby(minor[key]).mean()
    groups['mean_margin']=minor.groupby(key).margin.mean()
    # All candidate targets in the group can be reconstructed exactly from the saved
    # prediction and target dictionary. Reverse bias uses the nearest competing target.
    reverse={}
    targetsets=freq.groupby(key)[target].agg(list)
    for name,sub in maj.groupby(key):
        alternatives=[a for a in targetsets[name] if a!=sub.major.iloc[0]]
        pred=unit(sub[pcols].to_numpy(float))
        nearest=np.full(len(sub),np.inf)
        for a in alternatives:nearest=np.minimum(nearest,dist(pred,dirs[a]))
        reverse[name]=float((sub.loss.to_numpy()-nearest>1e-9).mean())
    groups['reverse']=pd.Series(reverse,dtype=float)
    return groups,freq,z


def pooled(g):
    g=g.dropna(subset=['minor_bias'])
    return dict(groups=len(g),minor_n=int(g.minor_n.sum()),major_n=int(g.major_n.sum()),
        major_loss=np.average(g.major_loss,weights=g.major_n) if len(g) else None,
        minor_loss=np.average(g.minor_loss,weights=g.minor_n) if len(g) else None,
        group_mean_gap=g.gap.mean(),bias_rate=np.average(g.minor_bias,weights=g.minor_n) if len(g) else None,
        macro_bias=g.minor_bias.mean(),reverse_rate=np.average(g.reverse,weights=g.major_n) if len(g) else None,
        rho_ratio_margin=ranks(g[['ratio','mean_margin']]),rho_ratio_bias=ranks(g[['ratio','minor_bias']]))


def main(replay,prior,out):
    global pcols
    data=read(replay/'train0')
    ycols=list(data.filter(regex='^y[0-9]').columns);pcols=list(data.filter(regex='^p[0-9]').columns)
    def encode(t):
        keys=[np.asarray(r,dtype='float32').tobytes() for r in t[ycols].to_numpy()]
        t['target']=[lookup.setdefault(k,len(lookup)) for k in keys]
        vectors=np.array([np.frombuffer(k,dtype='float32') for k in lookup],dtype=float)
        directions=unit(vectors)
        axes,inv=np.unique(np.round(directions,10),axis=0,return_inverse=True)
        t['axis']=inv[t.target]
        # Native float32 SCE cross-check; downstream distance in float64 for stable ties.
        actual=dist(unit(t[pcols].to_numpy(float)),directions[t.target])
        assert np.max(np.abs(t.loss-actual))<2e-5
        t['native_loss']=t.loss;t['loss']=actual
        return vectors,directions,axes
    lookup={}
    vectors,directions,axes=encode(data)
    pd.DataFrame(vectors,columns=ycols).assign(target=np.arange(len(vectors)),axis=np.unique(np.round(directions,10),axis=0,return_inverse=True)[1]).to_csv(out/'TARGET_VECTORS.csv',index=False)
    ids=data[['window','id','draw','g0','g1','g2','size','kind','indeg','target','axis']]
    ids.to_csv(out/'SIMILAR_INPUT_GROUPS.csv',index=False)
    g,f,z=summarize(data,'g0','target',directions,True)
    f.to_csv(out/'LOCAL_TARGET_DISTRIBUTION.csv',index=False)
    g.to_csv(out/'TARGET_DISCREPANCY_ANALYSIS.csv')
    z[['window','id','g0','target','major','native_loss','loss','to_major','margin']+pcols].to_csv(out/'DIRECTIONAL_RECONSTRUCTION_BIAS.csv',index=False)
    levels=[];base=None
    for key in ['g0','g1','g2']:
        for target,dr in [('target',directions),('axis',axes)]:
            if key=='g0' and target=='target':gs=g
            else:gs,_,_=summarize(data,key,target,dr)
            mixed=gs.target_types>1
            levels.append(dict(level=key,target=target,types=len(gs),repeated=(gs.instances>1).sum(),mixed=mixed.sum(),mixed_instances=gs.loc[mixed,'instances'].sum(),distinct_nodes_mixed=data.loc[data[key].isin(gs.index[mixed]),'id'].nunique(),unequal=(mixed&(gs.ratio>1)).sum(),fourfold=(mixed&(gs.ratio>=4)).sum(),**pooled(gs)))
            if key=='g0' and target=='axis':base=gs
    pd.DataFrame(levels).to_csv(out/'SIMILARITY_SENSITIVITY.csv',index=False)
    # Empirical cutoffs from heterogeneous training groups only; never use test outcomes.
    hetero=g[(g.target_types>1)&~g.tied]
    q=float(hetero.max_major_SCE.median())
    # Small = direction equivalent, large = orthogonal-or-further or empirical median positive gap.
    pos=hetero.loc[hetero.max_major_SCE>1e-9,'max_major_SCE']
    cutoff=float(pos.median()) if len(pos) else 1.
    high=(g.max_major_SCE>=cutoff-1e-9)&(g.max_major_SCE>1e-9)
    arms={'G1':g[(g.max_major_SCE<=1e-9)&(g.ratio>=4)],'G2':g[high&(g.ratio<=2)],'G3':g[high&(g.ratio>=4)]}
    # Negative similarity control: pair equal-frequency-order strata of different in-degree;
    # all members across constituent groups differ in first-hop topology.
    cell=g.reset_index().sort_values(['instances','g0'])
    first=cell[cell.indeg%2==0];other=cell[cell.indeg%2==1]
    n=min(len(first),len(other))
    mapping={}
    if n:
        for i,(a,b) in enumerate(zip(first.g0.iloc[np.linspace(0,len(first)-1,n,dtype=int)],other.g0.iloc[np.linspace(0,len(other)-1,n,dtype=int)])):
            mapping[a]=mapping[b]=i
    combo=data[data.g0.isin(mapping)].copy();combo['low']=combo.g0.map(mapping)
    if len(combo):
        low,_,_=summarize(combo,'low','target',directions)
        # Retain pure constituent types so different targets in the pair always have different inputs.
        pure=combo.groupby('g0').target.nunique().eq(1)
        ok=combo.g0.map(pure).groupby(combo.low).all()
        arms['G4']=low[ok & (low.ratio>=4)&(low.max_major_SCE>=cutoff-1e-9)&(low.max_major_SCE>1e-9)]
    else:arms['G4']=g.iloc[:0]
    pd.DataFrame([dict(arm=a,**pooled(s)) for a,s in arms.items()]).to_csv(out/'FOUR_GROUP_COMPARISON.csv',index=False)
    for a,s in arms.items(): s.to_csv(out/(a+'_groups.csv'))
    controlled=[]
    for arm in ['G1','G2','G4']:
        controls=arms[arm].dropna(subset=['minor_bias'])
        for name,row in arms['G3'].dropna(subset=['minor_bias']).iterrows():
            pool=controls[(controls.modal_kind==row.modal_kind)&(np.abs(np.log(controls.instances/row.instances))<=np.log(2))]
            if len(pool)==0:continue
            distance=np.abs(np.log(pool.instances/row.instances))+np.abs(np.log(pool.mean_window_size/row.mean_window_size))+np.abs(np.log((1+pool.indeg)/(1+row.indeg)))
            best=distance.idxmin()
            controlled.append(dict(arm=arm,g3=name,control=best,match_distance=distance.loc[best],g3_bias=row.minor_bias,control_bias=pool.loc[best,'minor_bias'],g3_gap=row.gap,control_gap=pool.loc[best,'gap']))
    pd.DataFrame(controlled,columns=['arm','g3','control','match_distance','g3_bias','control_bias','g3_gap','control_gap']).to_csv(out/'MATCHED_MECHANISMS.csv',index=False)
    norm_pred=unit(data[pcols].to_numpy(float))
    # Column axis maps back to every exact competitor through TARGET_VECTORS.csv.
    np.savez_compressed(out/'DISTANCE_TO_EVERY_DIRECTION.npz',distance=np.stack([dist(norm_pred,axis) for axis in axes],axis=1).astype('float32'),directions=axes)
    # Continuous input contrast between pairs of distinct exact signatures within narrow certified cells.
    mc=list(data.filter(regex='^m[0-9]').columns)
    cent=data.groupby(['g1','g0'])[mc].mean()
    contrasts=[]
    for key,rows in cent.groupby(level=0):
        if len(rows)<2:continue
        vals=rows.to_numpy(); names=rows.index.get_level_values(1)
        for i in range(1,len(rows)):
            a,b=names[i-1],names[i]
            contrasts.append(dict(cell=key,context_a=a,context_b=b,visible_moment_distance=np.linalg.norm(vals[i]-vals[i-1]),target_distance=dist(directions[g.at[a,'major']],directions[g.at[b,'major']]),support_ratio=max(g.at[a,'instances'],g.at[b,'instances'])/min(g.at[a,'instances'],g.at[b,'instances']),margin_a=g.at[a,'mean_margin'],margin_b=g.at[b,'mean_margin']))
    pd.DataFrame(contrasts,columns=['cell','context_a','context_b','visible_moment_distance','target_distance','support_ratio','margin_a','margin_b']).to_csv(out/'CONTINUOUS_INPUT_COMPARISON.csv',index=False)
    # Second independent native draw for each window, analyzed on its own denominator.
    repeat=read(replay/'train1');_,dr,ax=encode(repeat)
    rep,_,_=summarize(repeat,'g0','target',dr)
    pd.DataFrame([dict(seed=seed,**pooled(frame)) for seed,frame in [(101,g),(202,rep)]]).to_csv(out/'MASK_SENSITIVITY.csv',index=False)
    # Step 6, first use of original decisions. Deterministic peak matches earlier audit.
    test=read(replay/'test0',prior)
    test=test.sort_values(['score','window'],ascending=[False,True]).drop_duplicates('id')
    pred=pd.read_pickle(prior/'pred.pkl')
    test=test.join(pred[['gt','prediction']],on='id')
    test=test[test['gt']==0].copy()
    assert len(test)==286858 and test.prediction.sum()==43
    vec,dr,ax=encode(test)
    z=bias(test,'g0','target',g,dr)
    z['density_distance']=z.score*json.loads((prior/'summary.json').read_text())['scale']
    z['N']=z.g0.map(g.instances).fillna(0)
    z['ratio']=z.g0.map(g.ratio);z['target_gap']=z.g0.map(g.max_major_SCE)
    z['own_N']=[fmap for fmap in f.set_index(['g0','target']).n.reindex(pd.MultiIndex.from_frame(z[['g0','target']])).fillna(0)]
    z['eligible']=(z.own_N>0)&z.distinct&~z.g0.map(g.tied).fillna(True)&(z.ratio>1)
    z['biased']=z.eligible&(z.margin>1e-9)
    cols=['id','window','kind','prediction','g0','target','major','N','own_N','ratio','target_gap','loss','to_major','margin','eligible','biased','density_distance','score']+pcols
    z[cols].to_csv(out/'NORMAL_POPULATION.csv.gz',index=False)
    # Same previously fixed controls; the controls were selected without masked outcomes.
    matched=pd.read_csv(sys.argv[3])
    cases=z.set_index('id')
    subset=cases.loc[np.unique(np.r_[matched.case,matched.control])]
    subset[cols[1:]+list(subset.filter(regex='^h[0-9]').columns)].to_csv(out/'FP_TN_LEARNING_BIAS_ANALYSIS.csv')
    matched.to_csv(out/'MATCHES.csv',index=False)
    counts=z.groupby(['prediction','eligible','biased']).size().to_dict()
    examples=hetero[(hetero.max_major_SCE>1e-9)&(hetero.ratio>=4)].sort_values('instances',ascending=False)
    examples.head(20).to_csv(out/'EXAMPLES.csv')
    examples.sort_values(['minor_bias','instances'],ascending=[True,False]).head(20).to_csv(out/'COUNTEREXAMPLES.csv')
    (out/'summary.json').write_text(json.dumps(dict(train_instances=len(data),nodes=data.id.nunique(),windows=data.window.nunique(),target_types=len(vectors),direction_types=len(axes),cutoff=cutoff,normal_counts={str(k):v for k,v in counts.items()},test_unique=len(test),original_FP=43),default=lambda x:x.item(),indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(os.environ['RUN_OUTPUT_DIR']))
