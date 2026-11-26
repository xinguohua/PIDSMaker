"""Experiment 1: observed native directed-label conditions; read-only frozen inputs."""
import argparse, hashlib, json, os, sqlite3, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

TRAIN_COMMIT = '8a7303915422a85dead0a627ab1a031b7ef0c6b5'
SQL = '''SELECT s.label src_label,t.label dst_label,e.op,COUNT(*) N_cr,
 MIN(e.w*1000000000+e.row) witness_key
 FROM windows w JOIN edges e ON e.w=w.w
 JOIN nodes s ON s.id=e.u JOIN nodes t ON t.id=e.v
 WHERE w.split='train' GROUP BY s.label,t.label,e.op'''
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--cell',required=True)
    a=p.parse_args();out=Path(os.environ['RUN_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    start=time.time();cell=Path(a.cell);manifest=json.loads((cell/'manifest.json').read_text())
    assert manifest['commit']==TRAIN_COMMIT
    config=dict(run_id=os.environ.get('RUN_ID'),analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_commit=TRAIN_COMMIT,training_run=manifest['task_id'],db=a.db,cell=a.cell,condition=['src_label','dst_label'],main_threshold=0.1,sensitivity=[0.05,0.2],high_rule='>=',threshold_status='Experiment convention; not a literature standard',type_unit='observed (condition, operation)',event_weight='N_cr',ties='all tied maxima count as highest',example_rule='per operation among multi-operation conditions: maximum N_cr local-high and maximum N_cr local-low; lexical tie break; no predictions used',sql=SQL,seed=None)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    c=sqlite3.connect('file:'+a.db+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
    before=Path(a.db).stat();audit={'manifest':manifest,'database_size':before.st_size,'database_mtime_ns':before.st_mtime_ns,'query_plan':c.execute('EXPLAIN QUERY PLAN '+SQL).fetchall()}
    windows=pd.read_sql_query("select w.*,count(e.row) actual_edges from windows w left join edges e on e.w=w.w where w.split='train' group by w.w",c)
    assert len(windows)==581 and windows.n.sum()==3426589 and (windows.n==windows.actual_edges).all()
    windows.to_csv(out/'train_windows.csv',index=False)
    audit['training_y_counts']=c.execute("select e.y,count(*) from edges e join windows w on w.w=e.w where w.split='train' group by e.y").fetchall()
    audit['null_labels']=c.execute("select count(*) from edges e join windows w on w.w=e.w left join nodes s on s.id=e.u left join nodes t on t.id=e.v where w.split='train' and (s.label is null or t.label is null)").fetchone()[0]
    assert audit['null_labels']==0
    d=pd.read_sql_query(SQL,c);assert d.N_cr.sum()==3426589
    keys=['src_label','dst_label'];g=d.groupby(keys,sort=True)
    d['N_c']=g.N_cr.transform('sum');d['operations_in_condition']=g.op.transform('size');d['local_max']=g.N_cr.transform('max')
    glob=d.groupby('op').N_cr.sum().sort_values(ascending=False).rename('N_r').reset_index();glob['global_rank']=glob.N_r.rank(method='min',ascending=False).astype(int)
    glob['P_r']=glob.N_r/glob.N_r.sum();glob['Qg']=glob.N_r/glob.N_r.max()
    independent=pd.read_sql_query("select e.op,count(*) n from edges e join windows w on w.w=e.w where w.split='train' group by e.op",c).set_index('op').n
    assert glob.set_index('op').N_r.sort_index().equals(independent.sort_index())
    d=d.merge(glob,on='op');d['P_r_given_c']=d.N_cr/d.N_c;d['Ql']=d.N_cr/d.local_max;d['is_highest']=d.N_cr==d.local_max
    conditions=d.drop_duplicates(keys)[keys+['N_c','operations_in_condition','local_max']]
    assert len(conditions)==36605 and (conditions.operations_in_condition>1).sum()==3492
    assert np.allclose(d.groupby(keys).P_r_given_c.sum(),1)
    conditions.to_csv(out/'conditions.csv',index=False);glob.to_csv(out/'global_operations.csv',index=False)
    d.to_csv(out/'observed_condition_operations.csv',index=False)
    multi=d[d.operations_in_condition>1];strata=[]
    for name,mask in [('single',conditions.operations_in_condition==1),('multi',conditions.operations_in_condition>1)]:
        v=conditions[mask];strata.append(dict(stratum=name,conditions=len(v),condition_fraction=len(v)/len(conditions),events=int(v.N_c.sum()),event_fraction=float(v.N_c.sum()/3426589)))
    pd.DataFrame(strata).to_csv(out/'condition_strata.csv',index=False)
    four=[];oprows=[];continuous=[]
    for scope,x in [('all',d),('multi',multi)]:
        for threshold in [0.1,0.05,0.2]:
            for gh,lh in [(True,True),(True,False),(False,True),(False,False)]:
                z=x[((x.Qg>=threshold)==gh)&((x.Ql>=threshold)==lh)]
                four.append(dict(scope=scope,threshold=threshold,group=('H' if gh else 'L')+('H' if lh else 'L'),types=len(z),type_fraction=len(z)/len(x),events=int(z.N_cr.sum()),event_fraction=float(z.N_cr.sum()/x.N_cr.sum()),conditions=len(z[keys].drop_duplicates())))
        for op,z in x.groupby('op'):
            for highest in [True,False]:
                v=z[z.is_highest==highest];oprows.append(dict(scope=scope,op=op,global_rank=int(z.global_rank.iloc[0]),N_r=int(z.N_r.iloc[0]),highest=highest,types=len(v),type_denominator=len(z),type_fraction=len(v)/len(z),events=int(v.N_cr.sum()),event_denominator=int(z.N_cr.sum()),event_fraction=float(v.N_cr.sum()/z.N_cr.sum())))
        for field in ['Qg','Ql','P_r_given_c']:
            v=x.sort_values(field)
            for weighting,weights in [('type',np.ones(len(v))),('event',v.N_cr.to_numpy())]:
                ecdf=np.cumsum(weights)/sum(weights)
                for q in [0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]:
                    i=min(np.searchsorted(ecdf,q),len(v)-1);continuous.append(dict(scope=scope,field=field,weighting=weighting,quantile=q,value=float(v[field].iloc[i])))
    pd.DataFrame(four).to_csv(out/'four_groups.csv',index=False);pd.DataFrame(oprows).to_csv(out/'operation_local_rank.csv',index=False);pd.DataFrame(continuous).to_csv(out/'continuous_quantiles.csv',index=False)
    examples=[]
    for op,z in multi.groupby('op'):
        hi=z[z.Ql>=.1];lo=z[z.Ql<.1]
        if len(hi) and len(lo):
            for side,v in [('high',hi),('low',lo)]:
                r=v.sort_values(['N_cr']+keys,ascending=[False,True,True]).iloc[0].to_dict();r['side']=side
                w,row=divmod(int(r['witness_key']),1000000000)
                witness=c.execute('select e.*,w.name,s.uuid,t.uuid from edges e join windows w on e.w=w.w join nodes s on e.u=s.id join nodes t on e.v=t.id where e.w=? and e.row=?',(w,row)).fetchone()
                r['witness']=json.dumps(witness);examples.append(r)
    pd.DataFrame(examples).to_csv(out/'representative_examples.csv',index=False)
    # Save continuous empirical CDF and exact binned plot inputs, without scatter overplotting.
    ecdf=multi.groupby(['op','Qg','Ql'],as_index=False).agg(types=('N_cr','size'),events=('N_cr','sum')).sort_values(['op','Ql'])
    for weight in ['types','events']: ecdf[weight+'_cdf']=ecdf.groupby('op')[weight].cumsum()/ecdf.groupby('op')[weight].transform('sum')
    ecdf.to_csv(out/'continuous_ecdf.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(glob.op.str.replace('EVENT_',''),glob.N_r,color='#3677a8');ax.set_yscale('log');ax.set_ylabel('Training edge occurrences (log scale)');ax.tick_params(axis='x',rotation=40);ax.set_title('CADETS E3 / Velox: global operation frequency');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/('global_frequency.'+ext),dpi=200)
    plt.close(fig)
    bins=np.geomspace(min(multi.Ql.min(),.001),1,45);ops=glob.op.tolist();binned=[];fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,weight in zip(axes,['types','events']):
        matrix=[]
        for op in ops:
            z=multi[multi.op==op];counts,_=np.histogram(z.Ql,bins=bins,weights=None if weight=='types' else z.N_cr)
            matrix.append(counts)
            for i,n in enumerate(counts):binned.append(dict(op=op,weight=weight,Ql_lower=bins[i],Ql_upper=bins[i+1],count=int(n)))
        arr=np.asarray(matrix).T;im=ax.pcolormesh(np.arange(len(ops)+1),bins,np.ma.masked_equal(arr,0),norm=LogNorm(vmin=1,vmax=max(2,arr.max())),cmap='viridis')
        ax.set_yscale('log');ax.set_xticks(np.arange(len(ops))+.5);ax.set_xticklabels([o.replace('EVENT_','')+'\nQg='+format(float(glob[glob.op==o].Qg.iloc[0]),'.3g') for o in ops],rotation=60,ha='right',fontsize=8)
        ax.axhline(.1,color='red',ls='--',lw=1);ax.set_title('Observed (c,r) types' if weight=='types' else 'Training event weight');fig.colorbar(im,ax=ax,label='Count (log color)')
    axes[0].set_ylabel('Local support Ql (multi-operation conditions)');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/('global_vs_local.'+ext),dpi=200)
    plt.close(fig);pd.DataFrame(binned).to_csv(out/'support_plot_bins.csv',index=False)
    assert Path(a.db).stat().st_mtime_ns==before.st_mtime_ns
    audit.update(windows=581,events=int(d.N_cr.sum()),conditions=len(conditions),multi_conditions=int((conditions.operations_in_condition>1).sum()),observed_types=len(d),multi_types=len(multi),strata=strata,main_multi=[r for r in four if r['scope']=='multi' and r['threshold']==.1],independent_operation_counts_match=True,window_counts_match=True,database_unchanged=True,elapsed_seconds=time.time()-start,final_encoder_vectors_verified=False)
    (out/'summary.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='manifest'},indent=2))
    (out/'output_sha256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='output_sha256.json'},indent=2))
if __name__=='__main__':main()
