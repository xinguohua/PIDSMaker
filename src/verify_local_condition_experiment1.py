"""Independent delivery checks against the frozen training-window index."""
import argparse,hashlib,json,os,re,sqlite3,subprocess
from pathlib import Path
import pandas as pd
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
out=Path(os.environ['RUN_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
cfg=json.loads((a.results/'config.json').read_text());cell=Path(cfg['cell']);c=sqlite3.connect('file:'+cfg['db']+'?mode=ro',uri=True)
d=pd.read_csv(a.results/'observed_condition_operations.csv');w=pd.read_csv(a.results/'train_windows.csv');ref=pd.read_csv(a.reference)
m=w.merge(ref,left_on='name',right_on='window_name',validate='one_to_one');assert len(m)==581
for left,right in [('n','edges'),('graph','transformed_file'),('tensor','tensor_file'),('graph_sha','graph_sha256'),('tensor_sha','tensor_sha256')]:assert (m[left]==m[right]).all(),(left,right)
assert ref.batches.sum()==3665
log=(cell.parents[1]/'run.log').read_text();assert 'Train num graphs: 3665' in log
# Regroup node-pair counts into label pairs independently from the original label SQL.
counts={};labels=dict(c.execute('select id,label from nodes'))
for u,v,op,n in c.execute("select e.u,e.v,e.op,count(*) from edges e join windows w on w.w=e.w where w.split='train' group by e.u,e.v,e.op"):
    k=(labels[u],labels[v],op);counts[k]=counts.get(k,0)+n
actual={(r.src_label,r.dst_label,r.op):r.N_cr for r in d.itertuples()};assert counts==actual
assert set(c.execute("select distinct y from edges e join windows w on w.w=e.w where w.split='train'"))=={(0,)}
for name,h in json.loads((a.results/'output_sha256.json').read_text()).items():assert sha(a.results/name)==h
for scope,x in [('all',d),('multi',d[d.operations_in_condition>1])]:
    four=pd.read_csv(a.results/'four_groups.csv');four=four[four.scope==scope]
    for _,z in four.groupby('threshold'):assert z.types.sum()==len(x) and z.events.sum()==x.N_cr.sum()
    ranks=pd.read_csv(a.results/'operation_local_rank.csv');ranks=ranks[ranks.scope==scope]
    assert ranks.types.sum()==len(x) and ranks.events.sum()==x.N_cr.sum()
    if scope=='multi':
        bins=pd.read_csv(a.results/'support_plot_bins.csv');assert bins[bins.weight=='types']['count'].sum()==len(x);assert bins[bins.weight=='events']['count'].sum()==x.N_cr.sum()
for r in pd.read_csv(a.results/'representative_examples.csv').itertuples():
    witness=json.loads(r.witness);e=c.execute('select u,v,op,uuid from edges where w=? and row=?',witness[:2]).fetchone()
    assert (labels[e[0]],labels[e[1]],e[2])==(r.src_label,r.dst_label,r.op) and e[3]==witness[6]
hl=d[(d.Qg>=.1)&(d.Ql<.1)];hl.to_csv(out/'global_high_local_low.csv',index=False)
config_text=(cell/'resolved_config.yml').read_text();assert re.search(r'fix_buggy_graph_reindexer:\s*false',config_text)
code=cell.parents[1]/'code';assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==cfg['training_commit']
source=subprocess.check_output(['git','-C',str(code),'show',cfg['training_commit']+':pidsmaker/utils/data_utils.py'],text=True)
lines=source.splitlines();extract='\n'.join(f'{i+1}: {line}' for i,line in enumerate(lines) if 793<=i+1<=916)
(out/'frozen_reindex_source.txt').write_text(extract)
(out/'training_config_excerpt.txt').write_text('\n'.join(f'{i+1}: {line}' for i,line in enumerate(config_text.splitlines()) if any(k in line for k in ['fix_buggy','node_features:','intra_graph_batch_size:','used_method:'])))
report=dict(analysis_commit=cfg['analysis_commit'],verification_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_commit=cfg['training_commit'],training_head_verified=True,reference_index=str(a.reference),reference_sha256=sha(a.reference),matched_windows=581,matched_batches=3665,matched_edges=3426589,window_paths_and_stored_graph_tensor_hashes_match=True,graph_tensor_bytes_rehashed_this_run=False,prior_rowwise_audit='Reuse NATIVE_INPUT_AUDIT.json: 3426589 graph/tensor/loader rows; current run validates index identity, not new vector audit',independent_node_pair_groupby_equals_label_groupby=True,all_training_y_zero=True,source_and_target='nodes[e.u].label -> nodes[e.v].label; original graph/tensor direction, not inferred actor/object',output_hashes_verified=True,example_witnesses_verified=16,plot_bin_totals_match=True,old_reindex_branch=True,final_encoder_vectors_verified=False,training_config_sha256=sha(cell/'resolved_config.yml'),training_manifest_sha256=sha(cell/'manifest.json'),results=str(a.results))
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
