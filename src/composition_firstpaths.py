"""Prepare remapped content paths and tie-preserving normal-node score evidence."""
import argparse
import csv
import json
import os
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml

from action_composition import ro, sha
from action_index import signature
from composition_compatibility import predictions
from velox_bias_validation.support import OPS

def valid(x):
    return x is not None and str(x).strip().lower() not in ('','null','none','<unknown>','unknown')

def prepare(root, out, audit):
    run=root/'tdsc-runs/cadets-velox-firstpaths-rebuild-20261126b'
    cell=run/'output/CADETS_E3-velox-full'
    old=root/'velox_action_composition'
    proof=json.loads(audit.read_text())
    pred,verified=predictions(cell)
    assert proof['topology_reusable'] and proof['predictions']==verified
    cfg=yaml.safe_load((cell/'resolved_config.yml').read_text())
    native=ro(old/'native.sqlite')
    native.execute('attach database ? as sem',((old/'semantic.sqlite').as_uri()+'?mode=ro',))
    old_sem={n:s for n,s in native.execute('select id,semantic from sem.attributes')}
    dic=torch.load(Path(cfg['construction']['_dicts_dir'])/'indexid2msg.pkl')
    idx=ro(root/'tdsc-analysis-index/CADETS_E3/20261124-v1/events.sqlite')
    evidence=root/'tdsc-runs/cadets-first-file-paths-20261126a/output'
    frozen=Path(json.loads((evidence/'summary.json').read_text())['evidence'])
    assert sha(frozen)==frozen.stem
    paths=json.loads(frozen.read_text())['paths']
    nodes={}
    signatures={}
    complete={}
    old_ids={}
    changed={}
    for n,kind,path,cmd,uid in idx.execute('select id,kind,path,cmd,uuid from nodes'):
        if n not in old_sem: continue
        if kind=='file' and not valid(path): path=paths.get(uid)
        sem=dict(kind=kind,paths=[path] if valid(path) else [],commands=[cmd] if valid(cmd) else [],subtypes=[])
        s=signature(sem)
        key=signatures.setdefault(s,len(signatures)+1)
        complete[key]=bool(sem['paths'] or sem['commands'])
        if kind=='netflow':
            complete[key]=all(valid(t) for t in str(path).replace(' -> ',':').split(':'))
        old_sig=signature(old_sem.pop(n))
        old_key=old_ids.setdefault(old_sig,len(old_ids)+1)
        native_kind,label=dic[str(n)]
        assert kind==native_kind
        nodes[n]=(key,old_key,kind,label,s!=old_sig)
    assert not old_sem
    del old_ids,dic
    pd.DataFrame([(v,k,complete[v]) for k,v in signatures.items()],columns=['id','content','complete']).to_csv(out/'signatures.csv.gz',index=False)
    counts=defaultdict(Counter)
    conditions=defaultdict(Counter)
    actions={}
    complete_actions={}
    old_actions=set()
    remaps=set()
    config_changes=Counter()
    degree=Counter()
    scores={r['node_id']:r['score'] for r in map(json.loads,(cell/'knee/objects.jsonl').open()) if not r['gt']}
    ties=[]
    maxima={}
    loss_dir=Path(cfg['training']['_edge_losses_dir'])/'test'/'model_epoch_11'
    files=sorted(loss_dir.glob('*.csv'),key=lambda p:int(p.stem.rsplit('_b',1)[1]))
    assert len(files)==2197
    batch=0
    with (out/'events.tsv').open('w') as stream:
        for w,split,name,nrows in native.execute('select w,split,name,n from windows order by w'):
            test=split=='test'
            if test:
                dfs=[]
                for start in range(0,nrows,1024):
                    f=files[batch]
                    assert name in f.name and int(f.stem.rsplit('_b',1)[1])==batch
                    batch+=1
                    df=pd.read_csv(f)
                    assert len(df)==min(1024,nrows-start)
                    dfs.append(df)
                losses=list(pd.concat(dfs)[['loss','srcnode','dstnode','time','edge_type']].itertuples(index=False,name=None))
            rows=[]
            group=0
            last=None
            query='select e.row,e.u,e.v,e.ns,e.op,e.uuid,e.y,s.seq from edges e join sem.source s using(w,row) where w=? order by e.ns,s.seq,e.row'
            for row,u,v,ns,op,uid,y,seq in native.execute(query,(w,)):
                src,orig_u,_,sl,cs=nodes[u]
                dst,orig_v,_,dl,cd=nodes[v]
                cond=(sl,dl)
                key=(src,op,dst)
                aid=actions.setdefault(key,len(actions)+1)
                complete_actions[aid]=complete[src] and complete[dst]
                if test:
                    loss,lu,lv,ts,rel=losses[row]
                    assert (u,v,ns,op)==(lu,lv,ts,'EVENT_'+OPS[int(rel)-1])
                    for node in {u,v}:
                        if node in scores:
                            degree[node]+=1
                            maxima[node]=max(maxima.get(node,-1),loss)
                            if loss==scores[node]:
                                ties.append((node,w,row,u,v,ns,seq,uid,aid,loss,cond))
                if y or test and (u not in scores or v not in scores): continue
                if not test: conditions[cond][op]+=1
                counts[aid][split]+=1
                oldkey=(orig_u,op,orig_v)
                old_actions.add(oldkey)
                remaps.add((oldkey,aid))
                config_changes[split]+=int(cs or cd)
                if (ns,seq)!=last:
                    group+=1
                    last=(ns,seq)
                rows.append((row,u,v,group,aid,int(complete_actions[aid])))
            stream.write(f'{w} {int(test)} {len(rows)}\n')
            stream.writelines(' '.join(map(str,r))+'\n' for r in rows)
    assert maxima==scores and {r[0] for r in ties}==set(scores)
    record=[(a,*key,complete_actions[a],counts[a]['train'],counts[a]['test']) for key,a in actions.items()]
    pd.DataFrame(record,columns=['action','src','op','dst','complete','train','test']).to_pickle(out/'actions.pkl')
    pd.DataFrame(record,columns=['action','src','op','dst','complete','train','test']).to_csv(out/'actions.csv.gz',index=False)
    state=({n:(nodes[n][2],degree[n],s) for n,s in scores.items()},ties,conditions)
    with (out/'peak_input.pkl').open('wb') as fp: pickle.dump(state,fp)
    audit=dict(old_observed_actions=len(old_actions),new_normal_actions=len(counts),old_new_pairs=len(remaps),
               events_with_changed_signature=config_changes,peak_node_event_pairs=len(ties),
               peaks_source=str(loss_dir),firstpaths_sha256=frozen.stem)
    (out/'remap.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    args=p.parse_args()
    out=Path(os.environ['RUN_OUTPUT_DIR'])
    prepare(args.root,out,args.audit)
    exe=out/'composition_instances'
    subprocess.run(['g++','-std=c++17','-O3',str(Path(__file__).with_name('composition_instances.cpp')),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(out)],check=True)
