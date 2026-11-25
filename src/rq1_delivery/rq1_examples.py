"""Recover representative paths from test events, binding predictions to actual participants."""
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Full native semantic signatures may exceed the csv module default.
csv.field_size_limit(sys.maxsize)

out, root = map(Path, sys.argv[1:])
sys.path.insert(0, str(out/'CODE'))
from action_paths import pack, node_category
from action_index import signature
from action_window import read_window
from action_composition import ro, semantic_complete

db=ro(root/'native.sqlite')
db.execute('attach database ? as sem',(f'file:{root}/semantic.sqlite?mode=ro&immutable=1',))
counts=sqlite3.connect(f'file:{out.parent}/counts.sqlite?mode=ro',uri=True)
attrs={}
for n,raw in db.execute('select id,semantic from sem.attributes'):
    value=json.loads(raw)
    attrs[n]=(signature(value),semantic_complete(value))
keys={};train={};complete={}
with (out/'ACTION_COUNTS.csv').open() as stream:
    for r in csv.DictReader(stream):
        t=int(r['action_id'])
        keys[r['source_semantics'],r['operation'],r['destination_semantics']]=t
        train[t]=int(r['train_frequency']);complete[t]=bool(int(r['semantic_complete']))
# A filtered-only Action is never requested by read_window.
threshold=json.loads((out/'OPERATING_POINT.json').read_text())['policy']['threshold']
pred={n:(gt,int(score>threshold),score) for n,gt,score in db.execute('select id,gt,score from pred')}
attack={n for n,(gt,_,_) in pred.items() if gt}
cell=Path(json.loads((out/'INPUT_AUDIT.json').read_text())['cell'])
mapping={u:n for n,u in db.execute('select id,uuid from nodes')}
for rel in json.loads((cell/'manifest.json').read_text())['protocol']['ground_truth_relative_path']:
    for line in (cell.parents[1]/'code/Ground_Truth/orthrus'/rel).read_text().splitlines():
        attack.update(mapping[x] for x in line.replace('"','').strip().split(',') if x in mapping)
flags={(k,n):node_category(b) for k,n,b in counts.execute('select k,id,bits from node_flags')}
categories={'seen_test','compositional_zero_shot','action_unseen','zero_shot_FP','zero_shot_TN','seen_FP'}
found={}
def record(case,k,w,path,seq,tr,te,node):
    if case in found:return
    events=[]
    for _,edge in path:
        row=edge[0]
        sql='''select e.row,e.uuid,e.u,e.v,e.op,e.ns,s.seq,s.file,s.line,s.offset from edges e
               join sem.source s using(w,row) where e.w=? and e.row=?'''
        events.append(dict(zip(('row','uuid','u','v','operation','ns','sequence','raw_file','line','offset'),db.execute(sql,(w,row)).fetchone())))
    nodes={n for e in events for n in (e['u'],e['v'])}
    assert node is None or node in nodes
    assert all(a['v']==b['u'] and (a['ns'],a['sequence'])<(b['ns'],b['sequence']) for a,b in zip(events,events[1:]))
    found[case]=dict(category=case,length=k,window=w,action_ids=seq,train_frequency=tr,test_frequency=te,
        action_train_frequencies=[train[a] for a in seq],target_node=node,events=events,
        nodes={n:dict(semantics=json.loads(attrs[n][0]),gt=pred[n][0],prediction=pred[n][1],
                score=pred[n][2],threshold=threshold,group=flags.get((k,n),'Other')) for n in nodes})

for k in (2,3):
    required=categories-set(found)
    # A zero count in the final disjoint node groups rules out that node case at this length.
    for case,group,label in [('zero_shot_FP','Zero-shot',1),('zero_shot_TN','Zero-shot',0),('seen_FP','Seen',1)]:
        if not any(g==group and pred[n][1]==label for (length,n),g in flags.items() if length==k):required.discard(case)
    with (out/'ACTION_DISTRIBUTIONS'/f'length_{k}_zero_shot_groups.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['types'])==0:
                required.discard({'all_actions_seen':'compositional_zero_shot','unseen_action':'action_unseen'}.get(row['group']))
    for w, in db.execute("select w from windows where split='test' order by w"):
        if not required:break
        groups,_,_,_=read_window(db,w,True,attrs,keys,pred,attack)
        targets=set()
        for n in {n for batch in groups for e in batch for n in e[1:3]}:
            group=flags.get((k,n),'Other')
            if group=='Zero-shot' and ('zero_shot_FP' if pred[n][1] else 'zero_shot_TN') in required:targets.add(n)
            if group=='Seen' and pred[n][1] and 'seen_FP' in required:targets.add(n)
        generic={'seen_test','compositional_zero_shot','action_unseen'}
        if not targets and required.isdisjoint(generic):continue
        incoming=defaultdict(list);outgoing=defaultdict(list)
        target_left=set();target_right=set()
        for g,batch in enumerate(groups):
            for e in batch:
                if e[1] in targets or e[2] in targets:
                    target_left.add(e[2]);target_right.add(e[1])
                incoming[e[2]].append((g,e));outgoing[e[1]].append((g,e))
        lookup={}
        def check(path):
            seq=tuple(e[3] for _,e in path)
            if seq not in lookup:
                lookup[seq]=tuple(map(int,counts.execute('select train,test from counts where k=? and seq=?',(k,pack(seq))).fetchone()))
            tr,te=lookup[seq]
            kind='seen_test' if tr else 'semantic_unknown' if not all(complete[a] for a in seq) else 'compositional_zero_shot' if all(train[a] for a in seq) else 'action_unseen'
            if kind in required:record(kind,k,w,path,seq,tr,te,None);required.discard(kind)
            for _,e in path:
                for n in e[1:3]:
                    g=flags.get((k,n),'Other')
                    case='zero_shot_'+('FP' if pred[n][1] else 'TN') if g=='Zero-shot' and kind=='compositional_zero_shot' else 'seen_FP' if g=='Seen' and pred[n][1] and kind=='seen_test' else None
                    if case in required:record(case,k,w,path,seq,tr,te,n);required.discard(case)
        for g,batch in enumerate(groups):
            if not required:break
            for e in batch:
                if not required:break
                if required.isdisjoint(generic) and e[1] not in targets and e[2] not in targets and e[2] not in target_right and (k==2 or e[1] not in target_left):continue
                for h,f in outgoing[e[2]]:
                    if h<=g:continue
                    if k==2:check(((g,e),(h,f)))
                    else:
                        for j,b in incoming[e[1]]:
                            if j>=g:break
                            check(((j,b),(g,e),(h,f)))
                            if not required:break
                    if not required:break
(out/'CASE_COVERAGE.json').write_text(json.dumps({c:'found' if c in found else 'absent_after_check' for c in sorted(categories)},indent=2))
with (out/'EXAMPLES.jsonl').open('w') as stream:
    for r in found.values():stream.write(json.dumps(r,ensure_ascii=False)+'\n')
