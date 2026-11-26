"""Join frozen composition frequencies and node decisions using native event windows."""
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from action_composition import ro, sha
from action_index import ActionIndex, signature
from action_paths import pack, unpack
from action_window import read_window
from baseline_knee import knee_summary


def read_json(path):
    return json.loads(path.read_text())


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = args.compositions.parent
    results = args.compositions / 'results'
    counts = sqlite3.connect(f'file:{args.compositions}/counts.sqlite?mode=ro', uri=True)
    binding = json.loads(counts.execute('select binding from protocol').fetchone()[0])
    assert binding == read_json(args.compositions/'PROTOCOL.json')
    for name in ('action_index.py', 'action_window.py'):
        assert binding['code'][name] == sha(Path(__file__).with_name(name))
    db = ro(root/'native.sqlite')
    db.execute('attach database ? as sem', (f'file:{root}/semantic.sqlite?mode=ro&immutable=1',))
    manifest = read_json(args.cell/'manifest.json')
    report = read_json(args.predictions/'report.json')
    policy = knee_summary(report, args.predictions)
    assert report['identity']['checkpoint_sha256'] == read_json(args.cell/'checkpoint.json')['sha256']
    assert (manifest['dataset'], manifest['model'], manifest['task_id']) == (
        args.dataset, args.detector, report['identity']['task_id'])
    assert report['identity']['dataset']==args.dataset and report['identity']['model']==args.detector
    assert report['decisions_sha256']==binding['decision_hash']
    preds = {}
    native = {n:(u,g,s) for n,u,g,s in db.execute('select id,uuid,gt,score from pred')}
    with (args.predictions/'objects.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            n = r['node_id']
            assert native.pop(n) == (r['uuid'], r['gt'], r['score'])
            assert r['threshold']==policy['default_threshold'] and r['prediction']==int(r['score']>r['threshold'])
            preds[n] = r['gt'], r['prediction']
    digest = hashlib.sha256()
    for n in sorted(preds):
        digest.update(f'{n}\t{preds[n][1]}\n'.encode())
    assert not native and digest.hexdigest()==report['decisions_sha256']
    (out/'RQ1_Exp3_Operating_Point.json').write_text(json.dumps(dict(report, **{'dataset':args.dataset,'detector':args.detector,
        'source':report['source'],'paths':{k:str(v) for k,v in vars(args).items()}}),indent=2))
    attack = {n for n,(g,_) in preds.items() if g}
    by_uuid = {u:n for n,u in db.execute('select id,uuid from nodes')}
    for rel, expected in binding['ground_truth'].items():
        gt = args.cell.parents[1]/'code/Ground_Truth/orthrus'/rel
        assert sha(gt)==expected
        for line in gt.read_text().splitlines():
            attack.update(by_uuid[u] for u in line.replace('"','').strip().split(',') if u in by_uuid)
    del by_uuid
    attrs = {n:(signature(s),) for n,s in db.execute('select id,semantic from sem.attributes')}
    csv.field_size_limit(sys.maxsize)
    keys, frequency, missing = {}, {}, set()
    with (results/'ACTION_COUNTS.csv').open() as f:
        for r in csv.DictReader(f):
            a = int(r['action_id'])
            keys[r['source_semantics'],r['operation'],r['destination_semantics']] = a
            frequency[a] = int(r['test_frequency'])
            if r['semantic_complete']=='0':
                missing.add(a)
    windows = [w for w, in db.execute("select w from windows where split='test' order by w")]
    assert set(windows) <= {w for w, in counts.execute('select w from window_done')}
    flags, incomplete, anchored = {}, {}, defaultdict(list)
    for k in args.lengths:
        old = dict(counts.execute('select id,bits from node_flags where k=?',(k,)))
        flags[k] = {n:int(bool(b&1)) | (2 if b&6 else 0) for n,b in old.items()}
        incomplete[k] = {n for n,b in old.items() if b&8}
        for blob, seen in counts.execute("select seq,train!='0' from counts where k=? and test!='0'",(k,)):
            seq = unpack(blob)
            if not missing.isdisjoint(seq):
                anchored[min(seq,key=frequency.__getitem__)].append((seq,1 if seen else 2))
    print('subset types',sum(map(len,anchored.values())),flush=True)
    recovered = defaultdict(set)
    def groups(w):
        return read_window(db,w,True,attrs,keys,preds,attack)[0]
    for i,w in enumerate(windows):
        idx = ActionIndex(groups(w))
        present = set(idx.variants)
        for anchor in present.intersection(anchored):
            for seq,bit in anchored[anchor]:
                if all(a in present for a in seq):
                    k=len(seq)
                    for variant,_,_ in idx.bounds(seq):
                        for n in idx.ends[variant]:
                            recovered[k].add(n)
                            flags[k][n] |= bit
        if i%40==0:
            print('membership',i+1,len(windows),flush=True)
    for k in args.lengths:
        assert recovered[k]==incomplete[k] and all(flags[k].values())
        assert all(preds[n][0]==0 for n in flags[k])
    node_rows = []
    labels = ['Entire','Seen','Zero-shot','Seen-only','Zero-shot-only','Both']
    sets = [{1,2,3},{1,3},{2,3},{1},{2},{3}]
    for k, table in flags.items():
        for label,mask in zip(labels,sets):
            nodes = [n for n,b in table.items() if b in mask]
            fp = sum(preds[n][1] for n in nodes)
            node_rows.append(dict(dataset=args.dataset,detector=args.detector,length=k,group=label,
                                  nodes=len(nodes),FP=fp,TN=len(nodes)-fp,FPR_pct=100*fp/len(nodes) if nodes else 'N/A'))
    def save(name,rows):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save('RQ1_Exp3_Node_Group_Statistics.csv',node_rows)
    # For a node stratum count every instance touching it once. Subtract paths
    # wholly inside the complement via compressed within-window counting.
    scratch=sqlite3.connect(Path(os.environ.get('TMPDIR', str(out.parent)))/('monochrome-'+out.parent.name+'.sqlite'))
    scratch.execute('create table mono(k integer,s blob,b integer,n integer,primary key(k,s,b)) without rowid')
    for i,w in enumerate(windows):
        batches=groups(w)
        for k,table in flags.items():
            for bit in (1,2,3):
                mono=[[e for e in batch if table.get(e[1])==table.get(e[2])==bit] for batch in batches]
                idx=ActionIndex(batch for batch in mono if batch)
                scratch.executemany('insert into mono values(?,?,?,?) on conflict(k,s,b) do update set n=n+excluded.n',
                                   ((k,pack(seq),bit,n) for seq,n in idx.sequences(k)))
        scratch.commit()
        if i%40==0:
            print('stratum support',i+1,len(windows),flush=True)
    totals=defaultdict(Counter)
    for k in args.lengths:
        stream=iter(scratch.execute('select s,b,n from mono where k=? order by s,b',(k,)))
        cur=next(stream,None)
        for s,seen,n in counts.execute("select seq,train!='0',cast(test as integer) from counts where k=? and test!='0' order by seq",(k,)):
            m=[0]*4
            while cur and cur[0]==s:
                m[cur[1]]=cur[2]
                cur=next(stream,None)
            assert not cur or cur[0]>s
            assert sum(m)<=n and (not m[2] if seen else not m[1])
            for label,num in zip(labels,(n,n if seen else 0,0 if seen else n,n-m[3] if seen else 0,0 if seen else n-m[3],n-m[1]-m[2])):
                if num:
                    totals[k,label]['composition_types']+=1
                    totals[k,label]['composition_instances']+=num
        assert not cur
    for r in node_rows:
        r.update({field:totals[r['length'],r['group']][field] for field in ('composition_instances','composition_types')})
    save('RQ1_Exp3_Detailed_Results.csv',node_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('compositions','cell','predictions'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(os.environ.get('RUN_OUTPUT_DIR','.')))
    p.add_argument('--dataset',required=True)
    p.add_argument('--detector',required=True)
    p.add_argument('--lengths',type=int,nargs='+',choices=(2,3),default=[2,3])
    run(p.parse_args())
