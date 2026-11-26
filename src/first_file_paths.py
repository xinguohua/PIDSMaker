"""Build a static path map using only UUID-verified evidence at the first indexed use."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import os
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo


def ro(path):
    return sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)


def build(events, native, semantic, manifest, out):
    db = ro(events)
    nodes = dict(db.execute("""select id,uuid from nodes where kind='file'
        and lower(trim(coalesce(path,''))) in ('','none','null')"""))
    first = {}
    for u,v,ns,uid in db.execute('select src,dst,ns,uuid from events'):
        for node in (u,v):
            if node in nodes:
                old = first.get(node)
                if old is None or ns < old[0]:
                    first[node] = ns,uid
                elif ns == old[0] and uid != old[1]:
                    first[node] = ns,None
    protocol = manifest['protocol']
    zone = ZoneInfo(protocol['timezone'])
    train_dates = set(protocol['train_dates'])
    native_db = ro(native)
    native_db.execute('attach database ? as sem',(f'file:{semantic.resolve()}?mode=ro',))
    lookup = {}
    for w, in native_db.execute("select w from windows where split='train'"):
        for u,v,ns,uid,f,offset in native_db.execute("""select e.u,e.v,e.ns,e.uuid,s.file,s.offset
                from edges e join sem.source s using(w,row) where e.w=? and e.y=0""",(w,)):
            for node in (u,v):
                if first.get(node)==(ns,uid):
                    lookup[node] = (f,offset,ns,uid)
    paths = {}
    audit = Counter()
    with ExitStack() as stack:
        sources = {}
        for node,(f,offset,ns,uid) in sorted(lookup.items(),key=lambda item:item[1][:2]):
            if f not in sources:
                sources[f] = stack.enter_context(open(f,'rb'))
            stream = sources[f]
            stream.seek(offset)
            event = json.loads(stream.readline())['datum']['com.bbn.tc.schema.avro.cdm18.Event']
            if event['uuid']!=uid or event['timestampNanos']!=ns or datetime.fromtimestamp(ns//10**9,zone).date().isoformat() not in train_dates:
                raise ValueError('First-use source differs from training manifest or native edge')
            for obj,path in [('predicateObject','predicateObjectPath'),('predicateObject2','predicateObject2Path')]:
                ident=event.get(obj)
                if isinstance(ident,dict): ident=next(iter(ident.values()))
                value=event.get(path)
                if isinstance(value,dict): value=value.get('string')
                if ident==nodes[node] and isinstance(value,str) and value.strip().lower() not in ('','none','null'):
                    if ident in paths and paths[ident]!=value:
                        raise ValueError('Ambiguous same-event path for '+ident)
                    paths[ident]=value
            audit['resolved' if nodes[node] in paths else 'missing_at_first_use']+=1
    payload=json.dumps(dict(version=1,dataset=manifest['dataset'],train_dates=sorted(train_dates),paths=paths),sort_keys=True).encode()
    out.mkdir(parents=True,exist_ok=True)
    path=out/(hashlib.sha256(payload).hexdigest()+'.json')
    with NamedTemporaryFile(dir=out,delete=False) as f:
        f.write(payload)
    os.replace(f.name,path)
    from pidsmaker.preprocessing.file_paths import install_evidence
    install_evidence(path,manifest['dataset'],train_dates)
    summary=dict(missing_file_nodes=len(nodes),unique_first_training_rows=len(lookup),counts=audit,evidence=str(path),
                 source_run=manifest['task_id'],source_commit=manifest['commit'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('events','native','semantic','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    build(a.events,a.native,a.semantic,json.loads(a.manifest.read_text()),a.output)
