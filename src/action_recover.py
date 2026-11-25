"""Resume CDM provenance recovery when UUIDs are reused in different source records."""
import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import pickle
import sqlite3
import time
from collections import Counter,defaultdict
from pathlib import Path

import orjson
from action_composition import dump,present,unbox,semantic_complete

REVERSE={'EVENT_EXECUTE','EVENT_LSEEK','EVENT_MMAP','EVENT_OPEN','EVENT_ACCEPT','EVENT_READ',
         'EVENT_RECVFROM','EVENT_RECVMSG','EVENT_READ_SOCKET_PARAMS','EVENT_CHECK_FILE_ATTRIBUTES','READ'}
CONTEXT={}

def attributes():
    return dict(paths=set(),cmds=set(),subtypes=set(),evidence=[])

def scan(item):
    path,old_count=item
    root=CONTEXT['root']
    tag=root/path.name
    marker=tag.with_name(tag.name+'.done.json')
    if marker.exists():return json.loads(marker.read_text())
    start=time.monotonic()
    old=old_count is not None
    attrs={}
    staged=Path(str(tag)+f'.{os.getpid()}.sqlite')
    con=sqlite3.connect(staged)
    con.execute('create table raw(uuid,ns integer,seq integer,actor,obj,op,line integer,offset integer)')
    con.execute('pragma synchronous=off')
    batch=[];offset=0;kept=0;old_candidates=0
    digest=hashlib.sha256()
    with path.open('rb',buffering=8<<20) as f:
        for line_no,line in enumerate(f,1):
            digest.update(line);here=offset;offset+=len(line)
            record=orjson.loads(line)['datum'];typ,r=next(iter(record.items()));uid=r.get('uuid')
            if typ.endswith('.Event'):
                for field in ('predicateObject','predicateObject2'):
                    entity=unbox(r.get(field));value=unbox(r.get(field+'Path'))
                    if entity in CONTEXT['nodes'] and present(value):
                        at=attrs.get(entity)
                        if at is None:at=attrs[entity]=attributes()
                        if value not in at['paths']:
                            at['paths'].add(value);at['evidence'].append([str(path),line_no,here,field+'Path',value])
                if uid not in CONTEXT['wanted']:continue
                old_candidates+=1
                ns=r['timestampNanos']
                if ns not in CONTEXT['times']:continue
                actor=unbox(r.get('subject'));obj=unbox(r.get('predicateObject'))
                ex=r.get('properties',{}).get('map',{}).get('exec')
                if actor in CONTEXT['nodes'] and present(ex):
                    at=attrs.get(actor)
                    if at is None:at=attrs[actor]=attributes()
                    at['cmds'].add(ex)
                if not old:
                    batch.append((uid,ns,unbox(r.get('sequence')),actor,obj,r['type'],line_no,here))
                    if len(batch)>=20000:
                        con.executemany('insert into raw values(?,?,?,?,?,?,?,?)',batch);batch.clear()
                kept+=1
            elif uid in CONTEXT['nodes']:
                at=attrs.get(uid)
                if at is None:at=attrs[uid]=attributes()
                at['subtypes'].add(r.get('type',''))
                cmd=unbox(r.get('cmdLine'))
                if present(cmd):at['cmds'].add(cmd)
                for props in (r.get('properties',{}).get('map',{}),r.get('baseObject',{}).get('properties',{}).get('map',{})):
                    for key in ('path','filename'):
                        value=props.get(key)
                        if present(value):at['paths'].add(value)
                if not at['evidence']:at['evidence'].append([str(path),line_no,here,'entity_record'])
    if old:assert old_count==old_candidates,(path,old_count,old_candidates)
    con.executemany('insert into raw values(?,?,?,?,?,?,?,?)',batch);con.commit();con.close()
    staged.replace(Path(str(tag)+'.sqlite'))
    with Path(str(tag)+'.attrs.pkl').open('wb') as fp:pickle.dump(attrs,fp,protocol=5)
    data=dict(path=str(path),bytes=offset,sha256=digest.hexdigest(),native_timestamp_candidates=kept,
              old_uuid_candidates=old_candidates,reused_previous_events=old,seconds=round(time.monotonic()-start,2))
    dump(marker,data)
    return data

def recover(a):
    shards=a.root/'source_shards';shards.mkdir(exist_ok=True)
    db=sqlite3.connect('file:'+str(a.root/'native.sqlite')+'?mode=ro',uri=True)
    known={p:n for p,n in db.execute('select file,count(*) from raw_events group by file')}
    nodes={u:(n,k) for n,k,u in db.execute('select id,kind,uuid from nodes')}
    CONTEXT.update(root=shards,nodes=nodes,wanted={r[0] for r in db.execute('select distinct uuid from edges')},
                   times={r[0] for r in db.execute('select distinct ns from edges')})
    jobs=[(p,known.get(str(p))) for p in sorted(a.raw.glob('ta1-cadets-e3-official*.json*'))]
    assert len(jobs)==10
    results=[]
    with multiprocessing.get_context('fork').Pool(a.workers) as pool:
        for r in pool.imap_unordered(scan,jobs):
            print('verified shard',Path(r['path']).name,r['seconds'],r['reused_previous_events'],flush=True)
            results.append(r)
    CONTEXT.clear()
    target=a.root/'semantic.sqlite'
    out=sqlite3.connect(target.as_uri(),uri=True)
    out.execute('pragma cache_size=-262144')
    out.execute('pragma mmap_size=8589934592')
    out.execute('pragma temp_store=memory')
    out.execute("attach database ? as native",('file:'+str(a.root/'native.sqlite')+'?mode=ro',))
    out.executescript("""create table if not exists extra(uuid,ns integer,seq integer,actor,obj,op,file,line integer,offset integer);
        create table if not exists loaded(path primary key);
        create table if not exists attributes(id integer primary key,semantic,complete integer,evidence);
        create table if not exists source(w integer,row integer,seq integer,actor,obj,file,line integer,offset integer,primary key(w,row)) without rowid;
        create table if not exists audited(w integer primary key,rows integer,duplicates integer);""")
    attrs={}
    index=sqlite3.connect('file:'+str(a.events)+'?mode=ro',uri=True)
    for n,k,path,cmd,u in index.execute('select * from nodes'):
        if u not in nodes:continue
        at=attrs[u]=attributes()
        if present(path):at['paths'].add(path)
        if present(cmd):at['cmds'].add(cmd)
    for r in sorted(results,key=lambda x:x['path']):
        p=Path(r['path']);base=shards/p.name
        with Path(str(base)+'.attrs.pkl').open('rb') as f:local=pickle.load(f)
        for u,v in local.items():
            at=attrs.get(u)
            if at is None:at=attrs[u]=attributes()
            for key in ('paths','cmds','subtypes'):at[key].update(v[key])
            at['evidence'].extend(v['evidence'])
        del local
        if r['reused_previous_events'] or out.execute('select 1 from loaded where path=?',(str(p),)).fetchone():continue
        out.execute('attach database ? as shard',(str(base)+'.sqlite',))
        with out:
            out.execute('insert into extra select uuid,ns,seq,actor,obj,op,?,line,offset from shard.raw',(str(p),))
            out.execute('insert into loaded values(?)',(str(p),))
        out.execute('detach shard')
    out.execute('create index if not exists lookup_extra on extra(uuid,ns,op)')
    if not out.execute('select count(*) from attributes').fetchone()[0]:
        with out:
            for u,(n,k) in nodes.items():
                at=attrs.get(u) or attributes()
                sem=dict(kind=k,paths=sorted(at['paths']),commands=sorted(at['cmds']),subtypes=sorted(filter(None,at['subtypes'])))
                complete=semantic_complete(sem)
                out.execute('insert into attributes values(?,?,?,?)',(n,json.dumps(sem,sort_keys=True,ensure_ascii=False),int(complete),json.dumps(at['evidence'],ensure_ascii=False)))
    del attrs
    print('matching by timestamp, UUID, operation and directed endpoint identities',flush=True)
    unresolved=[]
    for w,expected in db.execute('select w,n from windows order by w'):
        if out.execute('select 1 from audited where w=?',(w,)).fetchone():continue
        matched=defaultdict(list)
        for table in ('native.raw_events','extra'):
            query=f"""select e.row,e.u,e.v,e.op,r.seq,r.actor,r.obj,r.file,r.line,r.offset
                from native.edges e join {table} r on e.uuid=r.uuid and e.ns=r.ns and e.op=r.op where e.w=?"""
            for row,u,v,op,seq,actor,obj,path,line,offset in out.execute(query,(w,)):
                endpoints=(nodes.get(actor,(None,))[0],nodes.get(obj,(None,))[0])
                if endpoints[::-1 if op in REVERSE else 1]!=(u,v):continue
                matched[row].append((seq,actor,obj,path,line,offset))
        records=[];dup=0
        for row in range(expected):
            candidates=matched.get(row,[])
            if not candidates or len({r[0] for r in candidates})!=1 or not isinstance(candidates[0][0],int):
                unresolved.append((w,row,len(candidates),[r[0] for r in candidates]));continue
            records.append((w,row,*min(candidates,key=lambda r:(r[3],r[4]))))
            dup+=len(candidates)-1
        if len(records)==expected:
            with out:
                out.executemany('insert into source values(?,?,?,?,?,?,?,?)',records)
                out.execute('insert into audited values(?,?,?)',(w,expected,dup))
        if w%100==0:print('audit',w,'unresolved',len(unresolved),flush=True)
    result=dict(files=sorted(results,key=lambda r:r['path']),edges=sum(n for n, in db.execute('select n from windows')),
                mapped=out.execute('select count(*) from source').fetchone()[0],
                duplicate_equivalent_sources=out.execute('select sum(duplicates) from audited').fetchone()[0],
                mismatches={'unresolved':len(unresolved)} if unresolved else {},
                semantic_coverage=out.execute('select n.kind,a.complete,count(*) from attributes a join native.nodes n using(id) group by n.kind,a.complete').fetchall())
    dump(a.root/'RAW_AUDIT.json',result)
    if unresolved:
        dump(a.root/'SOURCE_UNRESOLVED.json',unresolved)
        raise ValueError(('unresolved source records',len(unresolved),unresolved[:3]))
    assert result['edges']==result['mapped']
    print('source audit complete',result['mapped'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','events','raw'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    recover(p.parse_args())
