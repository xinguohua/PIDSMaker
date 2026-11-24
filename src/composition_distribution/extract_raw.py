"""Project physical CDM18 event occurrences and timestamp provenance from the frozen shards."""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

TZ = ZoneInfo('America/New_York')
BOUNDS = [(int(datetime(2018,4,a,tzinfo=TZ).timestamp())*10**9,
           int(datetime(2018,4,b,tzinfo=TZ).timestamp())*10**9) for a,b in [(2,6),(7,10)]]
TAG = re.compile(rb'cdm18\.([A-Za-z0-9_]+)"')
TS = re.compile(rb'"timestampNanos":(-?\d+)')
EVENT = pa.schema([(k,pa.int64()) for k in ['eid','byte_offset','byte_length','ts','seq']] +
                  [(k,pa.string()) for k in ['uuid','host','subject_uuid','object1_uuid','object2_uuid','op','path1','path2','exec_name','retval','event_name']] +
                  [('block',pa.int8())])
DECL = pa.schema([(k,pa.int64()) for k in ['did','byte_offset','byte_length','start_ns','prev_ts','next_ts']] +
                 [(k,pa.string()) for k in ['uuid','host','kind','subtype','pid','cmd','role','parent_uuid','principal_uuid','source_uuid','sink_uuid','prev_host','next_host']] +
                 [('payload',pa.binary())])

def unbox(v):
    if isinstance(v,dict) and len(v)==1:return next(iter(v.values()))
    return v

def block(t):
    for i,(lo,hi) in enumerate(BOUNDS,1):
        if lo<=t<hi:return i
    return 0

class Sink:
    def __init__(self,p,schema):
        self.path=p; self.schema=schema;self.rows=[]
        self.w=pq.ParquetWriter(str(p)+'.partial',schema,compression='zstd')
    def add(self,row):
        self.rows.append(row)
        if len(self.rows)>=30000:self.flush()
    def flush(self):
        if self.rows:
            self.w.write_table(pa.Table.from_pylist(self.rows,schema=self.schema))
            self.rows.clear()
    def close(self):
        self.flush();self.w.close();Path(str(self.path)+'.partial').replace(self.path)

def scan(task):
    idx,path,out,expected=task
    marker=out/f'shard{idx:02}.json'
    if marker.exists():
        info=json.loads(marker.read_text());assert info['sha256']==expected
        return info
    es=Sink(out/f'events{idx:02}.parquet',EVENT)
    ds=Sink(out/f'decls{idx:02}.parquet',DECL)
    off=0; h=hashlib.sha256(); counts=[0,0,0];pending=[]
    prev_ts=prev_host=None
    host_pattern=re.compile(rb'"hostId":"([^"]+)"')
    with path.open('rb',buffering=2**20) as f:
        for lineno,line in enumerate(f,1):
            h.update(line); base=off;off+=len(line)
            m=TAG.search(line[:200]);assert m,(path,lineno)
            typ=m[1].decode()
            rid=10**7*idx+lineno
            if typ=='Event':
                match=TS.search(line)
                assert match,(path,lineno)
                ts=int(match[1]); b=block(ts); counts[b]+=1
                hm=host_pattern.search(line)
                host=hm[1].decode() if hm else None
                for record in pending:
                    record.update(next_ts=ts,next_host=host)
                    ds.add(record)
                pending.clear()
                prev_ts,prev_host=ts,host
                if not b:continue
                q=next(iter(orjson.loads(line)['datum'].values()))
                assert q['timestampNanos']==ts and q['hostId']==host
                props=unbox(q.get('properties')) or {}
                es.add(dict(eid=rid,byte_offset=base,byte_length=len(line),ts=ts,seq=unbox(q.get('sequence')),
                            uuid=q['uuid'],host=host,subject_uuid=unbox(q.get('subject')),object1_uuid=unbox(q.get('predicateObject')),
                            object2_uuid=unbox(q.get('predicateObject2')),op=q['type'],path1=unbox(q.get('predicateObjectPath')),
                            path2=unbox(q.get('predicateObject2Path')),exec_name=props.get('exec'),retval=props.get('return_value'),
                            event_name=unbox(q.get('name')),block=b))
            else:
                q=next(iter(orjson.loads(line)['datum'].values()))
                host=q.get('hostId') or (q.get('baseObject') or {}).get('hostId') or (q['uuid'] if typ=='Host' else None)
                props=unbox(q.get('properties')) or {}
                role=q.get('type') or typ
                if typ=='NetFlowObject':
                    # Match the raw service role without treating ephemeral identities as types.
                    local = q.get('localAddress');remote=q.get('remoteAddress')
                    loop=lambda ip: bool(ip) and (ip=='::1' or ip.startswith('127.'))
                    role='netflow:loopback' if loop(local) or loop(remote) else 'netflow:nonloopback' if local and remote else 'netflow:unknown'
                row=dict(did=rid,byte_offset=base,byte_length=len(line),kind=typ,subtype=q.get('type'),role=role,host=host,
                         uuid=q['uuid'],pid=str(q['cid']) if q.get('cid') is not None else None,cmd=unbox(q.get('cmdLine')),
                         parent_uuid=unbox(q.get('parentSubject')),principal_uuid=unbox(q.get('localPrincipal')),
                         source_uuid=unbox(q.get('sourceUUID')),sink_uuid=unbox(q.get('sinkUUID')),
                         start_ns=q.get('startTimestampNanos'), prev_ts=prev_ts,prev_host=prev_host,payload=line.rstrip(b'\r\n'))
                pending.append(row)
    for row in pending:ds.add(row)
    assert h.hexdigest()==expected,(path,h.hexdigest(),expected)
    es.close();ds.close()
    info=dict(shard=idx,path=str(path.resolve()),bytes=off,lines=lineno,events_by_block=counts,sha256=h.hexdigest())
    marker.write_text(json.dumps(info));print(info,flush=True);return info

def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('frozen',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=9);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    files=sorted(json.loads(a.frozen.read_text()),key=lambda s:s['name']);assert len(files)==10
    tasks=[(i,a.raw/s['name'],a.out,s['sha256']) for i,s in enumerate(files)]
    with mp.Pool(a.workers) as pool:stats=pool.map(scan,tasks,chunksize=1)
    (a.out/'source.json').write_text(json.dumps(stats,indent=2))
if __name__=='__main__':main()
