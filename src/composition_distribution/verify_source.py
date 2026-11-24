"""Independently compare each training event and every physical declaration to CDM18."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import orjson
import pyarrow.parquet as pq


def projected(path, fields):
    for batch in pq.ParquetFile(path).iter_batches(columns=fields,batch_size=8192):
        yield from zip(*(column.to_pylist() for column in batch.columns))


def uid(record, name):
    wrapped=record.get(name)
    if wrapped is None or isinstance(wrapped,str):
        return wrapped
    return wrapped['com.bbn.tc.schema.avro.cdm18.UUID']


def verify_one(task):
    manifest, directory, days = task
    idx=manifest['shard']
    intervals=[(key,lo,hi) for key,(lo,hi) in days.items()]
    event_fields='eid byte_offset byte_length uuid ts host subject_uuid object1_uuid object2_uuid op'.split()
    events=projected(directory/f'events{idx:02}.parquet',event_fields)
    declarations=projected(directory/f'decls{idx:02}.parquet',['did','byte_offset','byte_length','uuid','kind','payload'])
    counts=Counter(); classes=Counter(); operations=Counter(); digest=sha256();offset=0
    with Path(manifest['path']).open('rb',buffering=1<<20) as stream:
        for line,original in enumerate(stream,1):
            digest.update(original)
            datum=orjson.loads(original)['datum']
            if len(datum)!=1:raise ValueError((idx,line,'unexpected datum'))
            cls,record=next(iter(datum.items()))
            cls=cls.rsplit('.',1)[-1]
            identity=idx*10_000_000+line
            if cls=='Event':
                ts=record['timestampNanos']
                day=next((d for d,lo,hi in intervals if lo<=ts<hi),None)
                if day:
                    expected=(identity,offset,len(original),record['uuid'],ts,record['hostId'],uid(record,'subject'),uid(record,'predicateObject'),uid(record,'predicateObject2'),record['type'])
                    if next(events,None)!=expected:raise ValueError((idx,line,'training event differs'))
                    counts[day]+=1;operations[record['type']]+=1
            else:
                expected=(identity,offset,len(original),record['uuid'],cls,original.rstrip(b'\r\n'))
                if next(declarations,None)!=expected:raise ValueError((idx,line,'declaration differs'))
                classes[cls]+=1
            offset+=len(original)
    if next(events,None) or next(declarations,None) or offset!=manifest['bytes'] or digest.hexdigest()!=manifest['sha256']:
        raise ValueError((idx,'unconsumed projections or changed raw file'))
    return dict(file=Path(manifest['path']).name,sha256=digest.hexdigest(),training_dates=dict(counts),
                operations=dict(operations),declarations_by_kind=dict(classes))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('projection',type=Path);ap.add_argument('velox_manifest',type=Path)
    ap.add_argument('output',type=Path);ap.add_argument('--workers',type=int,default=5)
    a=ap.parse_args()
    config=json.loads(a.velox_manifest.read_text())['protocol']
    dates=config['train_dates']
    assert len(set(dates))==len(dates) and not set(dates)&set(config['val_dates']+config['test_dates'])
    tz=ZoneInfo(config['timezone'])
    def midnight(d):
        return int(datetime.combine(d,time.min,tzinfo=tz).timestamp())*1_000_000_000
    days={s:(midnight(date.fromisoformat(s)),midnight(date.fromisoformat(s)+timedelta(days=1))) for s in dates}
    manifest=json.loads((a.projection/'source.json').read_text())
    with ProcessPoolExecutor(a.workers) as jobs:
        results=list(jobs.map(verify_one,[(m,a.projection,days) for m in manifest]))
    daily=Counter();decls=Counter();ops=Counter()
    for r in results:
        daily.update(r['training_dates']);decls.update(r['declarations_by_kind']);ops.update(r['operations'])
    report=dict(passed=True,dates=dict(sorted(daily.items())),events=sum(daily.values()),operations=dict(sorted(ops.items())),
                declarations_all_shards=dict(sorted(decls.items())),files=results)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified physical training events:',report['events'],flush=True)

if __name__=='__main__':
    main()
