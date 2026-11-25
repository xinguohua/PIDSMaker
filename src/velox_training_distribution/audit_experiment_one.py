"""Replay revised joint records against immutable native events; emit small deliverables."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
import duckdb
from connectable import output,SOURCE

p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--source',type=Path,default=Path(SOURCE));args=p.parse_args()
r=args.results;out=Path(os.environ['RUN_OUTPUT_DIR'])
assert json.loads((r/'summary.json').read_text())['status']=='complete'
d=duckdb.connect();d.execute(f"attach '{args.source}/native_units.duckdb' as g (read_only)")
d.execute(f"create temp table evidence as select eid from read_csv_auto('{r}/NATIVE_EVENT_EVIDENCE.csv')")
links=defaultdict(set)
for e,u,v in d.execute('''select eid,uid,-1-resource from g.resource_events semi join evidence using(eid)
    union all select eid,src_uid,dst_uid from g.proc_edges semi join evidence using(eid) where op='EVENT_FORK' ''').fetchall():
    links[e].add(tuple(sorted((u,v))))
owner=dict(d.execute('select eid,owner from g.events semi join evidence using(eid)').fetchall())
windows=dict(d.execute('select uid,segment from g.units').fetchall())
counts=Counter();anchors=defaultdict(set);keys=defaultdict(set);examples={};checked=0;used=set()

def check(hit):
    global checked
    w=hit['window'];a=tuple(hit['A_uids']);b=tuple(hit['B_uids']);path=hit['path'];bundles=hit['event_bundles']
    assert a and b and not set(a)&set(b)
    assert all(windows[u]==w for u in (*a,*b))
    assert path[0] in a and path[-1] in b and not set(path[1:-1])&(set(a)|set(b))
    assert len(path)==len(set(path)) and len(bundles)==len(path)-1
    events=set()
    for (u,v),es in zip(zip(path,path[1:]),bundles):
        assert es
        for e in es:
            assert e>>32==w and tuple(sorted((u,v))) in links[e]
            events.add(e)
    assert tuple(sorted(e for e in events if owner[e] not in set(a)|set(b)))==tuple(hit['extra_events'])
    used.update(events);checked+=1
    return (w,a,b,tuple(hit['extra_events']))

with (r/'JOINT_MATCHES.jsonl').open() as f:
    for line in f:
        hit=json.loads(line);key=(hit['hop'],hit['A'],hit['B']);identity=check(hit)
        assert identity not in keys[key]
        keys[key].add(identity);counts[key]+=1;anchors[key].add(identity[:3])
        examples.setdefault(key,hit)
for row in csv.DictReader((r/'PAIR_COUNTS.csv').open()):
    key=(int(row['hop']),row['A'],row['B'])
    assert counts[key]==int(row['joint_frequency']) and row['status']=='complete'
output(out/'JOINT_COUNT_AUDIT.csv','hop A B complete_event_realizations distinct_behavior_instance_pairs',[(int(row['hop']),row['A'],row['B'],int(row['joint_frequency']),len(anchors[int(row['hop']),row['A'],row['B']])) for row in csv.DictReader((r/'PAIR_COUNTS.csv').open())])
with (out/'MATCHED_EXAMPLES.jsonl').open('w') as f:
    for key,hit in sorted(examples.items()):f.write(json.dumps(hit)+'\n')
pc=json.loads((r/'POSITIVE_CONTROL.json').read_text());controls=[]
for c in pc['controls']:
    for hit in c['matches']:check(hit)
    assert c['actual_frequency']==len(c['matches'])>0
    controls.append(dict(kind=c['kind'],known_uids=c['known_uids'],actual_frequency=c['actual_frequency'],example=c['matches'][0]))
(out/'POSITIVE_CONTROL_SUMMARY.json').write_text(json.dumps(dict(passed=True,controls=controls),indent=2))
# Bundle all owned A/B events for compact representatives; large traces remain remote.
samples=list(examples.values())+[c['example'] for c in controls]
member_ids=sorted({u for hit in samples for u in hit['A_uids']+hit['B_uids']})
d.execute('create temp table sample_members as select unnest(?::bigint[]) as uid',[member_ids])
d.execute(f"copy (select e.* from g.events e semi join sample_members s on e.owner=s.uid order by eid) to '{out}/EXAMPLE_BEHAVIOR_EVENTS.csv' (header,delimiter ',')")
manifest=[]
for f in sorted(r.iterdir()):
    if f.is_file():
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        manifest.append(dict(file=str(f),bytes=f.stat().st_size,sha256=h.hexdigest()))
(out/'RESULT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
summary=dict(status='complete',joint_records=sum(counts.values()),positive_pairs=sum(n>0 for n in counts.values()),all_replayed_records=checked,distinct_replayed_connector_events=len(used),no_duplicate_joint_keys=True,all_path_events_match_native=True,positive_controls_passed=True,results=str(r))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
