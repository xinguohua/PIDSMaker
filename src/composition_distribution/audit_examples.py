"""Check full example membership and read each event back from its original CDM bytes."""
import argparse
import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import duckdb


def original_field(event, field):
    value = event.get(field)
    if isinstance(value, dict):
        return next(iter(value.values()))
    return value


def tuples(rows, columns):
    return Counter(tuple(row[name] for name in columns) for row in rows)


def audit(run):
    result_file = run / 'EXAMPLE_SOURCE_AUDIT.csv'
    cases = [json.loads(line) for line in (run / 'COMPOSITION_EXAMPLES.jsonl').read_text().splitlines()]
    if not cases:
        raise ValueError('No completed examples')
    source = {s['shard']: Path(s['path']) for s in json.loads((run / 'stage/source.json').read_text())}
    results = []
    with ExitStack() as stack:
        db = stack.enter_context(duckdb.connect())
        payload = str(run / 'RELATION_INDEX/event_payload.parquet')
        owner = str(run / 'EVENT_OWNERSHIP/event_units.parquet')
        nodes = str(run / 'RELATION_INDEX/nodes.parquet')
        procs = str(run / 'RELATION_INDEX/process_relations.parquet')
        resources = str(run / 'RELATION_INDEX/resource_incidence.parquet')
        files = {}
        for index, case in enumerate(cases, 1):
            cid = case['composition_id']
            assignments = run / 'COMPOSITION_ASSIGNMENTS' / f"{case['order']}_{case['hop']}.csv"
            members = [r[0] for r in db.execute(
                "select unit_id from read_csv_auto(?) where composition_id=?", [str(assignments),cid]).fetchall()]
            if Counter(members) != Counter(u['uid'] for u in case['units']):
                raise ValueError((index, 'incomplete or repeated members'))
            ownership = db.execute(
                'select eid, owner from read_parquet(?) where owner in (select unnest(?))',
                [owner, members]).fetchall()
            events = case['events']
            if Counter(ownership) != tuples(events, ['eid', 'owner']):
                raise ValueError((index, 'incomplete or repeated event ownership'))
            proc = db.execute('''select eid,src_uid,dst_uid,op,ts from read_parquet(?)
                where src_uid in (select unnest(?)) and dst_uid in (select unnest(?))''',
                [procs, members, members]).fetchall()
            incidence = db.execute('select uid,resource,segment from read_parquet(?) where uid in (select unnest(?))',
                                   [resources, members]).fetchall()
            if Counter(proc) != tuples(case['process_links'], ['eid','src_uid','dst_uid','op','ts']):
                raise ValueError((index, 'incomplete internal process relations'))
            if Counter(incidence) != tuples(case['resource_incidence'], ['uid','resource','segment']):
                raise ValueError((index, 'incomplete resource incidences'))
            if {r[1] for r in incidence} != {r['nid'] for r in case['resources']}:
                raise ValueError((index, 'incomplete resource inventory'))
            physical_ids = [e['eid'] for e in events]
            cur = db.execute('''select eid,uuid,ts,op,host,subject,object1,object2,byte_offset,byte_length,path1,path2,direction
                from read_parquet(?) where eid in (select unnest(?))''',[payload, physical_ids])
            cols = [c[0] for c in cur.description]
            physical = {r[0]:dict(zip(cols,r)) for r in cur.fetchall()}
            endpoint_ids = {e[col] for e in events for col in ('subject','object1','object2') if e[col] is not None}
            ids = {nid:(host,uuid) for nid,host,uuid in db.execute(
                'select nid,host,uuid from read_parquet(?) where nid in (select unnest(?))',[nodes,list(endpoint_ids)]).fetchall()}
            for event in events:
                proj = physical[event['eid']]
                if any(proj[c] != event[c] for c in cols if c != 'host'):
                    raise ValueError((index,event['eid'],'case differs from reconstruction'))
                shard, line = divmod(event['eid'],10_000_000)
                if line != event['source_line'] or source[shard] != Path(event['source_file']):
                    raise ValueError((index,event['eid'],'source locator'))
                if shard not in files:
                    files[shard] = stack.enter_context(source[shard].open('rb'))
                stream = files[shard]
                offset, length = event['byte_offset'], event['byte_length']
                stream.seek(max(0, offset-1))
                if offset and stream.read(1) != b'\n':
                    raise ValueError((index,event['eid'],'not a line boundary'))
                raw_line = stream.readline()
                if len(raw_line) != length:
                    raise ValueError((index,event['eid'],'wrong byte length'))
                raw = json.loads(raw_line)['datum']['com.bbn.tc.schema.avro.cdm18.Event']
                expected = [event['uuid'],event['ts'],proj['host'],event['op'],event['path1'],event['path2']]
                native = [original_field(raw,k) for k in ('uuid','timestampNanos','hostId','type','predicateObjectPath','predicateObject2Path')]
                if native != expected:
                    raise ValueError((index,event['eid'],'original identity, operation, time, host or paths'))
                for col,key in [('subject','subject'),('object1','predicateObject'),('object2','predicateObject2')]:
                    value = original_field(raw,key)
                    expected = (raw['hostId'],value) if value is not None else None
                    if ids.get(event[col]) != expected:
                        raise ValueError((index,event['eid'],col))
            results.append(dict(example=index,order=case['order'],hop=case['hop'],composition_id=cid,
                complete_units=len(members),complete_owned_events=len(ownership),source_events_verified=len(events),
                internal_process_relations_verified=len(proc),resource_incidences_verified=len(incidence),
                source_fields='uuid;host;timestampNanos;type;subject;predicateObject;predicateObject2;predicateObjectPath;predicateObject2Path',
                mismatches=0))
    with result_file.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=results[0]);writer.writeheader();writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    audit(parser.parse_args().run)
