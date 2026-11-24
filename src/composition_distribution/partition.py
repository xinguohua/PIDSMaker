"""Non-overlapping bounded BFS on process edges and lazy resource hyperedges."""
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import duckdb
import numpy as np


def lit(path):
    return "'" + str(path).replace("'", "''") + "'"


def write_csv(path, records):
    tmp = path.with_suffix('.partial')
    with tmp.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    tmp.replace(path)


def csr(keys, values, size):
    order = np.argsort(keys, kind='stable')
    offsets = np.empty(size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(np.bincount(keys, minlength=size))
    return offsets, values[order]


def breadth_first(seed_order, processes, resources, members, hops):
    size = len(seed_order)
    owners = np.full(size, -1, dtype=np.int64)
    depths = np.full(size, -1, dtype=np.int8)
    # Expanding a resource once claims every still-unowned member in one hop.
    # A previously claimed unit is never used to cross a partition boundary.
    expanded = np.zeros(len(members[0])-1, dtype=bool)
    for start in seed_order:
        if owners[start] != -1:
            continue
        owners[start], depths[start] = start, 0
        q = deque([(start, 0)])
        while q:
            u, d = q.popleft()
            if d == hops:
                continue
            neighbors = [processes[1][processes[0][u]:processes[0][u+1]]]
            for r in resources[1][resources[0][u]:resources[0][u+1]]:
                if not expanded[r]:
                    expanded[r] = True
                    neighbors.append(members[1][members[0][r]:members[0][r+1]])
            for batch in neighbors:
                for v in batch:
                    if owners[v] == -1:
                        owners[v], depths[v] = start, d+1
                        q.append((v, d+1))
    assert (owners >= 0).all() and (depths >= 0).all() and (depths <= hops).all()
    return owners, depths


def graph(db):
    u = db.execute('select uid from units order by start_ns asc nulls last, uid asc').fetchnumpy()['uid']
    assert len(u) and np.array_equal(np.sort(u), np.arange(len(u)))
    p = db.execute("""select distinct src_uid AS a,dst_uid AS b from proc_edges
        where src_uid<>dst_uid""").fetchnumpy()
    proc = csr(np.concatenate((p['a'],p['b'])), np.concatenate((p['b'],p['a'])), len(u))
    h = db.execute("""with shared as (
        select segment,resource,row_number() over(order by segment,resource)-1 AS rid
        from incidence group by segment,resource having count(*)>1)
        select uid,rid from incidence join shared using(segment,resource)""").fetchnumpy()
    m = int(h['rid'].max())+1 if len(h['rid']) else 0
    return u, proc, csr(h['uid'], h['rid'], len(u)), csr(h['rid'], h['uid'], m)


def copy(db, sql, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    db.execute(f"copy ({sql}) to {lit(temp)} (format parquet, compression zstd)")
    temp.replace(path)


def partition(root):
    source = root/'raw.duckdb'
    db = duckdb.connect(str(source), read_only=True)
    db.execute("set threads=8")
    out = root/'COMPOSITION_ASSIGNMENTS'
    out.mkdir(parents=True, exist_ok=True)
    order, proc, resources, members = graph(db)
    for direction, seeds in [('forward',order),('reverse',order[::-1])]:
        for h in range(1,6):
            cid, depth = breadth_first(seeds, proc, resources, members, h)
            dest = out / f'{direction}_{h}.csv'
            tmp = dest.with_suffix('.partial')
            with tmp.open('w',newline='') as stream:
                w = csv.writer(stream)
                w.writerow(['unit_id','composition_id','depth'])
                w.writerows(zip(range(len(cid)),cid,depth))
            tmp.replace(dest)
            print(direction,h,'units',len(order),'groups',len(np.unique(cid)),flush=True)
    db.close()


def create_payload(db, root):
    own = root/'EVENT_OWNERSHIP'
    index = root/'RELATION_INDEX'
    own.mkdir(exist_ok=True)
    # These three tables define unique event ownership and lossless payloads;
    # the complete graph can subsequently be read without the ingestion DB.
    for sub,name,sql in [
        (own,'event_units','select eid,owner from events'),
        (index,'event_payload','select * exclude(owner) from events'),
        (index,'nodes','select * from nodes'),
        (index,'unit_nodes','select uid,subject from units'),
        (index,'process_relations','select * from proc_edges'),
        (index,'resource_incidence','select * from incidence')]:
        copy(db,sql,sub/(name+'.parquet'))
    copy(db, """select distinct e.eid, e.owner uid, i.resource, i.segment
        from events e join incidence i on i.uid=e.owner and i.resource=e.object1
        union select e.eid,e.owner,i.resource,i.segment
        from events e join incidence i on i.uid=e.owner and i.resource=e.object2""",
         index/'event_resource_refs.parquet')


def view(db, name, path):
    db.execute(f'create or replace temp view {name} as select * from read_parquet({lit(path)})')


def scalar(db, sql):
    return int(db.execute(sql).fetchone()[0])


def audit(root):
    db = duckdb.connect()
    db.execute('set threads=8')
    db.execute("set memory_limit='28GB'")
    db.execute(f"set temp_directory={lit(root/'partition-spill')}")
    db.execute(f"attach {lit(root/'raw.duckdb')} as raw (read_only)")
    db.execute("use raw")
    create_payload(db, root)
    index = root/'RELATION_INDEX'
    view(db,'ledger',root/'EVENT_OWNERSHIP/event_units.parquet')
    view(db,'payload',index/'event_payload.parquet')
    view(db,'registry',index/'nodes.parquet')
    view(db,'p_edges',index/'process_relations.parquet')
    view(db,'r_inc',index/'resource_incidence.parquet')
    view(db,'r_refs',index/'event_resource_refs.parquet')
    db.execute('create temp view restored as select p.*,l.owner from payload p join ledger l using(eid)')
    db.execute('create temp view canonical as select * exclude(owner),owner from events')
    view(db,'unit_nodes',index/'unit_nodes.parquet')
    db.execute('''create temp table referenced_nodes as
        select subject nid from unit_nodes where subject is not null union
        select subject from payload where subject is not null union
        select object1 from payload where object1 is not null union
        select object2 from payload where object2 is not null union
        select resource from r_inc where resource is not null''')
    copy(db,'select n.nid from nodes n anti join referenced_nodes using(nid)',
         index/'unreferenced_nodes.parquet')
    view(db,'unreferenced',index/'unreferenced_nodes.parquet')
    db.execute('create temp view recovered_ids as select nid from referenced_nodes union all select nid from unreferenced')
    recovered_nodes=scalar(db,'select count(distinct nid) from recovered_ids')
    missing_node_ids=scalar(db,'select count(*) from nodes anti join recovered_ids using(nid)')
    extra_node_ids=scalar(db,'select count(*) from recovered_ids anti join nodes using(nid)')
    if missing_node_ids or extra_node_ids or recovered_nodes!=scalar(db,'select count(*) from recovered_ids'):
        raise ValueError('Node identity reconstruction from endpoints and isolated declarations failed')
    record_count = scalar(db,'select count(*) from events')
    node_count = scalar(db,'select count(*) from nodes')
    # Multiset equality on all columns handles repeated attributes, swapped
    # endpoints, reversal, lost event IDs, and changed timestamps independently.
    def compare(left, right):
        return sum(scalar(db,f'select count(*) from (select * from {a} except all select * from {b})')
                   for a,b in [(left,right),(right,left)])
    # Compare event IDs and explicit event columns by ID. Storing and re-hashing
    # 20M wide structs for EXCEPT ALL would unnecessarily exceed server memory.
    columns = [r[0] for r in db.execute('describe canonical').fetchall()]
    quoted = lambda c: '"' + c.replace('"','""') + '"'
    tests = ' OR '.join(f'a.{quoted(c)} is distinct from b.{quoted(c)}' for c in columns if c!='eid')
    unique_events = scalar(db,'select count(distinct eid) from events')
    ledger_n = scalar(db,'select count(*) from ledger')
    duplicate_event_owner = ledger_n - scalar(db,'select count(distinct eid) from ledger')
    payload_n = scalar(db,'select count(*) from payload')
    duplicate_payload = payload_n - scalar(db,'select count(distinct eid) from payload')
    if not record_count==unique_events==ledger_n==payload_n or duplicate_event_owner or duplicate_payload:
        raise ValueError('Duplicate/missing physical event IDs in event reconstruction basis')
    mismatches = scalar(db,f"""select count(*) from canonical a full join restored b using(eid)
        where a.eid is null or b.eid is null or {tests}""")
    nodediff = compare('nodes','registry')
    rel_diff = compare('proc_edges','p_edges') + compare('incidence','r_inc')
    unit_n = scalar(db,'select count(*) from units')
    orphan_e = scalar(db,'select count(*) from ledger l anti join units u on l.owner=u.uid')
    # Every eligible resource endpoint, and both when present, must be indexed.
    unmatched = scalar(db,"""select count(*) from r_inc i anti join
         (select distinct uid,resource,segment from r_refs) r using(uid,resource,segment)""")
    invalid_relations=scalar(db,'''select count(*) from p_edges p
        left join units s on p.src_uid=s.uid left join units t on p.dst_uid=t.uid
        where s.uid is null or t.uid is null or s.segment<>t.segment''')
    invalid_resources=scalar(db,'''select count(*) from r_inc p left join units u using(uid)
        where u.uid is null or u.segment<>p.segment''')
    if any((mismatches,nodediff,rel_diff,orphan_e,unmatched,invalid_relations,invalid_resources)):
        raise ValueError(dict(event_payload_or_id=mismatches,node_delta=nodediff,relation_delta=rel_diff,
                              unowned_event=orphan_e,resource_incidence_without_events=unmatched,invalid_process_pairs=invalid_relations,
                              invalid_resource_members=invalid_resources))
    db.execute("""create temp table degree as select uid,count(*) as n from (
        select src_uid uid from p_edges where src_uid<>dst_uid union all
        select dst_uid from p_edges where src_uid<>dst_uid union all
        select uid from r_inc join (select segment,resource from r_inc group by all having count(*)>1)
        using (segment,resource)) group by uid""")
    isolated = scalar(db,'select count(*) from units anti join degree using(uid)')
    source_stats=dict(original_nodes=node_count,original_events=record_count,original_units=unit_n,
                      original_process_relations=scalar(db,'select count(*) from proc_edges'),
                      resource_unit_incidences=scalar(db,'select count(*) from r_inc'),
                      different_event_id_or_fields=mismatches, registry_multiset_delta=nodediff,
                      relational_witness_multiset_delta=rel_diff, source_orphan_event_owners=orphan_e)
    (root/'partition_basis_audit.json').write_text(json.dumps(source_stats,indent=2)+'\n')
    audits=[];stats=[]
    for direction in ('forward','reverse'):
        for h in range(1,6):
            name = f'{direction}_{h}'
            assignment = root/'COMPOSITION_ASSIGNMENTS'/f'{name}.csv'
            db.execute(f"""create or replace temp table a as select unit_id uid,composition_id cid,depth
                  from read_csv({lit(assignment)}, header=true, columns={{'unit_id':'BIGINT',
                  'composition_id':'BIGINT','depth':'BIGINT'}})""")
            missing = scalar(db,'select count(*) from units anti join a using(uid)')
            extra = scalar(db,'select count(*) from a anti join units using(uid)')
            duplicate = scalar(db,'select count(*)-count(distinct uid) from a')
            invalid_root = scalar(db,f"""select count(*) from a l left join a r on l.cid=r.uid
              where r.uid is null or r.depth<>0 or r.cid<>r.uid or l.depth<0 or l.depth>{h}
              or ((l.uid=l.cid) <> (l.depth=0))""")
            lost_events = scalar(db,'select count(*) from ledger anti join a on ledger.owner=a.uid')
            if any((missing,extra,duplicate,invalid_root,lost_events)):
                raise ValueError(f'Bad partition {name}: {(missing,extra,duplicate,invalid_root,lost_events)}')
            # The owner->composition map is total and functional, so joining it
            # to the base event reconstruction preserves precisely the audited IDs.
            recovered_events = scalar(db,'select count(*) from ledger join a on ledger.owner=a.uid')
            if recovered_events != record_count:
                raise ValueError('Wrong recovered event multiplicity for '+name)
            edges_dir = root/'BOUNDARY_RELATIONS'/name
            db.execute("""create or replace temp table resource_comps as select
                segment,resource,cid,count(*)::bigint units from r_inc join a using(uid)
                group by all""")
            db.execute("""create or replace temp table shared as select segment,resource,
                count(*) memberships, sum(units) unit_count from resource_comps group by all
                having count(*)>1""")
            copy(db,'select rc.* from resource_comps rc join shared using(segment,resource)',
                 edges_dir/'shared_resource_groups.parquet')
            copy(db,"""select p.*,x.cid src_composition,y.cid dst_composition
                 from p_edges p join a x on p.src_uid=x.uid join a y on p.dst_uid=y.uid
                 where x.cid<>y.cid""",edges_dir/'process_edges.parquet')
            # Factor shared resource references. For a resource event, other
            # groups are retrieved by joining the source incidence and this
            # partition's shared_resource_groups, without materializing all pairs.
            # Subject-target events (e.g. SIGNAL) need no fabricated new relation class.
            copy(db,"""select e.eid,x.cid owner_composition,y.cid object1_composition,z.cid object2_composition
                 from payload e join ledger l using(eid) join a x on l.owner=x.uid
                 left join a y on e.object1_uid=y.uid left join a z on e.object2_uid=z.uid
                 where x.cid<>y.cid or x.cid<>z.cid""",edges_dir/'target_subjects.parquet')
            procfile=lit(edges_dir/'process_edges.parquet')
            subjfile=lit(edges_dir/'target_subjects.parquet')
            cross_e=scalar(db,f"""select count(*) from (
                select eid from r_refs join shared using(segment,resource)
                union select eid from {procfile} union select eid from {subjfile})""")
            proc_cross=scalar(db,f'select count(*) from {procfile}')
            rin=scalar(db,'select coalesce(sum(units*(units-1)//2),0) from resource_comps')
            # Per-resource unordered distinct unit pairs; parallel resources count independently.
            rout=scalar(db,"""select coalesce(sum(total*(total-1)//2-within_group),0) from
                (select sum(units)::bigint total,sum(units*(units-1)//2) within_group
                 from resource_comps group by segment,resource)""")
            groups,singles=db.execute('select count(*),count(*) filter(where size=1) from (select cid,count(*) size from a group by cid)').fetchone()
            shared_refs=scalar(db,'select coalesce(sum(memberships),0) from shared')
            audits.append(dict(order=direction,hops=h,original_nodes=node_count,recovered_unique_nodes=recovered_nodes,
                original_events=record_count,recovered_unique_events=recovered_events,
                missing_nodes=missing_node_ids,missing_events=lost_events,duplicate_event_owners=duplicate_event_owner,
                unassigned_units=missing,duplicate_unit_owners=duplicate,extra_units=extra,
                event_ids_or_field_mismatches=mismatches,cross_composition_events=cross_e,
                shared_resource_composition_refs=shared_refs,graph_isolated_units=isolated,single_unit_compositions=singles,
                node_coverage=1.0,event_coverage=1.0))
            stats.append(dict(order=direction,hops=h,units=unit_n,compositions=groups,single_unit_compositions=singles,
                multiunit_compositions=groups-singles,isolated_units=isolated,internal_process_relations=source_stats['original_process_relations']-proc_cross,
                cross_process_relations=proc_cross,internal_resource_unit_pairs=rin,cross_resource_unit_pairs=rout,
                internal_events=record_count-cross_e,cross_events=cross_e,shared_segment_resources=scalar(db,'select count(*) from shared'),
                shared_resource_composition_refs=shared_refs))
            write_csv(root/'INTEGRITY_AUDIT.csv',audits)
            write_csv(root/'PARTITION_STATISTICS.csv',stats)
            print('audited',name,groups,flush=True)
    db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('stage',choices=['partition','audit','all'],default='all',nargs='?')
    a=p.parse_args()
    if a.stage in ('partition','all'):
        partition(a.root)
    if a.stage in ('audit','all'):
        audit(a.root)
