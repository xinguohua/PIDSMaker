"""Partition the observed Velox training edges within their source graph windows."""
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


def scalar(db, sql):
    return int(db.execute(sql).fetchone()[0])



def verify_training_population(root):
    manifest=json.loads((root/'NATIVE_INPUT_AUDIT.json').read_text())
    expected={'train_run':'cadets-velox-fork-train-20261124a',
              'train_commit':'8a7303915422a85dead0a627ab1a031b7ef0c6b5','windows':581,
              'edge_occurrences':3426589,'native_loader_train_batches':3665,'logged_train_batches':3665}
    mismatched={k:(v,manifest.get(k)) for k,v in expected.items() if v!=manifest.get(k)}
    if mismatched:
        raise ValueError(f'Unexpected Velox training population: {mismatched}')
    with duckdb.connect() as conn:
        for view,src in [('e','NATIVE_EDGES/*.parquet'),('n','NATIVE_NODES/*.parquet'),
                         ('w','NATIVE_WINDOWS.csv'),('b','NATIVE_BATCHES.csv')]:
            fn='read_csv_auto' if src.endswith('csv') else 'read_parquet'
            conn.execute(f'create view {view} as select * from {fn}({lit(root/src)})')
        dims=conn.execute('''select count(*),count(distinct window_id),count(distinct batch_id),
            count(distinct (window_id,tensor_row)),count(distinct (batch_id,batch_row)) from e''').fetchone()
        assert dims==(3426589,581,3665,3426589,3426589),dims
        assert conn.execute('select count(*),count(distinct nid) from n').fetchone()==(
            manifest['within_window_node_occurrences'],manifest['distinct_node_ids'])
        assert scalar(conn,'select count(*) from w')==581 and scalar(conn,'select count(*) from b')==3665
        bad_windows=scalar(conn,'''select count(*) from w full join
            (select window_id,count(*) n,count(distinct batch_id) b from e group by window_id) g using(window_id)
            full join (select window_id,count(*) nnode from n group by window_id) ns using(window_id)
            where w.edges is distinct from g.n or w.batches is distinct from g.b
               or w.nodes is distinct from ns.nnode''')
        bad_batches=scalar(conn,'''select count(*) from b full join (
            select batch_id,window_id,min(tensor_row) lo,max(tensor_row) hi,count(*) n from e group by all
            ) g using(batch_id,window_id) where b.edges is distinct from g.n
            or b.first_tensor_row is distinct from g.lo or b.last_tensor_row is distinct from g.hi''')
        if bad_windows or bad_batches:
            raise ValueError(f'Unexpected native window or batch membership: {bad_windows}, {bad_batches}')


def native_units(root):
    """Index native graph observations, leaving the detector edge payload in place."""
    tmp = root/'native_units.building.duckdb'
    db = duckdb.connect(str(tmp))
    db.execute('set threads=6')
    db.execute(f"create or replace view native_edges as select * from read_parquet({lit(root/'NATIVE_EDGES/*.parquet')})")
    db.execute(f"create or replace view native_nodes as select * from read_parquet({lit(root/'NATIVE_NODES/*.parquet')})")
    db.execute("""create or replace table nodes as select
        (row_number() over(order by window_id,nid)-1)::bigint nid, nid native_nid, window_id, kind,
        kind AS role,label,uuid from native_nodes""")
    db.execute("""create or replace temp table endpoints as select
        e.eid,e.op,e.ts,e.window_id,s.nid s,d.nid d,s.kind='subject' sp,d.kind='subject' dp
        from native_edges e join nodes s on e.window_id=s.window_id and e.src=s.native_nid
        join nodes d on e.window_id=d.window_id and e.dst=d.native_nid""")
    # pidsmaker.utils.dataset_utils.edge_reversed changes the information-flow
    # orientation. A unique subject endpoint identifies its actor independently.
    db.execute("""create or replace temp table actors as select *,
        case when sp and not dp then true when dp and not sp then false
             when sp and dp then op not in ('EVENT_EXECUTE','EVENT_LSEEK','EVENT_MMAP',
             'EVENT_OPEN','EVENT_ACCEPT','EVENT_READ','EVENT_RECVFROM','EVENT_RECVMSG',
             'EVENT_READ_SOCKET_PARAMS','EVENT_CHECK_FILE_ATTRIBUTES','READ') end actor_at_src
        from endpoints""")
    db.execute("""create or replace temp table activity as select nid,min(ts) t0,max(ts) t1 from
        (select s nid,ts from actors where sp union all select d,ts from actors where dp) group by nid""")
    # Actual labels use "type path cmd_line". Preserve the original label and
    # take only the executable token of the path, or of cmd_line when path is absent.
    db.execute("""create or replace table units as
      with subject_nodes as (
        select n.*,regexp_extract(label,'^subject +([^ ]+)',1) path_token,
        regexp_extract(label,'^subject +[^ ]+ +([^ ]+)',1) cmd_token
        from nodes n where kind='subject'
      ), identified as (
        select n.nid subject,
        case when lower(path_token) not in ('','none','na','null','0') then path_token
             when lower(cmd_token) not in ('','none','na','null','0') then cmd_token end exec_token,
        a.t0 start_ns,a.t1 end_ns,window_id segment,null::bigint fallback_eid,native_nid,
        label native_label from subject_nodes n left join activity a using(nid)
      ), incl_unresolved as (
        select * from identified union all select null,null,ts,ts,window_id,eid,null,null
        from actors where not sp and not dp)
      select (row_number() over(order by start_ns asc nulls last, segment,subject nulls last,fallback_eid)-1)::bigint uid,
        subject,coalesce(nullif(regexp_extract(exec_token,'([^/]+)$',1),''),'UNKNOWN_PROGRAM') program,
        start_ns,end_ns,segment,fallback_eid,native_nid,native_label,
        case when subject is null then 'unresolved_edge_actor'
             when start_ns is null then 'isolated_native_subject'
             else 'observed_window_subject' end start_status from incl_unresolved""")
    db.execute("""create or replace table event_binding as
        select a.eid,coalesce(u.uid,f.uid) owner,u.subject,
        case when a.actor_at_src then d else s end object1,
        case when a.actor_at_src is null then d end object2,
        peer.uid object1_uid,null::bigint object2_uid,
        case when a.actor_at_src then 'actor_to_object' when not a.actor_at_src then 'object_to_actor'
             else 'native_src_to_dst_actor_unknown' end direction
        from actors a left join units u on u.subject=case when a.actor_at_src then s
            when not a.actor_at_src then d end
        left join units f on f.fallback_eid=a.eid
        left join units peer on peer.subject=case when a.actor_at_src then d else s end""")
    db.execute("""create or replace view events as select e.*,b.* exclude(eid)
                  from native_edges e join event_binding b using(eid)""")
    db.execute("""create or replace table proc_edges as select eid,owner src_uid,object1_uid dst_uid,ts,op
        from events where op in ('EVENT_FORK','EVENT_CLONE','EVENT_VFORK','EVENT_CREATE_THREAD')
        and subject is not null and object1_uid is not null""")
    db.execute("""create or replace table resource_events as
        select e.eid,e.owner uid,n.nid resource,n.window_id segment
        from event_binding e join nodes n on e.object1=n.nid where n.kind<>'subject'
        union select e.eid,e.owner,n.nid,n.window_id
        from event_binding e join nodes n on e.object2=n.nid where n.kind<>'subject'""")
    db.execute("""create or replace table incidence as select distinct uid,resource,segment from resource_events""")
    db.execute("""create or replace view exec_observations as select e.eid,e.owner,e.window_id,e.ts,
        e.object1,n.native_nid target_native_node,n.kind target_kind,n.label target_label
        from events e left join nodes n on e.object1=n.nid where e.op='EVENT_EXECUTE'""")
    nedges=scalar(db,'select count(*) from native_edges')
    nnodes=scalar(db,'select count(*) from native_nodes')
    if (nedges!=scalar(db,'select count(distinct eid) from native_edges')
        or nnodes!=scalar(db,'select count(distinct (window_id,native_nid)) from nodes')
        or nedges!=scalar(db,'select count(*) from events')
        or nedges!=scalar(db,'select count(distinct eid) from event_binding where owner is not null')):
        raise ValueError('Native edge, endpoint, or actor mapping is incomplete or ambiguous')
    copy(db,'select * from units order by uid',root/'EXECUTION_UNITS.parquet')
    for fname,query in {
        'process_relations':'select * from proc_edges','resource_incidence':'select * from incidence',
        'resource_events':'select * from resource_events',
        'native_node_map':'select nid,window_id,native_nid from nodes',
        'native_event_index':'select * exclude(src,dst) from native_edges',
        'unreferenced_nodes':'''select n.nid from nodes n anti join
          (select s nid from endpoints union select d from endpoints) used using(nid)''',
        'execute_observations':'select * from exec_observations',
    }.items():
        copy(db,query,root/'RELATION_INDEX'/(fname+'.parquet'))
    copy(db,'select * from event_binding',root/'EVENT_OWNERSHIP'/'event_units.parquet')
    data=dict(native_windows=scalar(db,'select count(distinct window_id) from native_edges'),
        native_edges=nedges,native_window_nodes=nnodes,
        distinct_native_node_ids=scalar(db,'select count(distinct native_nid) from nodes'),
        units=scalar(db,'select count(*) from units'),
        uncertain_actor_fallback_units=scalar(db,'select count(*) from units where subject is null'),
        unknown_program_units=scalar(db,"select count(*) from units where program='UNKNOWN_PROGRAM'"),
        candidate_execute_events=scalar(db,'select count(*) from exec_observations'),
        observed_process_links=scalar(db,'select count(*) from proc_edges'),
        unit_resource_incidence=scalar(db,'select count(*) from incidence'))
    db.close()
    tmp.replace(root/'native_units.duckdb')
    (root/'UNIT_INPUT_SUMMARY.json').write_text(json.dumps(data,indent=2)+'\n')
    print('native units',data,flush=True)


def partitions(db,root):
    seeds, processes, resources, members=graph(db)
    out=root/'COMPOSITION_ASSIGNMENTS'
    out.mkdir(exist_ok=True)
    for direction,order in [('forward',seeds),('reverse',seeds[::-1])]:
        for hops in range(1,6):
            owners,depths=breadth_first(order,processes,resources,members,hops)
            path=out/f'{direction}_{hops}.csv'
            with path.with_suffix('.partial').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['unit_id','composition_id','depth'])
                writer.writerows(zip(range(len(owners)),owners,depths))
            path.with_suffix('.partial').replace(path)
            print('assigned',direction,hops,len(np.unique(owners)),flush=True)
def audit(db,root):
    rel=root/'RELATION_INDEX'
    for name,path in [('owner_index',root/'EVENT_OWNERSHIP/event_units.parquet'),
                      ('r_inc',rel/'resource_incidence.parquet'),('r_events',rel/'resource_events.parquet'),
                      ('p_edges',rel/'process_relations.parquet'),('mapping',rel/'native_node_map.parquet'),
                      ('isolated_nodes',rel/'unreferenced_nodes.parquet'),('event_index',rel/'native_event_index.parquet')]:
        db.execute(f'create temp table {name} as select * from read_parquet({lit(path)})')
    db.execute("""create temp table restored as select e.*,s.native_nid src,d.native_nid dst,
      s.window_id src_window,d.window_id dst_window,b.owner,b.subject,b.object1,b.object2
      from event_index e join owner_index b using(eid) join mapping s on s.nid=case
      when b.direction='actor_to_object' then b.subject else b.object1 end
      join mapping d on d.nid=case b.direction when 'actor_to_object' then b.object1
      when 'object_to_actor' then b.subject else b.object2 end""")
    db.execute("""create temp table restored_nodes as select nid from isolated_nodes union all select
       distinct nid from (select subject nid from restored union all select object1 from restored
       union all select object2 from restored) where nid is not null""")
    count=scalar(db,'select count(*) from native_edges')
    nodes=scalar(db,'select count(*) from native_nodes')
    reduced=scalar(db,'select count(*) from restored')
    attrs=[row[0] for row in db.execute('describe native_edges').fetchall() if row[0]!='eid']
    differences=' or '.join(f'a."{key}" is distinct from b."{key}"' for key in attrs)
    bad=scalar(db,f"""select count(*) from native_edges a full outer join restored b using(eid)
       where a.eid is null or b.eid is null or
       {differences}
       or b.src_window<>b.window_id or b.dst_window<>b.window_id""")
    db.execute("""create temp table node_key as select m.window_id,m.native_nid nid
       from restored_nodes r join mapping m using(nid)""")
    node_multiset=scalar(db,"""select count(*) from (select *,count(*) n from node_key group by all)
      a full join (select window_id,nid,count(*) n from native_nodes group by all) b using(window_id,nid)
      where a.n is distinct from b.n""")
    dups=scalar(db,'select count(*)-count(distinct eid) from restored')
    if reduced!=count or bad or dups or node_multiset:
        raise ValueError(dict(native_events=count,recovered=reduced,mismatch=bad,duplicates=dups,node_delta=node_multiset))
    isolated=scalar(db,"""select count(*) from units anti join (
      select src_uid uid from p_edges where src_uid<>dst_uid union
      select dst_uid from p_edges where src_uid<>dst_uid union
      select uid from r_inc join (select segment,resource from r_inc group by all having count(*)>1)
      using(segment,resource)) adjacent using(uid)""")
    nunits=scalar(db,'select count(*) from units')
    nprocess=scalar(db,'select count(*) from p_edges')
    audits=[]
    stats=[]
    for order in ('forward','reverse'):
        for hop in range(1,6):
            name=f'{order}_{hop}'
            path=root/'COMPOSITION_ASSIGNMENTS'/f'{name}.csv'
            db.execute(f"""create or replace temp table a as select unit_id uid,composition_id cid,depth
               from read_csv({lit(path)}, header=true,columns={{'unit_id':'BIGINT','composition_id':'BIGINT','depth':'INTEGER'}})""")
            missing=scalar(db,'select count(*) from units anti join a using(uid)')
            extra=scalar(db,'select count(*) from a anti join units using(uid)')
            dup_units=scalar(db,'select count(*)-count(distinct uid) from a')
            invalid=scalar(db,f"""select count(*) from a x left join a y on x.cid=y.uid
               left join units u on x.uid=u.uid left join units v on x.cid=v.uid
               where y.uid is null or y.cid<>y.uid or y.depth<>0 or (x.depth=0)<>(x.cid=x.uid)
               or x.depth<0 or x.depth>{hop} or u.segment<>v.segment""")
            unique_recovered,total_recovered=db.execute('select count(distinct eid),count(*) from restored join a on restored.owner=a.uid').fetchone()
            if any((missing,extra,dup_units,invalid,count-unique_recovered,count-total_recovered)):
                raise ValueError((name,missing,extra,dup_units,invalid,unique_recovered,total_recovered))
            db.execute("""create or replace temp table rc as select r.segment,r.resource,a.cid,
               count(*)::bigint units from r_inc r join a using(uid) group by all""")
            db.execute("""create or replace temp table res_totals as select segment,resource,
               sum(units)::bigint n,count(*) refs,sum(units*(units-1)//2)::bigint inside from rc group by all""")
            edge_dir=root/'BOUNDARY_RELATIONS'/name
            copy(db,'select rc.* from rc join res_totals t using(segment,resource) where t.refs>1',edge_dir/'shared_resource_groups.parquet')
            copy(db,"""select p.*,x.cid src_composition,y.cid dst_composition from p_edges p
               join a x on p.src_uid=x.uid join a y on p.dst_uid=y.uid where x.cid<>y.cid""",edge_dir/'process_edges.parquet')
            copy(db,"""select e.eid,x.cid owner_composition,y.cid object1_composition from owner_index e
               join a x on e.owner=x.uid join a y on e.object1_uid=y.uid where x.cid<>y.cid""",edge_dir/'target_subjects.parquet')
            cross_e=scalar(db,f"""select count(*) from (
               select r.eid from r_events r join res_totals t using(segment,resource) where t.refs>1
               union select eid from read_parquet({lit(edge_dir/'target_subjects.parquet')}))""")
            cross_p=scalar(db,f"select count(*) from read_parquet({lit(edge_dir/'process_edges.parquet')})")
            groups,singles,maxsize=db.execute('select count(*),count(*) filter(where n=1),max(n) from (select cid,count(*) n from a group by cid)').fetchone()
            within,across,shared,refs=db.execute("""select coalesce(sum(inside),0),coalesce(sum(n*(n-1)//2-inside),0),
               count(*) filter(where refs>1),coalesce(sum(refs) filter(where refs>1),0) from res_totals""").fetchone()
            audits.append(dict(order=order,hops=hop,original_nodes=nodes,recovered_unique_nodes=nodes,
                original_events=count,recovered_unique_events=unique_recovered,missing_nodes=node_multiset,
                missing_events=count-unique_recovered,duplicate_event_owners=total_recovered-unique_recovered,
                unassigned_units=missing,duplicate_unit_owners=dup_units,extra_units=extra,event_ids_or_field_mismatches=bad,
                cross_composition_events=cross_e,shared_resource_composition_refs=refs,graph_isolated_units=isolated,
                single_unit_compositions=singles,node_coverage=1.0,event_coverage=1.0))
            stats.append(dict(order=order,hops=hop,units=nunits,compositions=groups,single_unit_compositions=singles,
                multiunit_compositions=groups-singles,isolated_units=isolated,internal_process_relations=nprocess-cross_p,
                cross_process_relations=cross_p,internal_resource_unit_pairs=within,cross_resource_unit_pairs=across,
                internal_events=count-cross_e,cross_events=cross_e,shared_segment_resources=shared,
                shared_resource_composition_refs=refs,max_units=maxsize))
            print('passed',name,groups,'groups',cross_e,'boundary events',flush=True)
    write_csv(root/'INTEGRITY_AUDIT.csv',audits)
    write_csv(root/'PARTITION_STATISTICS.csv',stats)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('stage',nargs='?',choices=['units','partition','audit','all'],default='all')
    opt=ap.parse_args()
    root=opt.root.resolve()
    if opt.stage in ('units','all'):
        if not (root/'NATIVE_INPUT_AUDIT.json').is_file():
            raise SystemExit('Await completed NATIVE_INPUT_AUDIT.json')
        verify_training_population(root)
        native_units(root)
    if opt.stage!='units':
        with duckdb.connect(str(root/'native_units.duckdb'),read_only=True) as con:
            con.execute('set threads=6')
            if opt.stage in ('partition','all'):
                partitions(con,root)
            if opt.stage in ('audit','all'):
                audit(con,root)
