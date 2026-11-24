"""Build raw training provenance and image instances from the physical CDM projection."""
import argparse
import ast
import json
import time
import os
from pathlib import Path
import duckdb

def build(out, code):
    staged=json.loads((out/'stage/source.json').read_text())
    db=duckdb.connect(str(out/f'raw.building.{os.getpid()}.duckdb'))
    db.execute("SET threads=8; SET memory_limit='42GB'; SET preserve_insertion_order=false")
    def run(sql):db.execute(sql)
    def table(name,query):
        print('building',name,flush=True)
        run(f'CREATE TABLE {name} AS {query}')
    def count(sql):return db.execute(sql).fetchone()[0]
    def view(name,query):run(f'CREATE VIEW {name} AS {query}')
    reverse=ast.literal_eval(next(n.value for n in ast.parse((code/'pidsmaker/utils/dataset_utils.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='edge_reversed' for t in n.targets)))
    rev=','.join(repr(v) for v in reverse)
    view('physical_events',f"select * from read_parquet('{out}/stage/events*.parquet')")
    view('physical_declarations',f"select * from read_parquet('{out}/stage/decls*.parquet')")
    # Source timestamps in ns for New York local midnight. No event from a test or validation day is read as a graph edge.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    T=lambda d:int(datetime(2018,4,d,tzinfo=ZoneInfo('America/New_York')).timestamp())*10**9
    run(f'create macro block_ns(t) as case when t >= {T(2)} and t < {T(6)} then 1 when t >= {T(7)} and t < {T(10)} then 2 end')
    table('declarations',"""select *,case when start_ns>0 then block_ns(start_ns)
      when prev_host=host and next_host=host and block_ns(prev_ts)=block_ns(next_ts) then block_ns(prev_ts) end as training_block,
      case when start_ns>0 then 'source_start_timestamp' when prev_host=host and next_host=host and block_ns(prev_ts)=block_ns(next_ts)
      then 'same_host_surrounded_by_training_events' else 'undated_or_outside_scope' end as scope_origin
      from physical_declarations""")
    table('segments',"""select (row_number() over(order by host,block)-1)::integer segment,* from (
        select distinct host,block from physical_events union select distinct host,training_block from declarations where training_block is not null)""")
    table('source_meta',"""select host,uuid,first(kind order by did) kind,
       first(role order by did) filter(where training_block is not null) as role,
       first(pid order by did) filter(where training_block is not null) as pid,
       first(cmd order by did) filter(where training_block is not null and cmd is not null) as cmd,
       count(distinct kind) kinds
       from declarations group by host,uuid""")
    assert count('select count(*) from source_meta where kinds!=1')==0
    table('scope_refs',"""select host,block,uuid,bit_or(basis) basis from (
       select distinct host,block,subject_uuid uuid,1 basis from physical_events where subject_uuid is not null union all
       select distinct host,block,object1_uuid,2 from physical_events where object1_uuid is not null union all
       select distinct host,block,object2_uuid,4 from physical_events where object2_uuid is not null union all
       select distinct host,block,host,8 from physical_events union all
       select distinct host,training_block,uuid,16 from declarations where training_block is not null union all
       select distinct host,training_block,unnest([parent_uuid,principal_uuid,source_uuid,sink_uuid]),32 from declarations where training_block is not null
       ) s where uuid is not null group by host,block,uuid""")
    table('nodes',"""select (row_number() over(order by s.host,s.uuid)-1)::integer nid,s.host,s.uuid,
       coalesce(m.kind,case when (s.basis&1)<>0 then 'Subject' else 'UnresolvedObject' end) kind,
       coalesce(m.role,m.kind || ':unknown_role',case when (s.basis&1)<>0 then 'SUBJECT_UNDECLARED' else 'UNKNOWN_OBJECT' end) as role,
       m.pid,m.cmd from (select host,uuid,bit_or(basis) basis from scope_refs group by host,uuid) s
       left join source_meta m on m.host=s.host and m.uuid=s.uuid""")
    table('node_scopes',"""select n.nid,g.segment,r.basis from scope_refs r join nodes n using(host,uuid)
        join segments g using(host,block)""")
    table('raw_events',f"""select e.eid,e.uuid,e.ts,e.seq,e.host,g.segment,s.nid subject,a.nid object1,b.nid object2,
        a.kind k1,b.kind k2,e.op,e.path1,e.path2,e.exec_name,e.retval,e.event_name,e.byte_offset,e.byte_length,
        case when e.object1_uuid is null then 'unspecified' when e.op in ({rev}) then 'object_to_actor'
             else 'actor_to_object' end direction
        from physical_events e join segments g using(host,block)
        left join nodes s on e.host=s.host and e.subject_uuid=s.uuid
        left join nodes a on e.host=a.host and e.object1_uuid=a.uuid
        left join nodes b on e.host=b.host and e.object2_uuid=b.uuid""")
    # Each null subject gets its own unresolved execution placeholder; target references do not invent actors.
    # Use the observed current-image property to distinguish failed exec attempts from real image transitions.
    table('actor_states',"""with raw as (
        select eid,subject,segment,ts,seq,op,coalesce(subject::bigint,-eid-1) actor,
        nullif(regexp_extract(trim(exec_name), '[^/]+$'),'') reported,
        nullif(regexp_extract(trim(path1), '[^/]+$'),'') target, retval
        from raw_events), look as (
        select *,first_value(reported ignore nulls) over(partition by segment,actor order by ts,seq nulls last,eid
          rows between 1 following and unbounded following) AS following_name from raw),
        evidence as (
        select *,op='EVENT_EXECUTE' and (try_cast(retval as bigint)=0 or
         (target is not null and reported is not null and following_name=target and following_name<>reported)) as switched
         from look),
        names as (select *,case when switched then coalesce(target,following_name) else reported end program_hint from evidence),
        prior as (select *,last_value(program_hint ignore nulls) over(partition by segment,actor order by ts,seq nulls last,eid
          rows between unbounded preceding and 1 preceding) AS previous_program from names)
        select eid,actor,segment,ts,seq,program_hint,case when switched then 'witnessed_execute'
            when previous_program is not null and program_hint is not null and program_hint<>previous_program then 'observed_image_change' end boundary
        from prior""")
    table('observations',"""select segment,actor,eid,ts,seq,0::tinyint slot,program_hint,boundary from actor_states
        union all select segment,object1::bigint,eid,ts,seq,1::tinyint,null,null from raw_events where k1='Subject'
        union all select segment,object2::bigint,eid,ts,seq,2::tinyint,null,null from raw_events where k2='Subject'""")
    # Peers with the same UUID as the actor receive the same post-boundary image on this physical event.
    table('image_observations',"""select *,sum((boundary is not null)::integer) over(partition by segment,actor
        order by ts,seq nulls last,eid,slot rows unbounded preceding)::integer image
        from observations""")
    table('unit_groups',"""select segment,actor,image,min(ts) observed_first,max(ts) observed_last,
        first(program_hint order by ts,seq nulls last,eid) filter(where program_hint is not null) image_program,
        first(eid order by ts,seq nulls last,eid) filter(where boundary is not null) boundary_eid,
        first(boundary order by ts,seq nulls last,eid) filter(where boundary is not null) image_boundary
        from image_observations group by segment,actor,image""")
    table('subject_starts',"""select n.nid,g.segment,min(d.start_ns) filter(where d.start_ns>0) declared_start,first(d.cmd order by d.did) filter(where d.cmd is not null) as declared_cmd
        from declarations d join nodes n using(host,uuid) join segments g on g.host=d.host and g.block=d.training_block
        where d.kind='Subject' group by n.nid,g.segment""")
    table('units',"""with all_groups as (select * from unit_groups union all
        select s.segment,n.nid::bigint,0,null,null,null,null,null from node_scopes s join nodes n using(nid)
        where n.kind='Subject' and not exists(select 1 from unit_groups u where u.segment=s.segment and u.actor=n.nid)),
        enriched as (select a.*,n.nid subject,n.pid,
          coalesce(a.image_program,nullif(regexp_extract(trim(st.declared_cmd), '^([^ ]*/)?([^ /]+)',2),''),'UNKNOWN_PROGRAM') as program,
          st.declared_start,
          case when a.image=min(a.image) over(partition by a.segment,a.actor) then least(a.observed_first,st.declared_start) else a.observed_first end start_ns
          from all_groups a left join nodes n on a.actor=n.nid left join subject_starts st on st.nid=n.nid and st.segment=a.segment)
        select (row_number() over(order by start_ns nulls last,segment,actor,image)-1)::integer uid,subject,program,start_ns,
           observed_last as end_ns,segment,actor,image,pid,declared_start,observed_first,observed_last,boundary_eid,image_boundary,
           case when subject is null then 'unresolved_actor_occurrence' when observed_first is null then 'declaration_only'
                when image_boundary is not null then image_boundary when declared_start is not null then 'declared_or_observed_start'
                else 'left_censored' end as start_status
           from enriched""")
    table('event_units',"""select o.eid,min(u.uid) filter(where slot=0) owner,min(u.uid) filter(where slot=1) object1_uid,
        min(u.uid) filter(where slot=2) object2_uid from image_observations o join units u using(segment,actor,image) group by o.eid""")
    table('events',"""select e.* exclude(k1,k2),u.owner,u.object1_uid,u.object2_uid from raw_events e join event_units u using(eid)""")
    table('resource_edges',"""select e.eid,e.owner uid,e.segment,e.object1 resource,1::tinyint slot
        from events e join nodes n on e.object1=n.nid where n.kind<>'Subject'
        union all select e.eid,e.owner,e.segment,e.object2,2::tinyint from events e join nodes n on e.object2=n.nid where n.kind<>'Subject'""")
    view('resource_events',"""select r.*,e.op,e.ts,case when r.slot=1 then e.direction else 'secondary_reference' end direction,
         n.role from resource_edges r join events e using(eid) join nodes n on n.nid=r.resource""")
    table('incidence',"""select distinct uid,resource,segment from resource_edges""")
    table('proc_edges',"""select eid,owner as src_uid,object1_uid as dst_uid,ts,op from events where
         op in ('EVENT_FORK','EVENT_CLONE','EVENT_VFORK','EVENT_CREATE_THREAD') and object1_uid is not null
         and coalesce(try_cast(retval as bigint),0)>=0
         union all select u.boundary_eid,previous_uid,u.uid,e.ts,'IMAGE_CHANGE' from
         (select *,lag(uid) over(partition by segment,actor order by image) previous_uid from units) u
         join events e on e.eid=u.boundary_eid where previous_uid is not null""")
    # End times above describe an observation bound; only the following view confirms actual exits.
    view('unit_lifetimes',"""select u.*,x.exit_ns,case when x.exit_ns is not null and x.exit_ns<u.observed_last then 'exit_then_later_record_uncertain' when x.exit_ns is not null then 'observed_exit' when
       lead(u.uid) over(partition by u.segment,u.actor order by u.image) is not null then 'next_image_observed'
       else 'right_censored' end end_status from units u left join
       (select owner,max(ts) exit_ns from events where op='EVENT_EXIT' group by owner) x on x.owner=u.uid""")
    run(f"COPY (SELECT * FROM unit_lifetimes ORDER BY uid) TO '{out}/EXECUTION_UNITS.parquet' (FORMAT parquet, COMPRESSION zstd)")
    expected=sum(sum(s['events_by_block'][1:]) for s in staged)
    print('source verification',expected,flush=True)
    graph_query="""select e.eid,e.uuid,e.ts,e.seq,e.host,s.uuid subject_uuid,a.uuid object1_uuid,b.uuid object2_uuid,
       e.op,e.path1,e.path2,e.exec_name,e.retval,e.event_name,e.byte_offset,e.byte_length,g.block
       from events e left join nodes s on e.subject=s.nid left join nodes a on e.object1=a.nid left join nodes b on e.object2=b.nid join segments g using(segment)"""
    # Both sides carry a physical primary key. The match checks both nullness and ordered end-point slots.
    cols='uuid ts seq host subject_uuid object1_uuid object2_uuid op path1 path2 exec_name retval event_name byte_offset byte_length block'.split()
    same=' and '.join(f'd.{k} is not distinct from p.{k}' for k in cols)
    bad=count(f"select count(*) from ({graph_query}) d full join physical_events p using(eid) where d.eid is null or p.eid is null or not ({same})")
    assert bad==0
    n_events,n_ids,n_missing=db.execute('select count(*),count(distinct eid),count(*) filter(where owner is null) from events').fetchone()
    assert (n_events,n_ids,n_missing)==(expected,expected,0)
    assert count('select count(*) from units')==count('select count(distinct uid) from units')
    stats={'events':expected,'nodes':count('select count(*) from nodes'),'units':count('select count(*) from units'),
        'raw_projected_fields_mismatched':bad,'unowned_events':n_missing,'repeated_physical_ids':n_events-n_ids,
        'date_segments':db.sql('select * from segments order by segment').fetchall(),
        'event_ops':dict(db.sql('select op,count(*) from events group by op').fetchall()),
        'node_classes':dict(db.sql('select kind,count(*) from nodes group by kind').fetchall()),
        'start_status':dict(db.sql('select start_status,count(*) from units group by start_status').fetchall()),
        'end_status':dict(db.sql('select end_status,count(*) from unit_lifetimes group by end_status').fetchall()),
        'source_declaration_scope':db.sql("select scope_origin,training_block,count(*) from declarations group by all").fetchall(),
        'unscoped_distinct_declarations':count('select count(*) from source_meta m anti join nodes n using(host,uuid)'),
        'proc_edges':count('select count(*) from proc_edges'),'resource_edges':count('select count(*) from resource_edges'),
        'resource_memberships':count('select count(*) from incidence'),
        'unresolved_subject_rows':count('select count(*) from events where subject is null'),
        'unknown_kind_nodes':count("select count(*) from nodes where kind='UnresolvedObject'")}
    for rel in ['scope_refs','source_meta','raw_events','observations','image_observations','unit_groups','actor_states','event_units','subject_starts']:
        run('drop table '+rel)
    run('checkpoint');db.close();(out/f'raw.building.{os.getpid()}.duckdb').rename(out/'raw.duckdb')
    (out/'raw_summary.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    build(a.run.resolve(),Path(__file__).resolve().parents[2])
