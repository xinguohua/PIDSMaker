"""Account for process-relationship evidence and raw declarations outside the training graph."""
import argparse
import json
from pathlib import Path
import duckdb

def records(db,sql):
    cur=db.execute(sql);fields=[c[0] for c in cur.description]
    return [dict(zip(fields,row)) for row in cur.fetchall()]

def main(run):
    db=duckdb.connect(str(run/'raw.duckdb'),read_only=True)
    db.execute('set threads=6')
    fork=records(db,"""select e.op,case when try_cast(retval as bigint)>=0 then 'nonnegative' when try_cast(retval as bigint)<0 then 'negative'
      else 'unknown' end as outcome,n.kind as object1_kind,e.object1_uid is not null as child_image_known,
      p.eid is not null as edge_recorded,count(*) as events
      from events e left join nodes n on n.nid=e.object1
      left join proc_edges p on p.eid=e.eid and p.op=e.op and p.src_uid=e.owner and p.dst_uid=e.object1_uid and p.ts=e.ts
      where e.op in ('EVENT_FORK','EVENT_CLONE','EVENT_VFORK','EVENT_CREATE_THREAD') group by all order by 1,2,3""")
    outside=records(db,"""with identified as (
        select host,uuid,any_value(kind) kind,bool_or(start_ns>0) dated,
        bool_or(prev_host=host and next_host=host and
          timezone('America/New_York',to_timestamp(prev_ts::double/1000000000))::date =
          timezone('America/New_York',to_timestamp(next_ts::double/1000000000))::date) bracketed,
        count(*) declarations
        from declarations anti join nodes using(host,uuid) group by host,uuid )
        select kind, case when dated then 'source_start_outside_training'
             when bracketed then 'only_source_neighbors_on_nontraining_day'
             else 'cannot_assign_an_unreferenced_declaration_to_one_day' end outcome,
             count(*) unique_ids,sum(declarations)::bigint declaration_rows from identified group by all order by 1,2""")
    unreferenced=records(db,"""with used as (select distinct subject nid from events union select distinct object1 from events union select distinct object2 from events),
       distinct_nodes as (select n.* from nodes n anti join used using(nid))
       select kind,count(*) unique_nodes from distinct_nodes group by kind order by kind""")
    execs=records(db,"""select e.retval,case when u.image_boundary='witnessed_execute' and u.boundary_eid=e.eid
       then 'witnessed_new_image_at_execute' when u.image_boundary='observed_image_change' and u.boundary_eid=e.eid
       then 'observed_current_image_changed' else 'no_new_image_proven_at_this_event' end evidence,count(*) as physical_events
       from events e join units u on e.owner=u.uid where e.op='EVENT_EXECUTE' group by all order by 1,2""")
    report=dict(fork_reconciliation=fork,execute_evidence=execs,
                declared_ids_outside_training_universe=outside,
                training_nodes_not_an_event_endpoint=unreferenced,
                meaning='Date evidence for excluded declarations is either its own start timestamp, two neighboring events on the same host and day, or unresolved. Cohort is unique (host,UUID) with no membership in the training graph.')
    assert sum(r['events'] for r in fork)==db.sql("select count(*) from events where op in ('EVENT_FORK','EVENT_CLONE','EVENT_VFORK','EVENT_CREATE_THREAD')").fetchone()[0]
    assert sum(r['unique_ids'] for r in outside)==json.loads((run/'raw_summary.json').read_text())['unscoped_distinct_declarations']
    (run/'RAW_BOUNDARIES.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run.resolve())
