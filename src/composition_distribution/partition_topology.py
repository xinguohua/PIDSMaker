"""Describe component scale and the observed high-degree resources without projecting cliques."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from partition import lit, write_csv


def local_time(ns):
    seconds,fraction=divmod(ns,10**9)
    d=datetime.fromtimestamp(seconds,timezone.utc).astimezone(ZoneInfo('America/New_York'))
    return f'{d:%Y-%m-%d %H:%M:%S}.{fraction:09}{d:%z}'


def describe(root):
    db=duckdb.connect(str(root/'raw.duckdb'),read_only=True)
    db.execute('set threads=6')
    total=db.execute('select count(*) from units').fetchone()[0]
    rows=[]
    for order in ['forward','reverse']:
        for h in range(1,6):
            path=root/'COMPOSITION_ASSIGNMENTS'/f'{order}_{h}.csv'
            largest=db.execute(f'''select a.composition_id,count(*) n,min(s.segment),max(s.segment),
                                  min(s.host),min(s.block)
                                from read_csv_auto({lit(path)}) a join units u on u.uid=a.unit_id
                                join segments s using(segment) group by a.composition_id
                                order by n desc,a.composition_id limit 2''').fetchall()
            assert all(lo==hi for _,n,lo,hi,host,block in largest)
            for rank,(cid,n,lo,hi,host,block) in enumerate(largest,1):
                rows.append(dict(order=order,hop=h,rank=rank,composition_id=cid,units=n,total_units=total,
                                 unit_fraction=n/total,two_largest_fraction=sum(x[1] for x in largest)/total,
                                 segment=lo,host_uuid=host,training_block=block))
    write_csv(root/'LARGEST_COMPOSITIONS.csv',rows)
    db.execute('''create temp table top as select * from (
        select *,row_number() over(partition by segment order by degree desc,resource) r from (
        select segment,resource,count(*) degree from incidence group by segment,resource))
        where r<=3''')
    db.execute('''create temp table evidence as select t.*,e.eid,e.ts,e.op,
        case p.slot when 1 then e.path1 else e.path2 end observed_path
        from top t join resource_edges p using(segment,resource) join events e using(eid)''')
    out=[]
    for seg,resource,degree,rank,kind,role,uuid,host,block in db.execute('''
          select t.*,n.kind,n.role,n.uuid,s.host,s.block from top t join nodes n on n.nid=t.resource
          join segments s using(segment) order by s.segment,t.r''').fetchall():
        low,high,n,slots,first_event=db.execute('''select min(ts),max(ts),count(distinct eid),count(*),min(eid)
                from evidence where segment=? and resource=?''',[seg,resource]).fetchone()
        paths=db.execute('''select observed_path,count(*) n from evidence where segment=? and resource=?
                and observed_path is not null group by observed_path order by n desc,observed_path limit 3''',[seg,resource]).fetchall()
        ops=db.execute('''select op,count(distinct eid) n from evidence where segment=? and resource=?
                          group by op order by n desc,op''',[seg,resource]).fetchall()
        out.append(dict(segment=seg,host_uuid=host,training_block=block,rank_in_segment=rank,resource_id=resource,
            resource_uuid=uuid,node_class=kind,role=role,distinct_units=degree,implicit_unit_pairs=degree*(degree-1)//2,
            distinct_events=n,endpoint_slot_references=slots,first_ns=low,last_ns=high,
            first_local=local_time(low),last_local=local_time(high),sample_eid=first_event,
            observed_paths=' | '.join(f'{p} ({k})' for p,k in paths),
            observed_operations=' | '.join(f'{p} ({k})' for p,k in ops)))
    write_csv(root/'HIGH_DEGREE_RESOURCES.csv',out)
    for row in out:print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    describe(p.parse_args().root)
