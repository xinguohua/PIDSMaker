"""Verify real partitions against BFS distance and ordered-seed constraints in SQL."""
import argparse
from pathlib import Path

import duckdb

from partition_native import lit, scalar, write_csv


def verify(root):
    db=duckdb.connect(str(root/'native_units.duckdb'),read_only=True)
    db.execute('set threads=6')
    results=[]
    for order in ('forward','reverse'):
        sorting='start_ns asc nulls last, uid asc' if order=='forward' else 'start_ns desc nulls first, uid desc'
        db.execute(f'create or replace temp table unit_rank as select uid,row_number() over(order by {sorting}) r from units')
        for hops in range(1,6):
            path=root/'COMPOSITION_ASSIGNMENTS'/f'{order}_{hops}.csv'
            db.execute(f'''create or replace temp table group_member as select u.uid, u.r uid_rank, g.r cid_rank,
                          source.depth
                          from read_csv_auto({lit(path)}) source
                          join unit_rank u on source.unit_id=u.uid
                          join unit_rank g on source.composition_id=g.uid''')
            # The first unclaimed unit is the group seed; neighbors may have been
            # claimed by an earlier group, but never by a later group during BFS.
            bad_seed=scalar(db,"""select count(*) from group_member where uid_rank<cid_rank
               or ((uid_rank=cid_rank)<>(depth=0))""")
            db.execute("""create or replace temp table pair as select x.uid a,y.uid b,
               x.cid_rank ca,y.cid_rank cb,x.depth da,y.depth db from proc_edges p
               join group_member x on p.src_uid=x.uid join group_member y on p.dst_uid=y.uid""")
            bad_proc=scalar(db,f"""select count(*) from pair where ca=cb and abs(da-db)>1
               or ca<cb and da<{hops} or cb<ca and db<{hops}""")
            db.execute('''create or replace temp table rm as
                    select i.segment,i.resource,g.uid,g.cid_rank,g.depth from incidence i
                    join group_member g using(uid)''')
            db.execute('''create or replace temp table bands as select *,
                max(cid_rank) over(partition by segment,resource) last_group from (
                select segment,resource,cid_rank,min(depth) lo,max(depth) hi from rm group by all)''')
            bad_resource=scalar(db,f'''select count(*) from bands where hi-lo>1
                    or (cid_rank<last_group and lo<{hops})''')
            # Every positive depth must have an actual previous-depth neighbor in
            # the same composition. This also proves existence of a rooted path.
            missing_parent=scalar(db,'''select count(*) from group_member g anti join (
                select a uid from pair where ca=cb and da=db+1
                union select b from pair where ca=cb and db=da+1
                union select r.uid from rm r join bands b using(segment,resource,cid_rank)
                    where r.depth=b.lo+1
                ) witnesses using(uid) where depth>0''')
            row=dict(order=order,hops=hops,incorrect_seed=bad_seed,
                     process_distance_or_later_claim_violation=bad_proc,
                     resource_distance_or_later_claim_violation=bad_resource,
                     unit_without_previous_layer_witness=missing_parent)
            if any([bad_seed,bad_proc,bad_resource,missing_parent]):
                raise ValueError(row)
            results.append(row)
            print(row,flush=True)
    write_csv(root/'BFS_AUDIT.csv',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    verify(parser.parse_args().root)
