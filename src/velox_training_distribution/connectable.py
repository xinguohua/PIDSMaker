"""Rank cached Velox partitions and find witnessed links between complete types."""
import argparse
import csv
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

import duckdb

SOURCE = '/mnt/bigdata/nsas2020/tdsc-runs/rq1-velox-native-distribution-20261124a'


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def output(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns.split())
        writer.writerows(rows)


def setup(db, source, n):
    db.execute(f"ATTACH '{source}/native_units.duckdb' AS g (READ_ONLY)")
    db.execute(f"ATTACH '{source}/native_types.duckdb' AS t (READ_ONLY)")
    db.execute("set threads=4")
    db.execute("set memory_limit='6GB'")
    db.execute(f"""create temp table inst as select hop,composition_id cid, composition_type tag
        from read_csv_auto('{source}/COMPOSITION_TYPE_INSTANCES.csv') where "order"='forward'""")
    db.execute("""create temp table ranks as select *,row_number() over
        (partition by hop order by f desc,tag) rk from
        (select hop,tag,count(*) f from inst group by all)""")
    db.execute(f"""create temp table assignment as
        select regexp_extract(filename,'forward_([1-5])',1)::int hop, unit_id uid,composition_id cid
        from read_csv_auto('{source}/COMPOSITION_ASSIGNMENTS/forward_*.csv',filename=true)""")
    db.execute(f"""create temp table m as select i.hop,i.tag,i.cid,a.uid,u.native_nid sid,u.segment w
        from assignment a join inst i using(hop,cid) join ranks k using(hop,tag)
        join g.units u using(uid) where k.rk<={n}""")
    assert db.execute('select count(*) from assignment').fetchone()[0] == 136367*5
    assert db.execute('select count(distinct (hop,uid)) from assignment').fetchone()[0] == 136367*5
    db.execute("""create temp table forks as select p.eid,p.ts,p.op,p.src_uid,p.dst_uid,
        x.native_nid pa,y.native_nid ch,x.segment w from g.proc_edges p
        join g.units x on x.uid=p.src_uid join g.units y on y.uid=p.dst_uid
        where p.src_uid<>p.dst_uid and x.segment=y.segment""")
    db.execute("""create temp table ports as select m.*,nn.native_nid rid,nn.kind,
        e.eid,e.ts,e.op,e.direction,t.ut.unit_type
        from m join g.resource_events r using(uid)
        join g.nodes nn on nn.nid=r.resource join t.actor e using(eid) join t.ut on t.ut.uid=m.uid""")
    db.execute("""create temp table shared as select distinct n.native_nid rid from
        (select resource from g.incidence group by resource having count(distinct uid)>1) s
        join g.nodes n on s.resource=n.nid""")
    db.execute("""create temp table access as select hop,tag,rid,sid,min(eid) eid from ports
        semi join shared using(rid) group by all""")
    db.execute("create temp table subj as select distinct hop,tag,sid from m")
    db.execute("""create temp table rpair as select distinct a.hop,a.tag a,b.tag b from access a
        join access b on a.hop=b.hop and a.rid=b.rid and a.tag<=b.tag and a.sid<>b.sid""")
    db.execute("""create temp table ppair as select distinct a.hop,least(a.tag,b.tag) a,greatest(a.tag,b.tag) b
        from forks f join subj a on f.pa=a.sid join subj b on b.hop=a.hop and f.ch=b.sid""")

def clique(order, graph, k):
    if k==0:
        return []
    for pos,node in enumerate(order):
        if len(order)-pos<k:
            break
        nxt=[x for x in order[pos+1:] if x in graph[node]]
        if len(nxt)>=k-1:
            rest=clique(nxt,graph,k-1)
            if rest is not None:
                return [node]+rest
    return None


def linked_graphs(db):
    """Components only index search. Reconstruct every retained indirect witness below."""
    import igraph as ig
    rows=db.execute("""select distinct u.segment,u.uid,-1-i.resource node from g.incidence i
        join g.units u using(uid) semi join (select resource from g.incidence group by resource
        having count(*)>1) shared using(resource)
        union all select w,src_uid,dst_uid from forks order by 1""").fetchall()
    subjects=dict(db.execute("select uid,native_nid from g.units").fetchall())
    anchors=defaultdict(lambda:defaultdict(set))
    for h,tag,sid in db.execute('select * from subj').fetchall():
        anchors[sid][h].add(tag)
    graphs={}; possibilities=defaultdict(set)
    for w, chunk in itertools.groupby(rows,key=lambda r:r[0]):
        edges=[(a,b) for _,a,b in chunk]
        ids=sorted(set(itertools.chain.from_iterable(edges)))
        ix={v:i for i,v in enumerate(ids)}
        g=ig.Graph(len(ids),[(ix[a],ix[b]) for a,b in edges])
        graphs[w]=(g,ids,ix)
        for comp in g.connected_components():
            tags=defaultdict(lambda:defaultdict(set))
            for j in comp:
                uid=ids[j]
                if uid<0:continue
                sid=subjects[uid]
                for h,types in anchors.get(sid,{}).items():
                    for tag in types:tags[h][tag].add(uid)
            for h,groups in tags.items():
                for a,b in itertools.combinations_with_replacement(sorted(groups),2):
                    if len(groups[a]|groups[b])>1:
                        possibilities[h].add((a,b))
    return graphs,possibilities,subjects,anchors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--top',type=int,default=100)
    parser.add_argument('--selection-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    db=duckdb.connect()
    audit=json.loads((args.source/'NATIVE_INPUT_AUDIT.json').read_text())
    assert (audit['windows'],audit['edge_occurrences'],audit['native_loader_train_batches'])==(581,3426589,3665)
    for r in csv_rows(args.source/'INTEGRITY_AUDIT.csv'):
        assert all(int(r[k])==0 for k in ('missing_nodes','missing_events','duplicate_event_owners','duplicate_unit_owners','event_ids_or_field_mismatches'))
    setup(db,args.source,args.top)
    ranking=db.execute('select hop,rk,tag,f from ranks where rk<=? order by hop,rk',[args.top]).fetchall()
    output(args.out/'TOP_N_BEHAVIORS.csv','hop rank behavior_type frequency',ranking)
    ranks={(h,t):(r,f) for h,r,t,f in ranking}
    direct=defaultdict(set)
    for h,a,b in db.execute('select * from rpair union select * from ppair').fetchall():
        direct[h].add((a,b))
    print('direct',[(h,len(direct[h])) for h in range(1,6)],flush=True)
    graphs,possible,subjects,anchors=linked_graphs(db)
    adjacent=defaultdict(lambda:defaultdict(set))
    for h in range(1,6):
        for a,b in possible[h]|direct[h]:
            adjacent[h][a].add(b)
            adjacent[h][b].add(a)
    common=set.intersection(*[{t for hh,r,t,f in ranking if hh==h} for h in range(1,6)])
    order=sorted(common,key=lambda t:(sum(ranks[h,t][0] for h in range(1,6)),tuple(ranks[h,t][0] for h in range(1,6)),t))
    combined={t:set.intersection(*(adjacent[h][t] for h in range(1,6))) for t in common}
    shared=clique(order,combined,10)
    chosen={}
    for h in range(1,6):
        ordered=[t for hh,r,t,f in ranking if hh==h]
        choice=shared or clique(ordered,adjacent[h],10)
        if choice is None:
            if args.top<500:raise RuntimeError('expand initial ranking')
            choice=next(c for k in range(9,0,-1) if (c:=clique(ordered,adjacent[h],k)) is not None)
        chosen[h]=sorted(choice,key=lambda t:ranks[h,t][0])
    output(args.out/'SELECTED_BEHAVIORS.csv','hop position rank behavior_type frequency',[
        (h,i,*[ranks[h,t][0],t,ranks[h,t][1]]) for h,types in chosen.items() for i,t in enumerate(types,1)])
    print('selected',[(h,[ranks[h,t][0] for t in types],sum(tuple(sorted((a,b))) not in direct[h] for a,b in itertools.combinations(types,2))) for h,types in chosen.items()],flush=True)
    if args.selection_only:return
    from connectable_matches import run
    run(db,args,chosen,direct,possible,graphs,subjects,anchors,ranks)

if __name__=='__main__':
    main()
