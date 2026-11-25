"""Independent upper bound on joint occurrences from native components, ignoring templates."""
import csv
import itertools
import sys
from collections import Counter,defaultdict
from pathlib import Path
import duckdb
import igraph as ig

src,results,out=map(Path,sys.argv[1:])
db=duckdb.connect()
files=lambda p:csv.DictReader(p.open())
types={(int(r['hop']),r['behavior_type']) for r in files(results/'SELECTED_BEHAVIORS.csv')}
db.execute(f"attach '{src}/native_units.duckdb' as g (read_only)")
db.execute(f"""create temp table selected as select hop,behavior_type tag from read_csv_auto('{results}/SELECTED_BEHAVIORS.csv')""")
# Reconstruct incidence from native tensor edge endpoints and native node kinds independently of relation DB.
db.execute("""create temp table native_arcs as select e.window_id w,e.src x,e.dst y,e.op,n.kind sk,d.kind dk
    from g.native_edges e join g.native_nodes n on n.nid=e.src and n.window_id=e.window_id
    join g.native_nodes d on d.nid=e.dst and d.window_id=e.window_id""")
arcs=db.execute("""select distinct w,least(x,y),greatest(x,y) from native_arcs where x<>y and
    ((sk='subject' and dk<>'subject') or (dk='subject' and sk<>'subject') or
    (sk='subject' and dk='subject' and op='EVENT_FORK')) order by 1""").fetchall()
holders=defaultdict(lambda:defaultdict(set))
rows=db.execute(f"""select i.hop,i.composition_type,i.composition_id,u.segment,u.native_nid
    from read_csv_auto('{src}/COMPOSITION_TYPE_INSTANCES.csv') i join selected s on
    s.hop=i.hop and s.tag=i.composition_type join
    (select unit_id uid,composition_id cid,regexp_extract(filename,'forward_([1-5])',1)::int hop
    from read_csv_auto('{src}/COMPOSITION_ASSIGNMENTS/forward_*.csv',filename=true)) a
    on a.hop=i.hop and a.cid=i.composition_id join g.units u using(uid) where i."order"='forward'""").fetchall()
for h,t,c,w,n in rows:holders[w][n].add((h,t,c))
joint=defaultdict(set);co=defaultdict(set)
def pairs(group,dest):
    byhop=defaultdict(set)
    for h,t,c in group:byhop[h].add((t,c))
    for h,items in byhop.items():
        for (a,ca),(b,cb) in itertools.combinations(sorted(items),2):
            dest[h,a,b].add((ca,cb))
for w,nodes in holders.items():
    pairs(itertools.chain.from_iterable(nodes.values()),co)
for w,part in itertools.groupby(arcs,key=lambda row:row[0]):
    if w not in holders:continue
    edges=[(x,y) for _,x,y in part]
    ids=sorted({i for e in edges for i in e}|set(holders[w]));ix={x:i for i,x in enumerate(ids)}
    graph=ig.Graph(len(ids),[(ix[x],ix[y]) for x,y in edges])
    for cc in graph.connected_components():
        pairs(itertools.chain.from_iterable(holders[w].get(ids[j],()) for j in cc),joint)
with (out/'NATIVE_CONNECTIVITY_AUDIT.csv').open('w',newline='') as stream:
    wr=csv.writer(stream);wr.writerow(['hop','A','B','same_window_pairs','native_connected_pairs'])
    for h in range(1,6):
        for a,b in itertools.combinations_with_replacement(sorted(t for hh,t in types if hh==h),2):
            wr.writerow((h,a,b,len(co[h,a,b]),len(joint[h,a,b])))
print('upper bounds distinct types',[(h,sum(len(v) for (hh,a,b),v in joint.items() if hh==h and a!=b)) for h in range(1,6)])
print('upper bounds same types',[(h,sum(len(v) for (hh,a,b),v in joint.items() if hh==h and a==b)) for h in range(1,6)])
with (out/'NATIVE_CONNECTED_CASES.csv').open('w',newline='') as stream:
    wr=csv.writer(stream);wr.writerow(['hop','A','B','cid_A','cid_B'])
    for k,cs in joint.items():
        for x,y in sorted(cs):wr.writerow([*k,x,y])
