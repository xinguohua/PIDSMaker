"""Record one qualifying witness or an exhaustive negative for every initial Top-N pair."""
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
import duckdb
from connectable import setup
from connectable_chains import WitnessSearch

src,res,out=map(Path,sys.argv[1:])
db=duckdb.connect()
setup(db,src,100)
pool={h:[] for h in range(1,6)}
for h,t in db.execute('select hop,tag from ranks where rk<=100 order by hop,rk').fetchall():pool[h].append(t)
direct={h:set() for h in pool}
witness={}
for h,a,b,r,ea,eb in db.execute("""select a.hop,a.tag,b.tag,first(a.rid order by a.rid,a.eid,b.eid),
    first(a.eid order by a.rid,a.eid,b.eid),first(b.eid order by a.rid,a.eid,b.eid)
    from access a join access b on a.hop=b.hop and a.rid=b.rid and a.sid<>b.sid and a.tag<b.tag
    group by a.hop,a.tag,b.tag""").fetchall():
    direct[h].add((a,b));witness[h,a,b]=dict(kind='resource',rid=r,port_eids=[ea,eb])
for h,a,b,parent,child,eid,donor in db.execute("""with pairs as (select a.hop,a.tag a,b.tag b,
    f.src_uid parent,f.dst_uid child,f.eid,b.uid donor from forks f join m a on a.uid=f.src_uid
    join m b on b.hop=a.hop and b.sid=f.ch and b.tag<>a.tag)
    select * from pairs qualify row_number() over(partition by hop,least(a,b),greatest(a,b)
    order by eid,donor)=1""").fetchall():
    k=(h,min(a,b),max(a,b));direct[h].add(k[1:])
    witness.setdefault(k,dict(kind='fork',parent_type=a,eid=eid,parent_uid=parent,child_uid=child,child_donor_uid=donor))
search=WitnessSearch(db)
print('direct pairs',len(witness),flush=True)
for w,path,options in search.paths(pool,direct,True):
    for key,reverse,da,dbu in options:
        witness.setdefault(key,dict(kind='chain',window=w,path=path[::-1] if reverse else path,
            donor_uids=([dbu,da] if reverse else [da,dbu])))
print('witnessed',len(witness),flush=True)
selected={h:set() for h in pool}
with (res/'SELECTED_BEHAVIORS.csv').open() as f:
    for r in csv.DictReader(f):selected[int(r['hop'])].add(r['behavior_type'])
with (out/'TOP_N_PAIR_SCREEN.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['hop','A','B','supported','both_selected','witness'])
    for h,types in pool.items():
        for a,b in itertools.combinations(sorted(types),2):
            v=witness.get((h,a,b));chosen=a in selected[h] and b in selected[h]
            assert not chosen or v
            wr.writerow((h,a,b,int(bool(v)),int(chosen),json.dumps(v,separators=(',',':')) if v else ''))
print('screen rows',sum(len(list(itertools.combinations(t,2))) for t in pool.values()))
