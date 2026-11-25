"""Bounded native-window profiling; isolated support database, no formal results."""
import json,time,sqlite3,tempfile,struct,sys
from pathlib import Path
import action_window as aw
from action_paths import paths,pack,commit_window
from action_index import signature
from action_path_check import check_real_window
root=Path(sys.argv[1])
cell=Path(sys.argv[2])
def clock():return time.perf_counter()
db=sqlite3.connect(f'file:{root}/native.sqlite?mode=ro&immutable=1',uri=True)
db.execute('attach database ? as sem',(f'file:{root}/semantic.sqlite?mode=ro&immutable=1',))
start=clock()
from action_composition import semantic_complete
attrs={n:(signature(s),semantic_complete(json.loads(s))) for n,s in db.execute('select id,semantic from sem.attributes')}
pred={n:g for n,g in db.execute('select id,gt from pred')}
attack={n for n,g in pred.items() if g}
ids={u:n for n,u in db.execute('select id,uuid from nodes')}
for rel in json.loads((cell/'manifest.json').read_text())['protocol']['ground_truth_relative_path']:
 for line in (cell.parents[1]/'code/Ground_Truth/orthrus'/rel).read_text().splitlines():
  attack.update(ids[v] for v in line.replace('"','').strip().split(',') if v in ids)
keys={};missing=set()
for u,v,op in db.execute('select distinct u,v,op from edges order by u,v,op'):
 s,c=attrs[u];t,d=attrs[v];a=keys.setdefault((s,op,t),len(keys)+1)
 if not(c and d):missing.add(a)
print('SETUP',round(clock()-start,2),'s, RSS',aw.rss_mb(),flush=True)

def middle(split):
    rows=list(db.execute('select w from windows where split=? and w!=437 order by n,w',(split,)))
    return rows[len(rows)//2][0]

with tempfile.TemporaryDirectory(dir=root.parent,prefix='rq1-native-profile-') as tmp:
    tmp=Path(tmp)
    support=tmp/'probe.sqlite'
    c=sqlite3.connect(support)
    c.executescript("""pragma journal_mode=wal;
        create table counts(k integer,seq blob,train text,test text,w integer,primary key(k,seq)) without rowid;
        create table node_flags(k,id,bits,primary key(k,id));
        create table window_done(w integer primary key,audit text);""")
    c.create_function('bigadd',2,lambda a,b:str(int(a)+int(b)))
    train=set()
    ordinary=middle('train')
    for jobs, workers in [([(ordinary,'train')],1), ([(437,'train')],1),
                          ([(middle('test'),'test')],1),
                          ([(0,'train'),(1,'train'),(2,'train')],2)]:
        aw.configure(root,attrs,keys,pred,attack,missing,train,support,(paths,check_real_window,pack))
        for audit,path,flags,_ in aw.window_results(jobs,workers):
            size=path.stat().st_size
            t=clock()
            commit_window(c,audit,aw.result_rows(path),flags)
            print(json.dumps(dict(audit=audit,workers=workers,spool_bytes=size,merge_seconds=clock()-t)),flush=True)
        train.update(struct.unpack('<I',b)[0] for b, in c.execute("select seq from counts where k=1 and train!='0'"))
    query="select seq,train,test,w from counts where k=? order by length(train) desc,train desc,length(test) desc,test desc,seq"
    plan=list(c.execute('explain query plan '+query,(3,)))
    t=clock()
    n=sum(1 for _ in c.execute(query,(3,)))
    print(json.dumps(dict(plan=plan,sorted_rows=n,sort_read_seconds=clock()-t,db_bytes=support.stat().st_size,
        scope='profile only; test membership uses measured training windows')),flush=True)
