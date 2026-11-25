"""Frozen native Velox event paths. Analysis only: never train or alter model inputs."""
import argparse,csv,gzip,hashlib,itertools,json,math,os,sqlite3,sys,time
from pathlib import Path
from collections import Counter,defaultdict

def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,fields,rows):
 with open(p,'w',newline='') as f:
  w=csv.writer(f);w.writerow(fields);w.writerows(rows)
def ro(p):return sqlite3.connect('file:'+str(p)+'?mode=ro&immutable=1',uri=True)
def unbox(x):return next(iter(x.values())) if isinstance(x,dict) and len(x)==1 else x
def present(x):return isinstance(x,str) and x.strip().lower() not in ('','null','none')

def extract(a):
 import torch,yaml
 sys.path.insert(0,str(a.cell.parents[1]/'code'))
 from pidsmaker.utils.feature_cache import load_feature_data
 cfg=yaml.safe_load((a.cell/'resolved_config.yml').read_text());manifest=json.loads((a.cell/'manifest.json').read_text())
 a.root.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(a.root/'native.sqlite')
 db.executescript('''create table windows(w integer primary key,split text,name text,graph text,tensor text,n integer,graph_sha text,tensor_sha text);
 create table nodes(id integer primary key,kind text,label text,uuid text);
 create table edges(w integer,row integer,u integer,v integer,ns integer,op text,uuid text,y integer,primary key(w,row));
 create table pred(id integer primary key,uuid text,gt integer,prediction integer,score real,threshold real);
 ''')
 mapping=json.loads((a.cell/'uuid_mapping.json').read_text());mark=json.loads((a.cell/'export_complete.json').read_text())
 assert sha(a.cell/'objects.jsonl')==mark['files']['objects.jsonl']
 for line in (a.cell/'objects.jsonl').open():
  r=json.loads(line);assert mapping[str(r['node_id'])]==r['uuid']
  db.execute('insert into pred values(?,?,?,?,?,?)',tuple(r[k] for k in ('node_id','uuid','gt','prediction','score','threshold')))
 audit=[];w=0
 for split in ('train','test'):
  files=sorted((Path(cfg['feat_inference']['_edge_embeds_dir'])/split).iterdir())
  for fp in files:
   name=fp.name.removesuffix('.TemporalData.simple');gp=Path(cfg['transformation']['_graphs_dir'])/('graph_'+name[:10])/name
   assert name[:10] in manifest['protocol'][split+'_dates']
   g=torch.load(gp,map_location='cpu');d=load_feature_data(fp).to('cpu')
   assert g.number_of_edges()==d.src.numel()
   rows=[]
   for i,(u,v,key,e) in enumerate(g.edges(keys=True,data=True)):
    assert (int(u),int(v),int(e['time']),int(e['y']))==(int(d.src[i]),int(d.dst[i]),int(d.t[i]),int(d.y[i]))
    rows.append((w,i,int(u),int(v),int(e['time']),e['label'],e['event_uuid'],int(e['y'])))
   for n,at in g.nodes(data=True):
    row=(int(n),at['node_type'],at['label'],mapping[str(n)])
    old=db.execute('select * from nodes where id=?',(n,)).fetchone()
    assert old is None or old==row
    db.execute('insert or ignore into nodes values(?,?,?,?)',row)
   db.executemany('insert into edges values(?,?,?,?,?,?,?,?)',rows)
   record=(w,split,name,str(gp),str(fp),len(rows),sha(gp),sha(fp));db.execute('insert into windows values(?,?,?,?,?,?,?,?)',record)
   db.commit();w+=1
   if w%25==0:print('native windows',w,flush=True)
 db.executescript('create index edge_uuid on edges(uuid);create index window_time on edges(w,ns,row);')
 assert db.execute("select sum(n) from windows where split='train'").fetchone()[0]==3426589
 assert db.execute("select count(*) from windows where split='train'").fetchone()[0]==581
 write(a.root/'NATIVE_WINDOWS.csv',[x[0] for x in db.execute('select * from windows').description],db.execute('select * from windows'))
 dump(a.root/'INPUT_AUDIT.json',dict(cell=str(a.cell),training_manifest=manifest,checkpoint=json.loads((a.cell/'checkpoint.json').read_text()),metrics=json.loads((a.cell/'metrics.json').read_text()),export=mark,window_counts=db.execute('select split,count(*),sum(n) from windows group by split').fetchall(),native_rows_verified=True))
 db.close()

def recover(a):
 # Raw log audit restores only attributes/evidence; it never adds events to native.sqlite.
 db=sqlite3.connect(a.root/'native.sqlite');raw=ro(a.events)
 db.executescript('create table raw_events(uuid text primary key,ns integer,seq integer,actor text,obj text,op text,file text,line integer,offset integer,exec text);create table attributes(id integer primary key,semantic text,complete integer,evidence text);')
 nodes={r[3]:(r[0],r[1],r[2]) for r in db.execute('select * from nodes')}
 wanted={r[0] for r in db.execute('select distinct uuid from edges')}
 attrs={u:dict(paths=set(),cmds=set(),subtypes=set(),evidence=[]) for u in nodes}
 for n,kind,path,cmd,u in raw.execute('select * from nodes'):
  if u not in attrs:continue
  if present(path):attrs[u]['paths'].add(path)
  if present(cmd):attrs[u]['cmds'].add(cmd)
 files=sorted(a.raw.glob('ta1-cadets-e3-official*.json*'));assert len(files)==10
 inventory=[];prefix='com.bbn.tc.schema.avro.cdm18.'
 for fp in files:
  h=hashlib.sha256();offset=0;batch=[];count=0
  with fp.open('rb',buffering=4<<20) as f:
   for ln,line in enumerate(f,1):
    h.update(line);start=offset;offset+=len(line)
    record=json.loads(line)['datum'];typ=next(iter(record));r=record[typ];uid=r.get('uuid')
    if typ==prefix+'Event':
     for key in ('predicateObject','predicateObject2'):
      u=unbox(r.get(key));path=unbox(r.get(key+'Path'))
      if u in attrs and present(path):
       if path not in attrs[u]['paths']:attrs[u]['evidence'].append([str(fp),ln,start,key+'Path',path])
       attrs[u]['paths'].add(path)
     if uid not in wanted:continue
     actor=unbox(r.get('subject'));obj=unbox(r.get('predicateObject'));ex=r.get('properties',{}).get('map',{}).get('exec')
     batch.append((uid,int(r['timestampNanos']),unbox(r.get('sequence')),actor,obj,r['type'],str(fp),ln,start,ex))
     count+=1
     if len(batch)>=10000:db.executemany('insert into raw_events values(?,?,?,?,?,?,?,?,?,?)',batch);batch=[]
    elif uid in attrs:
     at=attrs[uid];at['subtypes'].add(r.get('type',''))
     cmd=unbox(r.get('cmdLine'))
     if present(cmd):at['cmds'].add(cmd)
     for props in (r.get('properties',{}).get('map',{}),r.get('baseObject',{}).get('properties',{}).get('map',{})):
      for key in ('path','filename'):
       path=props.get(key)
       if present(path):at['paths'].add(path)
     if not at['evidence']:at['evidence'].append([str(fp),ln,start,'entity_record'])
  db.executemany('insert into raw_events values(?,?,?,?,?,?,?,?,?,?)',batch);db.commit()
  inventory.append(dict(path=str(fp),bytes=offset,sha256=h.hexdigest(),native_events=count))
  print('raw',fp.name,count,flush=True)
 for uid,(n,kind,label) in nodes.items():
  at=attrs[uid];sem=dict(kind=kind,paths=sorted(at['paths']),commands=sorted(at['cmds']),subtypes=sorted(x for x in at['subtypes'] if x))
  complete=bool(at['cmds'] or at['paths']) if kind=='subject' else bool(at['paths'])
  db.execute('insert into attributes values(?,?,?,?)',(n,json.dumps(sem,sort_keys=True,ensure_ascii=False),int(complete),json.dumps(at['evidence'],ensure_ascii=False)))
 reverse={'EVENT_EXECUTE','EVENT_LSEEK','EVENT_MMAP','EVENT_OPEN','EVENT_ACCEPT','EVENT_READ','EVENT_RECVFROM','EVENT_RECVMSG','EVENT_READ_SOCKET_PARAMS','EVENT_CHECK_FILE_ATTRIBUTES','READ'}
 mismatches=Counter();n=0
 for u,v,ns,op,uid,t,s,actor,obj,rawop in db.execute('select e.u,e.v,e.ns,e.op,e.uuid,r.ns,r.seq,r.actor,r.obj,r.op from edges e left join raw_events r using(uuid)'):
  n+=1
  if t is None:mismatches['unmatched_raw']+=1;continue
  au=nodes.get(actor,(None,))[0];ob=nodes.get(obj,(None,))[0]
  expected=(ob,au) if rawop in reverse else (au,ob)
  if (u,v)!=expected:mismatches['direction_or_entity_mismatch']+=1
  if ns!=t or op!=rawop:mismatches['timestamp_or_operation_mismatch']+=1
  if not isinstance(s,int):mismatches['sequence_missing']+=1
 db.commit()
 dump(a.root/'RAW_AUDIT.json',dict(files=inventory,edges=n,mismatches=dict(mismatches),semantic_coverage=db.execute('select n.kind,a.complete,count(*) from attributes a join nodes n using(id) group by n.kind,a.complete').fetchall()))
 if mismatches:raise ValueError(mismatches)
 db.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','recover']);p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',type=Path);p.add_argument('--events',type=Path);p.add_argument('--raw',type=Path);a=p.parse_args();globals()[a.phase](a)
