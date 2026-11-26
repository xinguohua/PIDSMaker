"""Resolve repeated CDM UUIDs through existing (window,row) source bindings."""
import argparse,json,os,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
x=json.loads(a.audit.read_text());db=sqlite3.connect(f'file:{a.root}/velox_action_composition/semantic.sqlite?mode=ro',uri=True)
native=sqlite3.connect(f'file:{a.root}/velox_action_composition/native.sqlite?mode=ro',uri=True)
def unbox(x):return next(iter(x.values())) if isinstance(x,dict) and len(x)==1 else x
for r in x['witnesses']:
 edge=r['native_edge'];w,row,u,v,ns,op,uid,y=edge
 source=db.execute('select * from source where w=? and row=?',(w,row)).fetchone();assert source
 _,_,seq,actor,obj,file,line,offset=source
 with open(file,'rb') as f:f.seek(offset);datum=json.loads(f.readline())['datum']
 typ,raw=next(iter(datum.items()));assert typ.endswith('.Event')
 assert raw['uuid']==uid and int(raw['timestampNanos'])==ns and raw['type']==op
 assert unbox(raw['subject'])==actor and unbox(raw['predicateObject'])==obj
 endpoints={native.execute('select uuid from nodes where id=?',(i,)).fetchone()[0] for i in (u,v)}
 assert endpoints=={actor,obj}
 old=r.pop('raw_record',None) or r.pop('unverified_uuid_only_candidate',None);r['uuid_only_lookup_matches']=old is not None and old[1]==ns and old[5]==op
 r['verified_raw_source']={'binding':source,'record':raw,'checks':'UUID, exact integer timestamp, operation, actor/object UUIDs and native endpoints all match'}
x['raw_source_warning']='native.sqlite raw_events indexed only by UUID can select a different reused UUID. Use semantic.sqlite source(w,row); read exact raw file offset and validate.'
out=Path(os.environ['RUN_OUTPUT_DIR']);(out/'source_audit.json').write_text(json.dumps(x,indent=2,ensure_ascii=False));print(json.dumps({'witnesses_verified':len(x['witnesses']),'uuid_only_mismatches':sum(not r['uuid_only_lookup_matches'] for r in x['witnesses'])}))
