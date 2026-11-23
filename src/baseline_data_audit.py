"""Bounded, sequential read-only verification of the existing public E3 databases."""
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
import psycopg2
from server_credentials import database_options
from baseline_matrix import protocol,ROOT,save,digest
p=argparse.ArgumentParser();p.add_argument('--server-config',type=Path,required=True)
p.add_argument('--dates-only',action='store_true')
p.add_argument('--statement-timeout-seconds',type=int,default=90)
a=p.parse_args()
if not 1 <= a.statement_timeout_seconds <= 600: p.error('statement timeout must be 1..600 seconds')
server=json.loads(a.server_config.read_text());out=Path(os.environ['RUN_OUTPUT_DIR']);results={}
def query(c,q,params=None):
 t=time.monotonic()
 try:
  with c.cursor() as cur:cur.execute(q,params);rows=cur.fetchall()
  return dict(ok=True,rows=rows,seconds=time.monotonic()-t,sql=q)
 except psycopg2.Error as e:
  c.rollback();return dict(ok=False,error_type=type(e).__name__,sqlstate=e.pgcode,seconds=time.monotonic()-t,sql=q)
for ds,cfg in protocol().items():
 result={'dataset':ds,'database':cfg['database'],'scope':'existing public database; raw fragment completeness not asserted','queries':{}}
 results[ds]=result
 c=psycopg2.connect(dbname=cfg['database'],**database_options(server));c.set_session(readonly=True,autocommit=False)
 def q(name,sql):
  with c.cursor() as cur:
   cur.execute("SELECT set_config('statement_timeout',%s,true)",(str(a.statement_timeout_seconds*1000),))
   cur.execute("SET LOCAL lock_timeout='3s'; SET LOCAL temp_file_limit='1GB'; SET LOCAL search_path=public,pg_catalog")
  result['queries'][name]=query(c,sql);save(out/'summary.json',results);print(ds,name,result['queries'][name]['ok'],flush=True)
 q('identity',"SELECT version(),current_setting('data_directory'),current_database(),current_setting('transaction_read_only')")
 q('date_counts',"SELECT (date '1970-01-01'+((timestamp_rec-14400000000000)/86400000000000)::integer)::text AS eastern_date, count(*),min(timestamp_rec),max(timestamp_rec),count(*) FILTER (WHERE (timestamp_rec-14400000000000)%86400000000000=0) FROM public.event_table GROUP BY 1 ORDER BY 1")
 if a.dates_only:
  c.rollback();c.close();continue
 q('nodes',"SELECT 'file',count(*),count(DISTINCT node_uuid),count(DISTINCT index_id) FROM public.file_node_table UNION ALL SELECT 'subject',count(*),count(DISTINCT node_uuid),count(DISTINCT index_id) FROM public.subject_node_table UNION ALL SELECT 'netflow',count(*),count(DISTINCT node_uuid),count(DISTINCT index_id) FROM public.netflow_node_table")
 q('node_global_unique',"SELECT count(*),count(DISTINCT node_uuid),count(DISTINCT index_id) FROM (SELECT node_uuid,index_id FROM public.file_node_table UNION ALL SELECT node_uuid,index_id FROM public.subject_node_table UNION ALL SELECT node_uuid,index_id FROM public.netflow_node_table) n")
 # Match read_ground_truth_uuids: nonempty first column, no header in these files.
 import csv
 gt=collections.Counter();files={}
 for name in cfg['ground_truth_relative_path']:
  f=ROOT/'Ground_Truth/orthrus'/name
  rows=list(csv.reader(f.open()));files[name]={'sha256':digest(f),'csv_rows':len(rows)}
  for row in rows:
   if row and row[0].strip():gt[row[0].strip()]+=1
 result['gt_files']=files
 result['gt']={'unique':len(gt),'duplicate_occurrences':sum(gt.values())-len(gt),'matched':{},'missing':[]}
 matched={}
 for table in ['file_node_table','subject_node_table','netflow_node_table']:
  res=query(c,'SELECT node_uuid,index_id FROM public.'+table+' WHERE node_uuid=ANY(%s)',(list(gt),))
  if res['ok']:
   for uuid,nid in res['rows']:matched.setdefault(uuid,[]).append({'table':table,'index_id':nid})
  else:result['gt']['query_error']=res
 result['gt']['matched_count']=len(matched)
 result['gt']['multiple_db_matches']={k:v for k,v in matched.items() if len(v)>1}
 result['gt']['missing']=[uuid for uuid in gt if uuid not in matched]
 result['gt'].pop('matched')
 try:
  h=hashlib.sha256();count=0
  for table in ['file_node_table','subject_node_table','netflow_node_table']:
   with c.cursor(name='audit_'+table) as cur:
    cur.itersize=10000;cur.execute('SELECT node_uuid,index_id FROM public.'+table+' ORDER BY node_uuid,index_id')
    for uuid,nid in cur:
     h.update((table+'\t'+str(uuid)+'\t'+str(nid)+'\n').encode());count+=1
  result['uuid_mapping']={'sha256':h.hexdigest(),'rows':count,'canonical':'table order file,subject,netflow; ORDER BY node_uuid,index_id; tab separated UTF8 with newline'}
 except psycopg2.Error as e:result['uuid_mapping']={'error_type':type(e).__name__};c.rollback()
 c.rollback();c.close();save(out/'summary.json',results)
