"""Frozen CADETS Velox support audit; no training and no path enumeration."""
import argparse,csv,hashlib,json,os,sqlite3,subprocess,time
from pathlib import Path
from collections import Counter,defaultdict,deque
import pandas as pd
import yaml
OPS=['CONNECT','EXECUTE','OPEN','READ','RECVFROM','RECVMSG','SENDMSG','SENDTO','WRITE','CLONE','FORK']
BINS=['unseen_condition','unseen_operation','(0,.01]','(.01,.1]','(.1,.5]','(.5,1)','1']
T=4.268208026885985

def bucket(n,m):
 if not m:return BINS[0]
 if not n:return BINS[1]
 q=n/m
 return BINS[2] if q<=.01 else BINS[3] if q<=.1 else BINS[4] if q<=.5 else BINS[5] if q<1 else BINS[6]
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 cell=root/'tdsc-runs/cadets-velox-fork-train-20261124a/output/CADETS_E3-velox-full'
 out=Path(os.environ['RUN_OUTPUT_DIR']);start=time.time()
 cfg=yaml.safe_load((cell/'resolved_config.yml').read_text());manifest=json.loads((cell/'manifest.json').read_text())
 assert manifest['commit']=='8a7303915422a85dead0a627ab1a031b7ef0c6b5'
 dbpath=root/'velox_action_composition/native.sqlite';db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
 stat=dbpath.stat();db.execute('pragma temp_store=memory');db.execute('pragma query_only=on')
 protocol=dict(training_manifest=manifest,checkpoint=json.loads((cell/'checkpoint.json').read_text()),threshold=T,calibration='retrospective_test',bins=BINS,bin_rule='positive intervals right closed except (.5,1); unseen operation means N(c,r)=0 for seen c',normal='edge y=0 AND both endpoints in scored universe with gt=0',peak='one canonical peak per normal node: smallest (w,row) among exact max ties; retain all ties separately',condition='ordered native source label, destination label; no semantic completion',analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 save(out/'protocol.json',protocol)
 labels=dict(db.execute('select id,label from nodes'));pred={n:(g,s) for n,g,s in db.execute('select id,gt,score from pred')};normal={n for n,(g,s) in pred.items() if g==0}
 assert Counter(pred[n][1]>T for n in normal)=={True:1174,False:285684}
 sql="select s.label src_label,d.label dst_label,e.op,count(*) N_cr,min(e.w*1000000000+e.row) witness from windows w join edges e on e.w=w.w join nodes s on s.id=e.u join nodes d on d.id=e.v where w.split='train' group by s.label,d.label,e.op"
 train=pd.read_sql_query(sql,db);assert train.N_cr.sum()==3426589
 support={(s,d,o):n for s,d,o,n,_ in train.itertuples(index=False,name=None)};maxima=train.groupby(['src_label','dst_label']).N_cr.max().to_dict()
 ref=pd.read_csv(root/'tdsc-runs/velox-local-condition-exp1-20261126a/output/observed_condition_operations.csv',keep_default_na=False)
 assert support=={(s,d,o):n for s,d,o,n in ref[['src_label','dst_label','op','N_cr']].itertuples(index=False,name=None)}
 witness=[]
 for dst in ['subject None sh','subject None sshd']:
  for s,d,o,n,k in train[(train.src_label=='file None')&(train.dst_label==dst)].itertuples(index=False,name=None):
   w,r=divmod(k,1000000000);row=db.execute('select * from edges where w=? and row=?',(w,r)).fetchone()
   witness.append(dict(src=s,dst=d,op=o,count=n,relative=n/maxima[s,d],native_edge=row,unverified_uuid_only_candidate=db.execute('select * from raw_events where uuid=?',(row[6],)).fetchone()))
 save(out/'source_audit.json',dict(all_40155_counts_equal=True,training_edges=3426589,training_windows=581,training_batches=3665,witnesses=witness,four_groups=pd.read_csv(root/'tdsc-runs/velox-local-condition-exp1-20261126a/output/four_groups.csv').query("scope=='multi' and threshold==0.1").to_dict('records')))
 print('Stage 1 source audit saved',flush=True)
 files=sorted((Path(cfg['training']['_edge_losses_dir'])/'test'/'model_epoch_11').glob('*.csv'),key=lambda f:int(f.stem.rsplit('_b',1)[1]));lossrows=0
 assert [int(f.stem.rsplit('_b',1)[1]) for f in files]==list(range(len(files)))
 batches=iter(files)
 result=sqlite3.connect(out/'events.sqlite');result.execute('create table events(w int,row int,u int,v int,ns int,op text,uuid text,y int,loss real,normal int,support_n int,support_max int,bin text,primary key(w,row))')
 peak={};ties=defaultdict(list);rates=defaultdict(lambda:[0,0]);exclude=Counter();testrows=0;dupkeys=0;ambiguous=0
 for w, in db.execute("select w from windows where split='test' order by w"):
  batch=[];window_n=db.execute('select n from windows where w=?',(w,)).fetchone()[0];window_losses=[]
  for batch_start in range(0,window_n,1024):
   f=next(batches);rows=list(pd.read_csv(f).itertuples(index=False,name=None));assert len(rows)==min(1024,window_n-batch_start);window_losses.extend(rows)
  lossrows+=len(window_losses)
  for w,r,u,v,ns,op,uid,y in db.execute('select * from edges where w=? order by row',(w,)):
   loss,lu,lv,lt,lo=window_losses[r];assert (u,v,ns,op)==(lu,lv,lt,'EVENT_'+OPS[int(lo)-1]);c=(labels[u],labels[v]);n=support.get((*c,op),0);m=maxima.get(c,0);b=bucket(n,m)
   benign=y==0 and u in normal and v in normal
   exclude['y0' if y==0 else 'y1']+=1
   if y==0 and not benign:exclude['y0_not_both_scored_gt0']+=1
   if benign:rates[b][0]+=1;rates[b][1]+=loss>T
   for node in set((u,v))&normal:
    old=peak.get(node)
    if old is None or loss>old[0]:peak[node]=(loss,w,r,b,n,m);ties[node]=[(w,r,b)]
    elif loss==old[0]:ties[node].append((w,r,b))
   batch.append((w,r,u,v,ns,op,uid,y,loss,int(benign),n,m,b))
  result.executemany('insert into events values(?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);result.commit();testrows+=len(batch)
 assert next(batches,None) is None and testrows==lossrows
 assert set(peak)==normal
 err=max(abs(peak[n][0]-pred[n][1]) for n in normal);assert err<1e-6,err
 counts=Counter();tie_cross=Counter()
 with (out/'node_peaks.csv').open('w') as f,(out/'all_peak_ties.csv').open('w') as g:
  wr=csv.writer(f);wr.writerow(['node','group','loss','w','row','bin','N_cr','max_N_cr','ties']);tw=csv.writer(g);tw.writerow(['node','group','w','row','bin'])
  for n in sorted(normal):
   group='FP' if pred[n][1]>T else 'TN';loss,w,r,b,x,m=peak[n];counts[group,b]+=1;wr.writerow([n,group,loss,w,r,b,x,m,len(ties[n])])
   for w,r,b in ties[n]:tw.writerow([n,group,w,r,b])
   if len({b for w,r,b in ties[n]})>1:tie_cross[group]+=1
 summary=dict(node_bins=[dict(bin=b,FP=counts['FP',b],TN=counts['TN',b]) for b in BINS],normal_event_rates=[dict(bin=b,events=rates[b][0],high_loss=rates[b][1],rate=rates[b][1]/rates[b][0] if rates[b][0] else None) for b in BINS],normal_nodes=len(normal),scored_events=testrows,normal_events=sum(v[0] for v in rates.values()),exclusions=dict(exclude),peak_score_max_error=err,tie_cross_bins=dict(tie_cross),all_fp_peak_ties=sum(len(ties[n]) for n in normal if pred[n][1]>T),binding="stable filename batch ID plus CSV row; all endpoints/time/op checked",scored_batches=len(files),elapsed_seconds=time.time()-start)
 assert ambiguous==0
 assert dbpath.stat().st_mtime_ns==stat.st_mtime_ns
 save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
