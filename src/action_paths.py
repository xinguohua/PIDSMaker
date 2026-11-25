"""Exact event-path dynamic programming and descriptive frozen-result analysis."""
import argparse,csv,hashlib,json,math,sqlite3,struct,subprocess,time
from pathlib import Path
from collections import Counter,defaultdict
from action_composition import dump,write,ro
from action_path_check import check_real_window
from action_window import configure,window_results
LENGTHS=(1,2,3,5)

def paths(edges, memberships=False, groups=None):
 """Combine identical typed prefixes at their actual terminal node.

 Update lengths in descending order for every group of indistinguishable
 timestamps and source sequence numbers. No event in that group can extend
 another event of the same group.
 """
 states=[None]+[defaultdict(dict) for _ in range(4)]
 totals={k:{} for k in LENGTHS}
 batches=groups if groups is not None else ((edge,) for edge in edges)
 for batch in batches:
  for k in (5,4,3,2,1):
   for eid,u,v,typ in batch:
    previous=[((),(1,(),{u} if memberships else None))] if k==1 else states[k-1].get(u,{}).items()
    for prefix,(n,w,ns) in previous:
     seq=prefix+(typ,);witness=w+(eid,);node_set=(ns|{v}) if memberships else None
     if k<5:
      dst=states[k][v]
      if seq in dst:
       old=dst[seq];old[0]+=n
       if memberships:old[2].update(node_set)
      else:dst[seq]=[n,witness,node_set]
     if k in totals:
      dst=totals[k]
      if seq in dst:
       old=dst[seq];old[0]+=n
       if memberships:old[2].update(node_set)
      else:dst[seq]=[n,witness,node_set.copy() if memberships else None]
 return totals

def selfcheck():
 # Exhaustive independent event-ID enumeration: loops, repeated types, shared events,
 # separated instances of the same semantic type, and diverging prefixes.
 es=[(0,1,1,7),(1,1,2,7),(2,2,1,8),(3,1,2,7),(4,2,3,9),(5,3,1,8)]
 got=paths(es,True);expected={k:defaultdict(lambda:[0,set()]) for k in LENGTHS}
 def visit(ids,ns,types,start):
  if len(ids) in expected:
   r=expected[len(ids)][tuple(types)];r[0]+=1;r[1].update(ns)
  if len(ids)==5:return
  for i in range(start,len(es)):
   e,u,v,t=es[i]
   if ids and u!=ns[-1]:continue
   visit(ids+[e],(ns if ids else [u])+[v],types+[t],i+1)
 visit([],[],[],0)
 for k in LENGTHS:
  assert set(got[k])==set(expected[k])
  for seq,(n,w,ns) in got[k].items():assert (n,ns)==tuple(expected[k][seq]) and len(set(w))==k
 # Sparse semantic DP equals explicit adjacency walk enumeration, repetitions allowed.
 actions={1:('a','b'),2:('b','a'),3:('b','b')};n={t:1 for t in actions}
 for k in range(1,6):
  if k>1:
   by=Counter()
   for t,v in n.items():by[actions[t][1]]+=v
   n={t:by[x[0]] for t,x in actions.items()}
  def enum(prefix):
   if len(prefix)==k:return 1
   return sum(enum(prefix+[t]) for t in actions if not prefix or actions[prefix[-1]][1]==actions[t][0])
  assert sum(n.values())==enum([])
 print('exhaustive path/membership and candidate-space checks passed',flush=True)

def pack(seq):return struct.pack('<'+'I'*len(seq),*seq)
def unpack(b):return struct.unpack('<'+'I'*(len(b)//4),b)

def analyze(a):
 selfcheck();root=a.root;out=root/'results';out.mkdir(exist_ok=True)
 for name in ('COMPOSITION_COUNTS','ACTION_DISTRIBUTIONS'): (out/name).mkdir(exist_ok=True)
 db=ro(root/'native.sqlite')
 db.execute('attach database ? as sem',('file:'+str(root/'semantic.sqlite')+'?mode=ro&immutable=1',))
 raw_audit=json.loads((root/'RAW_AUDIT.json').read_text());assert not raw_audit['mismatches']
 attrs={n:(s,c) for n,s,c in db.execute('select id,semantic,complete from sem.attributes')}
 pred={n:(gt,p,score,thr) for n,gt,p,score,thr in db.execute('select id,gt,prediction,score,threshold from pred')}
 # Known attack identities include all GT UUIDs, including nodes outside scored universe.
 attack={n for n,(gt,*_) in pred.items() if gt==1}
 gt_dir=a.cell.parents[1]/'code/Ground_Truth/orthrus'
 from xml.etree import ElementTree
 manifest=json.loads((a.cell/'manifest.json').read_text())
 uuid_to_id={u:n for n,u in db.execute('select id,uuid from nodes')}
 for rel in manifest['protocol']['ground_truth_relative_path']:
  with (gt_dir/rel).open() as f:
   for line in f:
    for token in line.replace('"','').strip().split(','):
     if token in uuid_to_id:attack.add(uuid_to_id[token])
 action_key={};actions={};acount=[Counter(),Counter()];missing_action=set()
 # Freeze one semantic representation per entity for identical train/test matching.
 for u,v,op in db.execute('select distinct u,v,op from edges order by u,v,op'):
  sk,sc=attrs[u];dk,dc=attrs[v];key=(sk,op,dk)
  if key not in action_key:
   t=len(action_key)+1;action_key[key]=t;actions[t]=key
   if not(sc and dc):missing_action.add(t)
 counts=sqlite3.connect(root/'counts.sqlite')
 counts.create_function('bigadd',2,lambda x,y:str(int(x)+int(y)))
 counts.executescript("""pragma journal_mode=wal; pragma synchronous=normal; pragma cache_size=-1048576; pragma wal_autocheckpoint=262144;
  create table if not exists counts(k integer,seq blob,train text,test text,w integer,witness blob,primary key(k,seq)) without rowid;
  create table if not exists window_done(w integer primary key,audit text);
  create table if not exists node_flags(k integer,id integer,bits integer,primary key(k,id)) without rowid;""")
 flags={k:defaultdict(int) for k in (2,3,5)}
 audits=[];all_train=set();all_train_sequences={k:set() for k in (2,3,5)}
 done={w for w, in counts.execute('select w from window_done')}
 for split in ('train','test'):
  jobs=[(w,split) for w, in db.execute('select w from windows where split=? order by w',(split,)) if w not in done]
  configure(root,attrs,action_key,pred,attack,missing_action,all_train,all_train_sequences,(paths,check_real_window,pack))
  remaining=len(jobs)
  # Counts are decimal TEXT because some paths exceed SQLite's integer range.
  # The sum of two nonnegative 18-digit values fits a signed 64-bit integer.
  column=split
  merge_sql=f"""insert into counts values(?,?,?,?,?,?) on conflict(k,seq)
      do update set {column}=case
      when length({column})<=18 and length(excluded.{column})<=18
      then {column}+excluded.{column} else bigadd({column},excluded.{column}) end"""
  for audit,rows,node_flags,real in window_results(jobs,a.workers):
   w=audit['window'];start=time.monotonic()
   if real:dump(out/'REAL_PATH_CHECK.json',dict(window=w,**real))
   counts.executemany(merge_sql,rows)
   counts.executemany("""insert into node_flags values(?,?,?) on conflict(k,id)
       do update set bits=bits|excluded.bits""", ((k,n,b) for (k,n),b in node_flags.items()))
   audit['write_s']=round(time.monotonic()-start,4)
   counts.execute('insert into window_done values(?,?)',(w,json.dumps(audit)))
   done.add(w);remaining-=1
   if len(done)%12==0 or remaining==0 or len(rows)>200000:
    counts.commit()
    dump(root/'PROGRESS.json',dict(window=w,split=split,windows_done=len(done),last_window=audit))
    print('committed windows',len(done),split,flush=True)
  if split=='train':
   for k,b in counts.execute("select k,seq from counts where train!='0'"):
    if k==1:all_train.add(unpack(b)[0])
    else:all_train_sequences[k].add(b)
 for b,tr,te in counts.execute('select seq,train,test from counts where k=1'):
  t=unpack(b)[0];acount[0][t]=int(tr);acount[1][t]=int(te)
 for k,n,b in counts.execute('select k,id,bits from node_flags'):flags[k][n]=b
 audits=[json.loads(row[0]) for row in counts.execute('select audit from window_done order by w')]
 assert len(audits)==db.execute('select count(*) from windows').fetchone()[0]
 write(out/'WINDOW_AUDIT.csv',list(audits[0]),([r[k] for k in audits[0]] for r in audits))
 write(out/'ACTION_COUNTS.csv',['action_id','source_semantics','operation','destination_semantics','semantic_complete','train_frequency','test_frequency'],((t,*actions[t],int(t not in missing_action),acount[0][t],acount[1][t]) for t in sorted(actions,key=lambda t:(-acount[0][t],-acount[1][t],t)) if acount[0][t] or acount[1][t]))
 # Fully identified subspace is calculable without treating two unknown file paths as equal.
 known_train={t for t in all_train if t not in missing_action};dp={t:1 for t in known_train};candidate={1:len(dp)}
 for k in range(2,6):
  by=Counter()
  for t,n in dp.items():by[actions[t][2]]+=n
  dp={t:by[actions[t][0]] for t in known_train}
  candidate[k]=sum(dp.values())
 observable_dp={t:1 for t in all_train};observable_space={1:len(all_train)}
 for k in range(2,6):
  by=Counter()
  for t,n in observable_dp.items():by[actions[t][2]]+=n
  observable_dp={t:by[actions[t][0]] for t in all_train}
  observable_space[k]=sum(observable_dp.values())
 summary=[];corr=[];examples=[]
 zero_file=(out/'ZERO_SHOT_ANALYSIS.csv').open('w',newline='')
 zero_writer=csv.writer(zero_file)
 zero_writer.writerow(['length','action_ids','semantic_complete','group','test_instances','action_train_frequencies'])
 for k in LENGTHS:
  train_instances=sum(r['instances_'+str(k)] for r in audits if r['split']=='train');test_instances=sum(r['instances_'+str(k)] for r in audits if r['split']=='test')
  if k==1:
   union=[((t,),acount[0][t],acount[1][t],None,None) for t in actions if acount[0][t] or acount[1][t]]
   union.sort(key=lambda r:(-r[1],-r[2],r[0]))
  else:
   union=((unpack(b),int(tr),int(te),w,unpack(wit)) for b,tr,te,w,wit in counts.execute(
       """select seq,train,test,w,witness from counts where k=? order by length(train) desc,train desc,
          length(test) desc,test desc,seq""",(k,)))
   detail=(out/'COMPOSITION_COUNTS'/f'length_{k}.csv').open('w',newline='')
   detail_writer=csv.writer(detail)
   detail_writer.writerow(['training_rank','action_ids','train_frequency','test_frequency','semantic_complete','observed_class','action_train_frequencies'])
  sum_train=sum_test=observable_zs=observable_out=observable_out_inst=0
  seen=zs=outside=outside_inst=unknown_types=unknown_test_inst=0;train_types=test_types=0
  all_dist=Counter();zs_dist=Counter();zsgroups=Counter();zstypes=Counter()
  for rank,(seq,tr,te,w,wit) in enumerate(union,1):
   train_types+=tr>0;test_types+=te>0;complete=not any(t in missing_action for t in seq);atomic=all(t in all_train for t in seq)
   sum_train+=tr;sum_test+=te
   observable_zs+=int(not tr and te>0 and atomic)
   observable_out+=int(te>0 and not atomic)
   if not atomic:observable_out_inst+=te
   if complete:
    for left,right in zip(seq,seq[1:]):assert actions[left][2]==actions[right][0]
    seen+=tr>0;zs+=tr==0 and te>0 and atomic
    outside+=te>0 and not atomic;outside_inst+=te if not atomic else 0
   else:unknown_types+=1;unknown_test_inst+=te
   kind='Seen' if tr else 'Zero-shot_all_actions_seen' if atomic else 'Zero-shot_unseen_action'
   support=[acount[0][t] for t in seq]
   if te:
    for t in seq:all_dist[t]+=te
    if not tr:
     g='all_actions_seen' if atomic else 'unseen_action';zsgroups[g]+=te;zstypes[g]+=1
     for t in seq:zs_dist[t]+=te
     if k>1:zero_writer.writerow((k,json.dumps(seq),int(complete),g,te,json.dumps(support)))
   if k>1:detail_writer.writerow((rank,json.dumps(seq),tr,te,int(complete),kind,json.dumps(support)))
   if k>1 and te and not tr and complete and sum(x['length']==k and x['category']==kind for x in examples)<2:
    # Stored witness may be training only for seen types; unseen types necessarily test.
    ev=[]
    for row in wit:
     q=db.execute('select e.*,r.seq,r.actor,r.obj,r.file,r.line,r.offset from edges e join sem.source r using(w,row) where e.w=? and e.row=?',(w,row)).fetchone()
     ev.append(dict(zip(['window','row','source_id','destination_id','timestamp_ns','operation','event_uuid','edge_label','sequence','actor_uuid','object_uuid','raw_file','raw_line','raw_offset'],q)))
    assert len({e['event_uuid'] for e in ev})==k
    assert all(x['destination_id']==y['source_id'] and (x['timestamp_ns'],x['sequence'])<(y['timestamp_ns'],y['sequence']) for x,y in zip(ev,ev[1:]))
    examples.append(dict(length=k,category=kind,actions=list(seq),action_train_frequencies=support,train_frequency=tr,test_frequency=te,events=ev,entity_semantics={str(n):json.loads(attrs[n][0]) for e in ev for n in (e['source_id'],e['destination_id'])}))
  assert (sum_train,sum_test)==(train_instances,test_instances)
  potential=candidate[k]-seen-zs;assert potential>=0
  summary.append((k,train_instances,train_types,test_instances,test_types,candidate[k],seen,zs,potential,seen/candidate[k] if candidate[k] else 'NA',outside,outside_inst,unknown_types,unknown_test_inst,'fully_identified_action_subspace; full_semantic_result_incomplete' if missing_action else 'complete'))
  assert observable_space[k]>=train_types+observable_zs
  summary[-1]+= (observable_space[k],train_types,observable_zs,observable_space[k]-train_types-observable_zs,
                 train_types/observable_space[k] if observable_space[k] else 'NA',observable_out,observable_out_inst)
  if k==1:continue
  detail.close()
  total_all=sum(all_dist.values());total_zs=sum(zs_dist.values());assert total_all==k*test_instances
  order=sorted(all_dist,key=lambda t:(-all_dist[t],t))
  dist=[(i+1,t,all_dist[t],zs_dist[t],all_dist[t]/total_all if total_all else 'NA',zs_dist[t]/total_zs if total_zs else 'NA',acount[0][t],int(t not in missing_action)) for i,t in enumerate(order)]
  fields=['rank','action_id','All_count','Zero_shot_count','All_fraction','Zero_shot_fraction','train_frequency','semantic_complete']
  write(out/'ACTION_DISTRIBUTIONS'/f'length_{k}_full.csv',fields,dist);write(out/'ACTION_DISTRIBUTIONS'/f'length_{k}_top25.csv',fields,dist[:25])
  for scope,sel in [('all_categories',order),('top25_shared_All_order',order[:25])]:
   x=[all_dist[t]/total_all for t in sel] if total_all else [];y=[zs_dist[t]/total_zs for t in sel] if total_zs else []
   status='ok';r='NA'
   if len(x)<2 or not y:status='empty_or_insufficient'
   else:
    mx=sum(x)/len(x);my=sum(y)/len(y);xx=sum((v-mx)**2 for v in x);yy=sum((v-my)**2 for v in y)
    if not xx or not yy:status='zero_variance'
    else:r=sum((v-mx)*(z-my) for v,z in zip(x,y))/math.sqrt(xx*yy)
   corr.append((k,scope,len(sel),total_all,total_zs,r,status))
  write(out/'ACTION_DISTRIBUTIONS'/f'length_{k}_zero_shot_groups.csv',['group','types','instances','type_fraction','instance_fraction'],((g,zstypes[g],zsgroups[g],zstypes[g]/sum(zstypes.values()) if zstypes else 'NA',zsgroups[g]/sum(zsgroups.values()) if zsgroups else 'NA') for g in ('all_actions_seen','unseen_action')))
 write(out/'COMPOSITION_SUMMARY.csv',['length','train_instances','train_observed_types','test_instances','test_observed_types','candidate_known_semantics','Seen_known_semantics','Zero_shot_known_semantics','Potential_known_semantics','training_coverage_known_semantics','test_unseen_action_known_types','test_unseen_action_known_instances','semantic_unknown_observed_types','semantic_unknown_test_instances','status','candidate_observable_signatures','Seen_observable_signatures','Zero_shot_observable_signatures','Potential_observable_signatures','coverage_observable_signatures','test_unseen_action_observable_types','test_unseen_action_observable_instances'],summary)
 zero_file.close()
 write(out/'ACTION_DISTRIBUTIONS'/'CORRELATIONS.csv',['length','range','categories','All_denominator','Zero_shot_denominator','pearson','status'],corr)
 with (out/'EXAMPLES.jsonl').open('w') as f:
  for r in examples:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 performance=[];node_rows=[]
 for k in (2,3,5):
  groups=defaultdict(list)
  for n,(gt,pr,score,thr) in pred.items():
   if gt:category='attack_context_unresolved'
   else:
    f=flags[k].get(n,0);category='Zero-shot' if f&2 else 'Seen' if f==1 else 'Other'
    groups['All'].append((n,pr,score))
   groups[category].append((n,pr,score));node_rows.append((k,n,gt,pr,score,thr,category,flags[k].get(n,0)))
  for group in ('All','Seen','Zero-shot','Other','attack_context_unresolved'):
   rows=groups[group];N=len(rows);positive=sum(x[1] for x in rows);scores=sorted(x[2] for x in rows)
   normal=group!='attack_context_unresolved'
   performance.append((k,group,N,'NA' if normal else positive,positive if normal else 'NA',N-positive if normal else 'NA','NA' if normal else N-positive,positive/N if normal and N else 'NA',(N-positive)/N if not normal and N else 'NA','ok' if normal else 'baseline_only_no_context_assignment',scores[0] if N else 'NA',scores[(N-1)//2] if N else 'NA',scores[-1] if N else 'NA'))
  for group in ('Seen','Zero-shot'):
   performance.append((k,'attack_'+group,'NA','NA','NA','NA','NA','NA','NA','normal_context_not_identifiable','NA','NA','NA'))
  baseline=json.loads((a.cell/'metrics.json').read_text());assert len(groups['All'])==baseline['FP']+baseline['TN'];assert sum(r[1] for r in groups['All'])==baseline['FP']
 write(out/'DETECTION_PERFORMANCE.csv',['length','group','nodes','TP','FP','TN','FN','FPR','FNR','status','score_min','score_median_lower','score_max'],performance)
 write(out/'NODE_SCORES.csv',['length','node_id','gt','prediction','score','threshold','group','membership_bits'],node_rows)
 dump(out/'ANALYSIS_AUDIT.json',dict(exact_dp=True,sampling=False,path_lengths=LENGTHS,no_duplicate_event_per_path=True,order='raw timestampNanos, raw sequence',windows=len(audits),semantic_unknown_actions=len(missing_action),candidate_scope='fully identified semantic endpoints only',baseline_matched=True,known_attack_nodes=len(attack),source=str(root/'native.sqlite')))
 print('COMPLETE observable statistics; see semantic completeness limitation',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--cell',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.self_check:selfcheck()
 else:analyze(a)
