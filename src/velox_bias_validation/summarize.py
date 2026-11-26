"""Export compact, complete-denominator tables and all FP peak probabilities."""
import argparse,csv,json,os,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();out=Path(os.environ['RUN_OUTPUT_DIR']);T=4.268208026885985
c=sqlite3.connect(f'file:{a.support}/events.sqlite?mode=ro',uri=True)
c.execute('attach database ? as native',(f'file:{a.root}/velox_action_composition/native.sqlite?mode=ro',))
c.execute('attach database ? as replay',(f'file:{a.replay}/replay.sqlite?mode=ro',))
coverage=c.execute('''select case when p.id is null or q.id is null then 'missing_scored_endpoint' when p.gt=1 or q.gt=1 then 'at_least_one_attack_endpoint' else 'both_scored_normal' end,count(*) from events e left join native.pred p on p.id=e.u left join native.pred q on q.id=e.v group by 1''').fetchall()
incidence=c.execute('''select e.bin,count(*),sum(e.loss>?) from events e join native.pred p on p.id=e.u or p.id=e.v where p.gt=0 group by e.bin''',(T,)).fetchall()
opgroups=c.execute('select op,bin,count(*),sum(loss>?) from events where normal=1 group by op,bin',(T,)).fetchall()
with (out/'normal_operation_support_rates.csv').open('w') as f:
 wr=csv.writer(f);wr.writerow(['op','bin','events','high_loss']);wr.writerows(opgroups)
ops=json.loads((a.replay/'operation_ids.json').read_text());hist=defaultdict(Counter)
for r in csv.DictReader((a.replay/'training_actual_input_counts.csv').open()):hist[int(r['src_vector']),int(r['dst_vector'])][int(r['operation_id'])]=int(r['count'])
peak_rows=[]
for r in csv.DictReader((a.support/'all_peak_ties.csv').open()):
 if r['group']!='FP':continue
 node,w,row=int(r['node']),int(r['w']),int(r['row'])
 e=c.execute('select e.u,e.v,e.op,e.uuid,e.loss,e.normal,r.src_vector,r.dst_vector,r.predicted,r.probabilities,r.changed,r.exact_bin from events e join replay.events r using(w,row) where w=? and row=?',(w,row)).fetchone()
 u,v,op,uid,loss,normal,s,d,guess,blob,changed,b=e;pr=np.frombuffer(blob,dtype='float32');h=hist.get((s,d),{})
 labels=[c.execute('select label from native.nodes where id=?',(n,)).fetchone()[0] for n in (u,v)]
 peak_rows.append(dict(node=node,w=w,row=row,event_uuid=uid,src_label=labels[0],dst_label=labels[1],operation=op,loss=loss,both_endpoints_normal=normal,src_vector=s,dst_vector=d,predicted=ops[str(guess)],probabilities=json.dumps({ops[str(i)]:float(x) for i,x in enumerate(pr)}),train_counts=json.dumps({ops[str(i)]:n for i,n in h.items()}),exact_bin=b,reindex_change_gt_1e5=changed))
with (out/'all_fp_peak_probabilities.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=peak_rows[0]);wr.writeheader();wr.writerows(peak_rows)
assert len(peak_rows)==1467 and len({r['node'] for r in peak_rows})==1174
focus=[]
for label,op in [('subject None sh','EVENT_EXECUTE'),('subject None sh','EVENT_OPEN'),('subject None sshd','EVENT_RECVMSG'),('subject None sshd','EVENT_OPEN')]:
 rows=c.execute('''select e.loss,r.predicted,r.probabilities,r.src_vector,r.dst_vector,r.changed from events e join native.nodes u on u.id=e.u join native.nodes v on v.id=e.v join replay.events r using(w,row) where e.normal=1 and u.label='file None' and v.label=? and e.op=?''',(label,op))
 n=hi=seen=prefer=changed=0;sums=np.zeros(11);guesses=Counter();tid=int(next(k for k,v in ops.items() if v==op));train_true=train_guess=0
 for loss,guess,blob,s,d,ch in rows:
  n+=1;hi+=loss>T;changed+=ch;pr=np.frombuffer(blob,dtype='float32');sums+=pr;guesses[ops[str(guess)]]+=1;h=hist.get((s,d),{})
  if h:seen+=1;prefer+=h.get(guess,0)>h.get(tid,0)
 focus.append(dict(dst=label,op=op,normal_events=n,high_loss=hi,actual_input_seen=seen,predicted_more_frequent_than_true=prefer,changed=changed,guesses=dict(guesses),mean_probabilities={ops[str(i)]:float(v/n) for i,v in enumerate(sums)} if n else {}))
summary=dict(event_coverage=coverage,all_normal_node_event_incidence_by_bin=incidence,fp_peak_probability_rows=len(peak_rows),fp_peak_nodes=1174,focus=focus)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
