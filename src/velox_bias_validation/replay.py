"""Recover every frozen test probability and actual ordered endpoint input.
Uses frozen source code and feature caches, native batch size/order, no updates.
"""
import argparse,csv,hashlib,json,os,sqlite3,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch,yaml
from yacs.config import CfgNode
from support import bucket,BINS,T,save
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--support',type=Path,required=True);a=p.parse_args()
base=a.root/'tdsc-runs/cadets-velox-fork-train-20261124a';cell=base/'output/CADETS_E3-velox-full';out=Path(os.environ['RUN_OUTPUT_DIR']);start=time.time()
sys.path[:0]=[str(base/'code/src'),str(base/'code')]
from baseline_matrix import load_evaluation_checkpoint
from pidsmaker.utils.feature_cache import load_feature_data
from pidsmaker.utils.data_utils import GraphReindexer,custom_temporal_data_loader,extract_msg_from_data
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.factory import build_model
cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()));ck=json.loads((cell/'checkpoint.json').read_text())
assert hashlib.sha256(Path(ck['path']).read_bytes()).hexdigest()==ck['sha256']
assert cfg.training.encoder.used_methods=='none' and not cfg.training.encoder.x_is_tuple and not cfg.batching.fix_buggy_graph_reindexer
assert cfg.training.decoder.predict_edge_type.loss=='cross_entropy'
assert json.loads((a.support/'summary.json').read_text())['normal_nodes']==286858
native=sqlite3.connect(f'file:{a.root}/velox_action_composition/native.sqlite?mode=ro',uri=True)
source=sqlite3.connect(f'file:{a.support}/events.sqlite?mode=ro',uri=True)
idx=GraphReindexer(device='cuda',num_nodes=native.execute('select max(id)+1 from nodes').fetchone()[0],fix_buggy_graph_reindexer=False)
ops=get_rel2id(cfg,from_zero=True);model=None;capture=[]
vector_ids={};hist=defaultdict(Counter);rounded_hist=defaultdict(Counter);round_ids={};vid_round={}
changes=Counter();coverage=Counter();rates={};peak_rates={};maxerr=0.;maxce=0.;test_batches=0
peaks=defaultdict(list)
with (a.support/'node_peaks.csv').open() as f:
 for r in csv.DictReader(f):peaks[int(r['w']),int(r['row'])].append((int(r['node']),r['group']))
result=sqlite3.connect(out/'replay.sqlite');result.execute('create table events(w int,row int,src_vector int,dst_vector int,predicted int,probabilities blob,loss real,changed int,exact_n int,exact_max int,exact_pred_n int,exact_bin text,primary key(w,row))')

def register(x):
 key=x.tobytes()
 if key not in vector_ids:
  i=len(vector_ids);vector_ids[key]=i
  q=np.rint(x*1000).astype('int32').tobytes();vid_round[i]=round_ids.setdefault(q,len(round_ids))
 return vector_ids[key]

def aggregate(container,key,truth,pred,pr,loss,h):
 z=container.setdefault(key,Counter());z['events']+=1;z['high_loss']+=loss>T;z['correct']+=truth==pred;z['sum_p_true']+=float(pr[truth]);z['sum_p_predicted']+=float(pr[pred])
 if h:
  m=max(h.values());modes=[k for k,n in h.items() if n==m];z['seen_input']+=1;z['predicted_is_train_mode']+=pred in modes;z['true_is_train_mode']+=truth in modes;z['predicted_more_frequent_than_true']+=h.get(pred,0)>h.get(truth,0);z['sum_p_train_modes']+=sum(float(pr[k]) for k in modes)

def export_rates(x):
 return [dict(group=k,**dict(v),high_loss_rate=v['high_loss']/v['events'],accuracy=v['correct']/v['events'],mean_p_true=v['sum_p_true']/v['events'],mean_p_train_modes=v['sum_p_train_modes']/v['seen_input'] if v['seen_input'] else None) for k,v in sorted(x.items())]

for split in ['train','test']:
 total=batches=0
 for w,tensor,n in native.execute('select w,tensor,n from windows where split=? order by w',(split,)):
  g=extract_msg_from_data([load_feature_data(tensor).to('cpu')],cfg)[0];offset=0
  old=source.execute('select row,u,v,ns,op,loss,normal from events where w=? order by row',(w,)).fetchall() if split=='test' else None
  for b in custom_temporal_data_loader(g,cfg.batching.intra_graph_batching.edges.intra_graph_batch_size):
   bs=len(b.dst);rawsrc=b.x_src.cpu().numpy().copy();rawdst=b.x_dst.cpu().numpy().copy()
   b.to('cuda');idx.reindex_graph(b,keep_edge_level_features=False,x_is_tuple=False)
   coords=b.edge_index.cpu().numpy();orig=b.original_edge_index.cpu().numpy();features=b.x.detach().cpu().numpy();truth=b.edge_type.argmax(1).cpu().numpy();ts=b.t.cpu().numpy()
   vids=[register(x) for x in features];pairs=[(vids[i],vids[j]) for i,j in coords.T]
   # Diagnostic only: compare final feature against pre-reindex edge feature.
   delta=np.maximum(np.max(np.abs(features[coords[0]]-rawsrc),axis=1),np.max(np.abs(features[coords[1]]-rawdst),axis=1))
   exact_changed=np.any(features[coords[0]]!=rawsrc,axis=1)|np.any(features[coords[1]]!=rawdst,axis=1)
   changes[split+'_any_bit_change']+=int(exact_changed.sum());changes[split+'_abs_change_gt_1e-5']+=int((delta>1e-5).sum())
   if split=='train':
    for pair,t in zip(pairs,truth):hist[pair][int(t)]+=1;rounded_hist[tuple(vid_round[v] for v in pair)][int(t)]+=1
   else:
    if model is None:
     model=build_model(b,device=torch.device('cuda'),cfg=cfg,max_node_num=idx.num_nodes);load_evaluation_checkpoint(model,ck['path']);model.eval()
     dec=next(m for m in model.modules() if type(m).__name__=='CustomEdgeMLP')
     dec.register_forward_hook(lambda m,i,o:capture.append((tuple(t.detach().cpu().numpy().copy() for t in i),o.detach())))
    with torch.no_grad():
     capture.clear();loss=model(b,inference=True,validation=False)['loss'].flatten();inputs,logits=capture[0]
     # These are the true decoder arguments, not a semantic proxy.
     assert np.array_equal(inputs[0],features[coords[0]]) and np.array_equal(inputs[1],features[coords[1]])
     probs=logits.softmax(1).cpu().numpy();predicted=probs.argmax(1);loss=loss.cpu().numpy()
     ce=-torch.log_softmax(logits,dim=1).gather(1,b.edge_type.argmax(1)[:,None]).flatten().cpu().numpy()
    maxce=max(maxce,float(np.max(np.abs(ce-loss))));insert=[]
    for j,pair in enumerate(pairs):
     row=offset+j;ref=old[row];t=int(truth[j]);pr=probs[j];guess=int(predicted[j]);l=float(loss[j]);h=hist.get(pair,{})
     assert (row,int(orig[0,j]),int(orig[1,j]),int(ts[j]),ops[t])==tuple(ref[:5])
     error=abs(l-ref[5]);maxerr=max(maxerr,error);assert error<1e-4,(w,row,l,ref[5])
     count=h.get(t,0);maximum=max(h.values()) if h else 0;bn=bucket(count,maximum);changed=int(delta[j]>1e-5)
     insert.append((w,row,*pair,guess,pr.astype('float32').tobytes(),l,changed,count,maximum,h.get(guess,0),bn))
     if ref[6]:
      aggregate(rates,'all_normal',t,guess,pr,l,h);aggregate(rates,'exact/'+bn,t,guess,pr,l,h);aggregate(rates,'changed/'+str(changed),t,guess,pr,l,h)
      rh=rounded_hist.get(tuple(vid_round[v] for v in pair),{});aggregate(rates,'rounded/'+bucket(rh.get(t,0),max(rh.values()) if rh else 0),t,guess,pr,l,rh)
      if h:aggregate(rates,'seen_exact/'+('multiple_operations' if len(h)>1 else 'single_operation'),t,guess,pr,l,h)
     for node,group in peaks.get((w,row),[]):
      aggregate(peak_rates,group,t,guess,pr,l,h);aggregate(peak_rates,group+'/'+bn,t,guess,pr,l,h)
    result.executemany('insert into events values(?,?,?,?,?,?,?,?,?,?,?,?)',insert)
   offset+=bs;batches+=1
  assert offset==n;total+=n
  if split=='test':result.commit()
  if w%50==0:print(split,w,total,flush=True);save(out/'progress.json',dict(split=split,window=w,edges=total,elapsed=time.time()-start))
 coverage[split+'_edges']=total;coverage[split+'_batches']=batches
 if split=='train':
  assert (total,batches)==(3426589,3665)
  with (out/'training_actual_input_counts.csv').open('w') as f:
   wr=csv.writer(f);wr.writerow(['src_vector','dst_vector','operation_id','count'])
   for pair,h in hist.items():
    for op,count in h.items():wr.writerow([*pair,op,count])
  save(out/'training_input_summary.json',dict(edges=total,batches=batches,exact_pairs=len(hist),multi_operation_pairs=sum(len(h)>1 for h in hist.values()),changes=dict(changes)))
  print('Training actual input counts saved',flush=True)
 else:assert total==source.execute('select count(*) from events').fetchone()[0]
assert maxerr<1e-4 and maxce<1e-5
assert peak_rates['FP']['events']==1174 and peak_rates['TN']['events']==285684
# Vectors and full per-event probabilities remain on server; small summaries are fetched.
np.save(out/'actual_endpoint_vectors.npy',np.stack([np.frombuffer(x,dtype='float32') for x in vector_ids]))
save(out/'operation_ids.json',{str(k):v for k,v in ops.items() if isinstance(k,int)})
summary=dict(coverage=dict(coverage),checkpoint=ck,max_original_loss_error=maxerr,max_ce_error=maxce,normal_rates=export_rates(rates),canonical_peak_rates=export_rates(peak_rates),changes=dict(changes),unique_endpoint_vectors=len(vector_ids),training_pairs=len(hist),probabilities='float32 blob in replay.sqlite, 11 entries ordered by operation_ids.json',inputs='actual_endpoint_vectors.npy indexed by src_vector,dst_vector; actual decoder arguments verified for every scored edge',rounded='sensitivity only: each component rounded to 0.001, not identical-input evidence',elapsed_seconds=time.time()-start)
save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
