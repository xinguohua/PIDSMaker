"""Stage 3. Final-checkpoint train/test accuracy, byte-level input matching and competing hypotheses."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from neighbors import loader

def main(run,source,first,second,out):
    cell,cfg,labels,ops,(train,val,test,maxnodes)=loader(run)
    from baseline_matrix import digest,load_evaluation_checkpoint
    from pidsmaker.factory import build_model
    from pidsmaker.utils.utils import get_device,set_seed
    ck=json.loads((cell/'checkpoint.json').read_text())
    assert digest(ck['path'])==ck['sha256']==json.loads((source/'protocol.json').read_text())['checkpoint']['sha256']
    set_seed(cfg,seed=0)
    device=get_device(cfg)
    model=build_model(train[0][0],device=device,cfg=cfg,max_node_num=maxnodes,all_data=(val,test))
    load_evaluation_checkpoint(model,ck['path'])
    encoder=next(m for m in model.modules() if type(m).__name__=='LinearEncoder')
    decoder=next(m for m in model.modules() if type(m).__name__=='CustomEdgeMLP')
    capture=[]
    encoder.lin1.register_forward_pre_hook(lambda m,a:capture.append(a[0].detach()))
    decoder.register_forward_hook(lambda m,a,o:capture.append(o.detach()))
    qs=pd.read_pickle(first/'queries.pkl')
    pair=['src','dst']
    target=set(zip(qs.src,qs.dst))
    keys,variants=pd.read_pickle(second/'training_variants.pkl')
    bypair=dict(zip(keys,(variants[i] for i in range(len(keys)))))
    def seen(k,a,z):return (a,z) in bypair.get(k,())
    count=defaultdict(Counter);conf=defaultdict(Counter);probs=defaultdict(Counter)
    matched=Counter();batches=0;maxdiff=0
    def add(split,k,truth,guess,pr):
        count[split,k]['n']+=1
        count[split,k]['right']+=truth==guess
        conf[split,k][truth+'>'+guess]+=1
        probs[split,k][truth]+=pr
    for b in train[0]:
        u,v=b.original_edge_index.cpu().tolist()
        chosen=[(i,k) for i,(a,z) in enumerate(zip(u,v)) if (k:=(labels[a],labels[z])) in target]
        if not chosen:continue
        b=b.to(device)
        with torch.no_grad():
            capture.clear();model(b,inference=True,validation=False)
            x,logits=capture;assert torch.equal(x,b.x)
            truth=b.edge_type.argmax(1)
            guess=logits.argmax(1).cpu().tolist()
            prob=logits.softmax(1).gather(1,truth[:,None]).flatten().cpu().tolist()
            truth=truth.cpu().tolist()
        for i,k in chosen:add('train',k,ops[truth[i]],ops[guess[i]],prob[i])
        batches+=1
    # Spot-check all actual query batches against the prior independent full test replay.
    for bi,part in qs.groupby('batch'):
        b=test[0][bi].to(device)
        with torch.no_grad():
            capture.clear()
            result=model(b,inference=True,validation=False)['loss'].flatten().cpu().numpy()
            args,logits=capture;assert torch.equal(args,b.x)
            guess=logits.argmax(1).cpu().tolist()
        for r in part.itertuples():
            assert ops[guess[r.row]]==r.guess
            maxdiff=max(maxdiff,abs(result[r.row]-r.loss))
    assert maxdiff<1e-6
    for df in pd.read_csv(source/'test_events.csv.gz',chunksize=120000,keep_default_na=False,float_precision='round_trip'):
        subset=df[df.set_index(pair).index.isin(target)]
        for r in subset.itertuples():
            k=(r.src,r.dst)
            add('test',k,r.op,r.guess,r.p_true)
            matched[k]+=seen(k,r.vec_src,r.vec_dst)
    fp_counts=qs[qs.canonical & (qs.prediction==1)].groupby(pair).size().to_dict()
    training=pd.read_csv(source/'condition_operations.csv',keep_default_na=False)
    h={k:dict(zip(g.op,g.n)) for k,g in training.groupby(pair)}
    comparisons=[]
    for k in sorted(target):
        a=count['train',k];z=count['test',k]
        assert a['n']==sum(h.get(k,{}).values())
        comparisons.append((*k,a['n'],a['right']/a['n'] if a['n'] else np.nan,z['n'],z['right']/z['n'],matched[k],matched[k]/z['n'],fp_counts.get(k,0),json.dumps(conf['train',k]),json.dumps(conf['test',k]),json.dumps(probs['train',k]),json.dumps(probs['test',k])))
    pd.DataFrame(comparisons,columns='src dst N train_accuracy test_events test_accuracy test_inputs_seen match_fraction FP training_confusion test_confusion sum_train_p_true_by_op sum_test_p_true_by_op'.split()).to_csv(out/'TRAIN_TEST_PREDICTION_COMPARISON.csv',index=False)
    nearest=pd.read_csv(second/'SIMILAR_CONTEXT_ANALYSIS.csv',keep_default_na=False)
    nn={}
    for key,group in nearest.groupby(['src','dst','vec_src','vec_dst','method']):
        nn[key]=[(r.train_count,json.loads(r.train_operations)) for r in group.itertuples()]
    def evidence(row,alternative,method):
        candidates=nn.get((row.src,row.dst,row.vec_src,row.vec_dst,method),[])
        support=Counter()
        high_mode=False
        for n,hist in candidates:
            support.update(hist)
            if n>=1000 and hist.get(alternative,0)==max(hist.values()) and hist.get(alternative,0)>hist.get(row.op,0):high_mode=True
        return len(candidates),support[alternative],support[row.op],high_mode
    node_peaks=pd.read_csv(source/'node_peaks.csv.gz',keep_default_na=False)
    records=[]
    for r in node_peaks.itertuples():
        k=r.src,r.dst;hist=h.get(k,{});n=sum(hist.values());wrong=r.op!=r.guess
        byte_match=seen(k,r.vec_src,r.vec_dst)
        ambiguous=float(r.input_ops or 0)>1
        records.append(dict(node=r.node,src=r.src,dst=r.dst,op=r.op,guess=r.guess,FP=int(r.prediction),N=n,
                            pair_unseen=n==0,operation_unseen=n>0 and r.n==0,within_pair_nonmode=0<r.n<r.maximum,
                            low_seen_pair=0<n<100,pair_mode=r.n>0 and r.n==r.maximum,sole_true=len(hist)==1 and r.n>0,
                            seen_pair_new_input=n>0 and not byte_match,global_input_ambiguous=ambiguous,
                            same_input_unambiguous_wrong=byte_match and wrong and float(r.input_ops or 0)==1 and float(r.input_true_n or 0)>0,
                            same_pair_input=byte_match))
    nodes=pd.DataFrame(records)
    rates=nodes.groupby('FP').sum(numeric_only=True).T
    rates.to_csv(out/'mechanism_counts.csv')
    rows=[]
    for r in qs.itertuples():
        rec=dict(node=r.node,batch=r.batch,row=r.row,canonical=r.canonical,FP=r.prediction)
        for m in ['feature','same_src','same_dst']:
            nc,err,true,high=evidence(r,r.guess,m)
            rec.update({m+'_neighbors':nc,m+'_guess_n':err,m+'_true_n':true,m+'_hf_mode':high})
        rows.append(rec)
    markers=pd.DataFrame(rows)
    fps=nodes[nodes.FP==1].merge(markers[markers.canonical],on=['node','FP'],validate='one_to_one')
    assert len(fps)==172
    fps['train_accuracy']=[count['train',(s,d)]['right']/n if n else np.nan for s,d,n in zip(fps.src,fps.dst,fps.N)]
    fps['test_accuracy']=[count['test',(s,d)]['right']/count['test',(s,d)]['n'] for s,d in zip(fps.src,fps.dst)]
    fps['unresolved']=~(fps.pair_unseen|fps.operation_unseen|fps.within_pair_nonmode|fps.low_seen_pair|fps.feature_hf_mode|fps.seen_pair_new_input|fps.global_input_ambiguous|fps.same_input_unambiguous_wrong)
    fps.to_csv(out/'ALL_FP_EVIDENCE.csv',index=False)
    markers.to_csv(out/'all_query_evidence.csv',index=False)
    canonical={r.node:r for r in qs[qs.canonical].itertuples()}
    contrasts=[]
    for a,b in pd.read_csv(first/'matched.csv').itertuples(index=False,name=None):
        fp,tn=canonical[a],canonical[b]
        assert fp.op==tn.op
        for method in ('feature','same_src','same_dst'):
            ef=evidence(fp,fp.guess,method);et=evidence(tn,fp.guess,method)
            contrasts.append((a,b,fp.op,fp.guess,method,*ef,*et))
    paired=pd.DataFrame(contrasts,columns='fp_node tn_node real competitor method fp_neighbors fp_other fp_true fp_high tn_neighbors tn_other tn_true tn_high'.split())
    paired.to_csv(out/'matched_contrasts.csv',index=False)
    summary=[]
    for m,g in paired.groupby('method'):
        g=g[(g.fp_neighbors>0)&(g.tn_neighbors>0)]
        mean=g.groupby('fp_node')[['fp_high','tn_high']].mean()
        summary.append(dict(method=m,fp_sets=len(mean),unique_tn=g.tn_node.nunique(),FP_high=mean.fp_high.mean(),TN_high=mean.tn_high.mean(),fp_only=int(((mean.fp_high==1)&(mean.tn_high==0)).sum()),tn_only=int(((mean.fp_high==0)&(mean.tn_high==1)).sum())))
    (out/'summary.json').write_text(json.dumps(dict(train_batches_replayed=batches,test_query_batches=int(qs.batch.nunique()),max_test_loss_difference=maxdiff,matched_summary=summary),indent=2,default=lambda x:x.item()))
if __name__=='__main__':main(*map(Path,sys.argv[1:]),Path(os.environ['RUN_OUTPUT_DIR']))
