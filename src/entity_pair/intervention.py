"""Seed-zero four-arm paired intervention on training-supported FP target pairs."""
from io import StringIO
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from neighbors import loader


def main(root, plan_dir, out):
    source=root/"velox-condition-firstpaths-20261126b/output"
    run=root/"cadets-velox-firstpaths-rebuild-20261126b"
    cell,cfg,labels,ops,(train,val,test,maxnodes)=loader(run)
    from pidsmaker.factory import build_model,optimizer_factory
    from pidsmaker.utils.utils import set_seed,get_device
    from baseline_matrix import load_evaluation_checkpoint
    plan=pd.read_pickle(plan_dir/"plan.pkl")
    eff=torch.load(plan_dir/"effective.pt",map_location="cpu")
    keys=plan["keys"]; index={k:i for i,k in enumerate(keys)}
    fp=pd.read_pickle(root/"pair-interference-s1-20261126a/output/queries.pkl").query("canonical and prediction==1")
    targets={index[k] for k in zip(fp.src,fp.dst) if k in index}
    rivals={j for i in targets for j in plan["groups"][i]["C"] if torch.dot(eff[i],eff[j])<0}-targets
    assert targets and rivals
    device=get_device(cfg)
    rng=np.random.default_rng(0)
    weights={arm:[] for arm in ("competitor","random","target")}
    changed=0
    for b,ids in zip(train[0],plan["batch_ids"]):
        assert not b.y.any()
        t=np.isin(ids,list(targets)); c=np.isin(ids,list(rivals))
        eligible=np.flatnonzero(~(t|c)); selected=np.flatnonzero(c)
        # Equal numbers of competitor and random edges per batch and operation.
        selected_c=[];selected_r=[]
        op=b.edge_type.argmax(1).numpy()
        for k in np.unique(op[selected]):
            cand=selected[op[selected]==k]; controls=eligible[op[eligible]==k]
            n=min(len(cand),len(controls))
            selected_c.extend(rng.choice(cand,n,replace=False))
            selected_r.extend(rng.choice(controls,n,replace=False))
        reserve=np.setdiff1d(eligible,selected_r)
        for arm,ix in (("competitor",selected_c),("random",selected_r),("target",np.flatnonzero(t))):
            w=np.ones(len(ids),np.float32)
            if len(ix) and len(reserve):
                delta=min(1,len(reserve)/len(ix)*.5) if arm=="target" else -.5
                w[ix]+=delta
                w[reserve]-=delta*len(ix)/len(reserve)
                if arm=="competitor":changed+=len(ix)
            assert abs(w.sum(dtype=float)-len(ids))<1e-3 and np.all(w>=0)
            weights[arm].append(torch.from_numpy(w))
    protocol=json.loads((source/"protocol.json").read_text())
    nodes=pd.read_csv(source/"nodes.csv").set_index("node")
    rows=[]
    for arm in ("original","competitor","random","target"):
        set_seed(cfg,seed=0)
        model=build_model(train[0][0],device=device,cfg=cfg,max_node_num=maxnodes,all_data=(val,test))
        if arm=="original":
            load_evaluation_checkpoint(model,protocol["checkpoint"]["path"])
        else:
            opt=optimizer_factory(cfg,set(model.parameters()))
            obj=next(m for m in model.modules() if type(m).__name__=="EdgeTypePrediction")
            native=obj.loss_fn
            def weighted(x,y,**kw):
                return (native(x,y,inference=True,weight=kw["weight"])*weight).mean()
            obj.loss_fn=weighted
            for epoch in range(12):
                model.reset_state()
                model.to_fine_tuning(False)
                model.train()
                for b,w in zip(train[0],weights[arm]):
                    weight=w.to(device)
                    b=b.to(device)
                    opt.zero_grad()
                    model(b)["loss"].backward()
                    opt.step()
                    b.to("cpu")
                print(arm,epoch,flush=True)
            obj.loss_fn=native
        model.eval()
        score=np.full(maxnodes+1,-np.inf)
        tr_right=tr_count=0
        capture=[]
        dec=next(m for m in model.modules() if type(m).__name__=="CustomEdgeMLP")
        hook=dec.register_forward_hook(lambda m,a,o:capture.append(o))
        with torch.no_grad():
            for b,ids in zip(train[0],plan["batch_ids"]):
                ix=np.flatnonzero(np.isin(ids,list(targets)))
                if len(ix):
                    b=b.to(device)
                    model(b,inference=True)
                    tr_right+=int((capture.pop().argmax(1)[ix]==b.edge_type.argmax(1)[ix]).sum())
                    tr_count+=len(ix)
                    b.to("cpu")
            hook.remove()
            for b in test[0]:
                b=b.to(device)
                loss=model(b,inference=True)["loss"].flatten().cpu().numpy()
                for ids in b.original_edge_index.cpu().numpy():
                    np.maximum.at(score,ids,loss)
                b.to("cpu")
        # Native inference writes float64 edge scores to CSV and node evaluation
        # reads with pandas' default parser before the strict threshold comparison.
        values,inverse=np.unique(score[nodes.index],return_inverse=True)
        parsed=pd.read_csv(StringIO(pd.DataFrame({"loss":values}).to_csv(index=False))).loss.to_numpy()
        actual=parsed[inverse]
        pred=actual>protocol["threshold"]
        if arm=="original":
            bad=pred!=nodes.prediction.to_numpy()
            print("original audit",sum(bad),np.max(abs(actual-nodes.score)),flush=True)
            if bad.any():
                print(nodes.loc[bad].assign(replayed=actual[bad]).head().to_string(),flush=True)
            assert not bad.any()
        gt=nodes["gt"].to_numpy()
        rows.append(dict(arm=arm,FP=int(sum(pred&(gt==0))),TP=int(sum(pred&(gt==1))),
            TN=int(sum(~pred&(gt==0))),FN=int(sum(~pred&(gt==1))),target_right=tr_right,target_n=tr_count,
            original_fp_resolved=int(sum(~pred&(nodes.prediction==1)&(gt==0))),
            new_fp=int(sum(pred&(nodes.prediction==0)&(gt==0))),changed_competitor_edges=changed))
        pd.DataFrame(rows).to_csv(out/"intervention.csv",index=False)


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]),Path(os.environ["RUN_OUTPUT_DIR"]))
