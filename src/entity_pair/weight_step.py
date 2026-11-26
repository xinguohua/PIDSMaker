"""Paired Adam interventions with fixed joint denominator and frozen native inputs."""
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from neighbors import loader


def main(root, plan_dir, replay_dir, out):
    run=root/"cadets-velox-firstpaths-rebuild-20261126b"
    cell,cfg,labels,ops,(train,val,test,maxnodes)=loader(run)
    from pidsmaker.factory import build_model, optimizer_factory
    from baseline_matrix import load_evaluation_checkpoint
    from pidsmaker.utils.utils import get_device,set_seed
    set_seed(cfg,seed=0)
    device=get_device(cfg)
    model=build_model(train[0][0],device=device,cfg=cfg,max_node_num=maxnodes,all_data=(val,test))
    checkpoint=json.loads((cell/"checkpoint.json").read_text())["path"]
    load_evaluation_checkpoint(model,checkpoint)
    initial=copy.deepcopy(model.state_dict())
    optimizer=optimizer_factory(cfg,model.parameters())
    moments=torch.load(replay_dir/"adam.pt",map_location=device)
    for n,p in model.named_parameters():
        optimizer.state[p]=moments[n]
    initial_adam=copy.deepcopy(optimizer.state_dict())
    plan=pd.read_pickle(plan_dir/"plan.pkl")
    eff=torch.load(plan_dir/"effective.pt",map_location=device)
    keys,groups,ids=plan["keys"],plan["groups"],plan["batch_ids"]
    obj=next(m for m in model.modules() if type(m).__name__=="EdgeTypePrediction")
    encoder=next(m for m in model.modules() if type(m).__name__=="LinearEncoder")
    # Store endpoint features from reindexed native batches. The native encoder and
    # decoder act independently on each endpoint pair in eval mode.
    from collections import defaultdict
    pieces=defaultdict(list)
    mass=defaultdict(float)
    used={j for i,g in groups.items() for j in [i,*g["C"]]}
    for b,bids in zip(train[0],ids):
        for j in set(bids)&used:
            ix=np.flatnonzero(bids==j)
            a,z=b.edge_index[:,ix]
            pieces[j].append((b.x[a],b.x[z],b.edge_type[ix].argmax(1)))
            mass[j]+=len(ix)/len(bids)
    def measure(js,alt):
        total=torch.zeros(4,device=device)
        for j in js:
            for a,z,y in pieces[j]:
                y=y.to(device)
                logits=obj.decoder(encoder(a.to(device))["h"],encoder(z.to(device))["h"])
                pr=logits.softmax(1)
                total+=torch.stack([obj.loss_fn(logits,y,inference=True).sum(),
                    pr.gather(1,y[:,None]).sum(),pr[:,alt].sum(),torch.ones_like(y).sum()])
        return (total[:3]/total[3]).tolist()
    rows=[]
    with torch.no_grad():
        for i,g in groups.items():
            competition=g["C"]
            if not competition:continue
            model.load_state_dict(initial)
            # Most common competitor operation differing from the target mode.
            counts=lambda js:torch.bincount(torch.cat([y for j in js for _,_,y in pieces[j]]),minlength=obj.edge_type_dim)
            h=counts(competition); h[counts([i]).argmax()]=-1
            alt=int(h.argmax())
            before=measure([i],alt),measure(competition,alt)
            for w in (0,.25,.5,1,2):
                model.load_state_dict(initial)
                optimizer.load_state_dict(copy.deepcopy(initial_adam))
                v=(eff[i]+w*sum(eff[j] for j in competition))/(mass[i]+sum(mass[j] for j in competition))
                pos=0
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad=v[pos:pos+p.numel()].reshape_as(p)
                        pos+=p.numel()
                assert pos==v.numel()
                optimizer.step()
                after=measure([i],alt),measure(competition,alt)
                for group,b,a in zip(("T","C"),before,after):
                    rows.append(dict(src=keys[i][0],dst=keys[i][1],group=group,weight=w,alternative=ops[alt],
                        base_loss=b[0],loss_delta=a[0]-b[0],true_p_delta=a[1]-b[1],alternative_p_delta=a[2]-b[2]))
            pd.DataFrame(rows).to_csv(out/"weight_steps.csv",index=False)


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]),Path(os.environ["RUN_OUTPUT_DIR"]))
