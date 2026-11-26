"""Replay fixed-budget seed 0 and verify final parameters before exporting Adam moments."""
import json
import os
from pathlib import Path
import sys

import torch
from neighbors import loader


def main(run, out):
    cell,cfg,_,_,(train,val,test,maxnodes)=loader(run)
    from pidsmaker.factory import build_model,optimizer_factory
    from pidsmaker.detection.training_methods.training_loop import remove_attacks_if_needed
    from pidsmaker.utils.utils import get_device,set_seed
    set_seed(cfg,seed=cfg.training.seed)
    device=get_device(cfg)
    model=build_model(train[0][0],device=device,cfg=cfg,max_node_num=maxnodes,all_data=(val,test))
    optimizer=optimizer_factory(cfg,set(model.parameters()))
    assert cfg.training.grad_accumulation==1 and not cfg.training.stable_optim
    assert cfg.evaluation.best_model_selection=="fixed_budget" and cfg.training.num_epochs==12
    for epoch in range(12):
        model.reset_state()
        model.to_fine_tuning(False)
        for b in train[0]:
            b=b.to(device)
            g=remove_attacks_if_needed(b,cfg)
            model.train()
            optimizer.zero_grad()
            model(g)["loss"].backward()
            optimizer.step()
            b.to("cpu")
        print("finished epoch",epoch,flush=True)
    model.eval()
    saved=torch.load(json.loads((cell/"checkpoint.json").read_text())["path"],map_location="cpu")
    diffs={n:(p.cpu()-saved[n]).abs().max().item() for n,p in model.named_parameters()}
    (out/"summary.json").write_text(json.dumps(diffs,indent=2))
    assert max(diffs.values())==0,diffs
    torch.save({n:optimizer.state[p] for n,p in model.named_parameters()},out/"adam.pt")


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(os.environ["RUN_OUTPUT_DIR"]))
