"""Frozen masked reconstruction. Signatures use only the masked encoder/decoder inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from yacs.config import CfgNode


def digest_bytes(value):
    return hashlib.blake2b(value, digest_size=16).digest()


def signatures(x, masked, edges, eps):
    # Unique leaf bytes and an explicit mask bit. Each round preserves incoming multiplicities.
    values = np.floor(x / eps).astype('int32') if eps else x
    leaf = np.column_stack((masked.astype(values.dtype), values))
    unique, inv = np.unique(leaf, axis=0, return_inverse=True)
    code = np.asarray([digest_bytes(row.tobytes()) for row in unique], dtype='V16')[inv]
    u, v = edges
    offsets = np.r_[0, np.cumsum(np.bincount(v, minlength=len(x)))]
    for _ in range(4):
        by_dest = np.lexsort((code[u].view('S16'), v))
        messages = code[u[by_dest]].tobytes()
        code = np.asarray([digest_bytes(c.tobytes() + messages[16*a:16*b])
                           for c,a,b in zip(code, offsets[:-1], offsets[1:])], dtype='V16')
    return code


def main(cell, prior, out):
    base = cell.parents[1]/'code'
    sys.path[:0] = [str(base), str(base/'src')]
    from baseline_matrix import digest, load_evaluation_checkpoint
    from pidsmaker.factory import build_model
    from pidsmaker.tasks.batching import get_preprocessed_graphs
    from pidsmaker.utils.replay_state import restore_replay_state
    from pidsmaker.utils.utils import get_device, set_seed
    cfg = CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    audit = json.loads((prior/'summary.json').read_text())
    ck = json.loads((cell/'checkpoint.json').read_text())
    assert digest(ck['path']) == audit['checkpoint'] and digest(cell/'knee/objects.jsonl') == audit['knee_sha256']
    train,val,test,maxnodes = get_preprocessed_graphs(cfg)
    device = get_device(cfg)
    set_seed(cfg, seed=0)
    model = build_model(train[0][0], device=device, cfg=cfg, max_node_num=maxnodes, all_data=(val,test))
    load_evaluation_checkpoint(model, ck['path'])
    restore_replay_state(model, Path(ck['path']).parent/'replay_state.pt', required=True)
    feat = next(m for m in model.modules() if type(m).__name__ == 'GMAEFeatReconstruction')
    assert feat.mask_rate == .5 and not model.training
    # Calibration uses visible training coordinates sampled independently of targets and outputs.
    samples=[]
    for w,b in enumerate(train[0]):
        x=b.x.cpu().numpy()
        perm=np.random.default_rng(200000+w).permutation(len(x))
        samples.extend(np.abs(x[perm[len(x)//2:][:100]]).ravel())
    a=np.asarray(samples);scale=float(np.quantile(a[a>0],.90))
    levels=[0.,.01*scale,.05*scale]
    (out/'identity.json').write_text(json.dumps(dict(previous=audit,levels=levels,scale=scale,
        mask_token=feat.mask_token.detach().cpu().tolist(),sample_seeds=[101,202]),indent=2))
    for split, batches in [('train',train[0]),('test',test[0])]:
        for repeat,seed in enumerate([101,202] if split=='train' else [303]):
            dest=out/f'{split}{repeat}';dest.mkdir()
            for w,batch in enumerate(batches):
                b=batch.to(device)
                full=b.x
                ids=b.original_n_id.cpu().numpy().ravel()
                with np.load(prior/split/f'{w:04d}.npz') as old:
                    assert np.array_equal(ids,old['id']) and np.array_equal(full.cpu().numpy(),old['x'])
                    n=len(ids); m=n//2
                    torch.manual_seed(seed*1000+w)
                    masked,idx=feat.mask_input(full)
                    masks=[idx]
                    if split=='test':
                        # Native-sized masks conditioned on including each test root once.
                        left=torch.ones(n,dtype=torch.bool,device=device);left[idx]=False
                        rest=torch.where(left)[0]
                        rest=rest[torch.randperm(len(rest),device=device)]
                        masks.append(rest[:m])
                        if n%2:masks.append(torch.cat((rest[-1:],idx[:m-1])))
                    edges=b.edge_index.cpu().numpy()
                    for draw,indices in enumerate(masks):
                        selected=indices if draw<2 else indices[:1]
                        rows=selected.cpu().numpy()
                        flags=np.zeros(n,dtype=bool);flags[indices.cpu().numpy()]=True
                        with torch.no_grad():
                            masked=full.clone();masked[indices]=feat.mask_token
                            b.x=masked
                            h=model.embed(b,inference=True)[0]
                            b.x=full
                            pred=feat.decoder(h,b.edge_index,edge_feats=getattr(b,'edge_feats',None))[selected]
                            loss=feat.loss_fn(pred,full[selected],inference=True)
                        xx=masked.detach().cpu().numpy()
                        keys={f'g{i}':signatures(xx,flags,edges,eps)[rows] for i,eps in enumerate(levels)}
                        # Lightweight continuous visible-input descriptor only for comparisons within certified cells.
                        moment=xx.astype('float64')
                        u,v=edges
                        indeg=np.bincount(v,minlength=n)
                        for depth in range(4):
                            msg=np.zeros_like(moment);np.add.at(msg,v,moment[u])
                            moment=(moment+msg)/(1+indeg[:,None])
                        np.savez_compressed(dest/f'{w:04d}-{draw}.npz',**keys,id=ids[rows],row=rows,
                            target=full[selected].cpu().numpy(),output=pred.cpu().numpy(),loss=loss.cpu().numpy(),
                            moment=moment[rows],indeg=indeg[rows],kind=old['kind'][rows],size=n)
                if w%100==0:print(split,repeat,w,flush=True)
    print('completed',flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),Path(os.environ['RUN_OUTPUT_DIR']))
