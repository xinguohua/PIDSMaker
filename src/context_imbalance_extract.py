"""Extract verified context counts and test edges from the frozen first-path Velox run."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from yacs.config import CfgNode
from velox_training_distribution.extract_native import digest

THRESHOLD = 5.9490790367126465
KEY = ['op', 'src_label', 'dst_label']


def extract(cell, out):
    run = cell.parents[1]
    meta = json.loads((cell/'manifest.json').read_text())
    assert meta['task_id'] == run.name == 'cadets-velox-firstpaths-rebuild-20261126b'
    assert meta['commit'] == subprocess.check_output(['git','-C',str(run/'code'),'rev-parse','HEAD'], text=True).strip()
    assert digest(cell/'resolved_config.yml') == meta['config_sha256']
    sys.path.insert(0, str(run/'code'))
    from pidsmaker.utils.dataset_utils import get_rel2id
    from pidsmaker.utils.feature_cache import load_feature_data

    cfg = CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    corrected = cell.parent/'verify/corrected_export/knee'
    h = digest(corrected/'objects.jsonl')
    assert digest(cell/'knee/objects.jsonl') == h
    report = json.loads((corrected/'report.json').read_text())
    assert report['policy']['threshold'] == THRESHOLD and report['scope'] == 'retrospective_test'
    p = pd.DataFrame(json.loads(line) for line in (corrected/'objects.jsonl').open()).set_index('node_id')
    assert p.index.is_unique and (p.prediction == (p.score > THRESHOLD)).all()
    assert (p.threshold == THRESHOLD).all() and (p.scope == 'retrospective_test').all()
    normal = p[p['gt'] == 0]
    assert normal.prediction.value_counts().to_dict() == {0:286686,1:172}
    p.to_pickle(out/'pred.pkl')
    audit = dict(run=run.name,commit=meta['commit'],config_sha256=meta['config_sha256'],seed=meta['seed'],
                 checkpoint=report['identity'],objects_sha256=h,threshold=THRESHOLD,scope=report['scope'],
                 FP=172,TN=286686,dates=meta['protocol'],splits={})
    def save():
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
    save()
    coverage = json.loads((cell.parent/'verify/coverage.json').read_text())
    opids = get_rel2id(cfg)
    width = cfg.dataset.num_node_types + cfg.featurization.emb_dim
    batch_size = cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
    assert cfg.transformation.used_methods == 'none' and cfg.batching.intra_graph_batching.used_methods == 'edges'
    counts = Counter()
    for split in ['train','test']:
        tensors = sorted((Path(cfg.feat_inference._edge_embeds_dir)/split).iterdir())
        batch_id = edges = 0
        if split == 'test':
            lossdir = Path(cfg.training._edge_losses_dir)/'test/model_epoch_11'
            losses = sorted(lossdir.glob('*.csv'),key=lambda f:int(f.stem.rsplit('_b',1)[1]))
            assert [int(f.stem.rsplit('_b',1)[1]) for f in losses] == list(range(len(losses)))
            (out/'events').mkdir(exist_ok=True)
        windows = []
        for w,tensor in enumerate(tensors):
            name = tensor.name.removesuffix('.TemporalData.simple')
            assert name[:10] in meta['protocol'][split+'_dates']
            g = torch.load(Path(cfg.transformation._graphs_dir)/('graph_'+name[:10])/name,map_location='cpu')
            data = load_feature_data(tensor).to('cpu')
            rows = [(int(u),int(v),int(a['time']),a['label'],a['y'],a['event_uuid'],
                     g.nodes[u]['label'],g.nodes[v]['label'],g.nodes[u]['node_type'],g.nodes[v]['node_type'])
                    for u,v,a in g.edges(data=True)]
            df = pd.DataFrame(rows,columns=['u','v','ns','op','y','uuid','src_label','dst_label','src_type','dst_type'])
            ids = df.op.map(opids)
            for ar,ten in [(df.u,data.src),(df.v,data.dst),(df.ns,data.t),(df.y,data.y),
                          (ids,data.msg[:,width:width+cfg.dataset.num_edge_types].argmax(1)+1)]:
                assert np.array_equal(ar,ten.numpy().reshape(-1)), (w,name)
            batches = (len(df)+batch_size-1)//batch_size
            if split == 'train':
                assert (df.y == 0).all()
                counts.update(df.groupby(KEY,dropna=False).size().to_dict())
            else:
                csvs = [pd.read_csv(f) for f in losses[batch_id:batch_id+batches]]
                lf = pd.concat(csvs,ignore_index=True)
                for left,right in [(df.u,lf.srcnode),(df.v,lf.dstnode),(ids,lf.edge_type),(df.ns,lf.time)]:
                    assert np.array_equal(left,right), (w,name)
                assert np.isfinite(lf.loss).all()
                df['loss'] = lf.loss
                df.to_pickle(out/'events'/f'{w:04d}.pkl.gz')
            windows.append((w,name,batch_id,batches,len(df)))
            batch_id += batches
            edges += len(df)
        assert edges == coverage[split]['edges'] and batch_id == coverage[split]['batches']
        pd.DataFrame(windows,columns=['w','window','first_batch','batches','edges']).to_csv(out/f'{split}_windows.csv',index=False)
        audit['splits'][split] = dict(windows=len(windows),edges=edges,batches=batch_id,batch_size=batch_size)
        if split == 'train':
            tr = pd.DataFrame([(*k,v) for k,v in counts.items()],columns=KEY+['N'])
            tr['P'] = tr.N/tr.groupby('op').N.transform('sum')
            tr['Qc'] = tr.N/tr.groupby('op').N.transform('max')
            tr.sort_values(['op','N'],ascending=[True,False]).to_csv(out/'training.csv.gz',index=False)
        save()
        print(split,audit['splits'][split],flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cell',type=Path)
    args = parser.parse_args()
    extract(args.cell,Path(os.environ['RUN_OUTPUT_DIR']))
