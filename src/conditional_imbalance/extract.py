"""Audit native training and replay the frozen Velox decoder, one scored batch at a time."""
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from yacs.config import CfgNode

run = Path(sys.argv[1]).resolve()
out = Path(os.environ['RUN_OUTPUT_DIR'])
cell = run / 'output/CADETS_E3-velox-full'
verify = run / 'output/verify'
sys.path[:0] = [str(run/'code/src'), str(run/'code')]
from baseline_matrix import digest, load_evaluation_checkpoint
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import get_device, set_seed
from pidsmaker.factory import build_model


def read(path):
    return json.loads(path.read_text())


def save(name, obj):
    (out/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def table(name, header, rows):
    with (out/name).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def bucket(n, m):
    if not m:
        return 'unseen_condition'
    if not n:
        return 'unseen_operation'
    for scale, label in [(100, 'lt01'), (10, 'lt10'), (2, 'lt50'), (1, 'lt100')]:
        if scale*n < m:
            return label
    return 'mode'


manifest = read(cell/'manifest.json')
summary = read(verify/'summary.json')
report = read(cell/'knee/report.json')
ck = read(cell/'checkpoint.json')
threshold = report['policy']['threshold']
corrected = Path(summary['default_predictions'])
assert corrected == verify/'corrected_export/knee/objects.jsonl'
assert digest(cell/'resolved_config.yml') == manifest['config_sha256']
assert digest(ck['path']) == report['identity']['checkpoint_sha256'] == ck['sha256']
assert manifest['commit'] == subprocess.check_output(['git','-C',str(run/'code'),'rev-parse','HEAD'],text=True).strip()
assert run.name == 'cadets-velox-firstpaths-rebuild-20261126b'
assert report['scope'] == 'retrospective_test' and read(corrected.parent/'report.json')['policy'] == report['policy']
original = cell/'knee/objects.jsonl'
same_bytes = digest(original) == digest(corrected)
def predictions(path):
    result = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            n = r['node_id']
            assert n not in result and r['threshold'] == threshold and r['prediction'] == int(r['score'] > threshold)
            result[n] = (r['uuid'], r['gt'], r['score'], r['prediction'])
    return result
pred = predictions(corrected)
if not same_bytes:
    assert predictions(original) == pred
counts = Counter((g,p) for _,g,_,p in pred.values())
assert len(pred) == 286930 and counts[0,1] == 172 and counts[0,0] == 286686
protocol = dict(run=run.name, detector_commit=manifest['commit'], analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config_sha256=manifest['config_sha256'], checkpoint=ck, seed=manifest['seed'], dates=manifest['protocol'], coverage=summary['coverage'],
                threshold=threshold, scope=report['scope'], predictions=str(corrected), original_identical_bytes=same_bytes,
                prediction_sha256=digest(corrected), nodes=len(pred), FP=counts[0,1], TN=counts[0,0], TP=counts[1,1], FN=counts[1,0])
save('protocol.json',protocol)
table('nodes.csv',['node','uuid','gt','score','prediction'],((k,*v) for k,v in pred.items()))
print('Prediction audit saved',flush=True)
cfg = CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
assert cfg.training.encoder.used_methods == 'none' and not cfg.training.encoder.x_is_tuple
device = get_device(cfg)
set_seed(cfg,seed=0)
train,val,test,max_nodes = get_preprocessed_graphs(cfg)
model = build_model(train[0][0],device=device,cfg=cfg,max_node_num=max_nodes,all_data=(val,test))
load_evaluation_checkpoint(model,ck['path'])
decoder = next(m for m in model.modules() if type(m).__name__ == 'CustomEdgeMLP')
capture = []
def projection_input(module, args):
    capture.append(args[0].detach())
encoder = next(m for m in model.modules() if type(m).__name__ == 'LinearEncoder')
encoder.lin1.register_forward_pre_hook(projection_input)
encoder.lin1.register_forward_hook(lambda module, args, output: capture.append(output.detach()))
decoder.lin_src.register_forward_pre_hook(projection_input)
decoder.lin_dst.register_forward_pre_hook(projection_input)
decoder.register_forward_hook(lambda module, args, output: capture.append(output.detach()))
ops = get_rel2id(cfg,from_zero=True)
windows = [json.loads(line) for line in (cell/'windows.jsonl').open()]
batch_size = cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
losses = sorted((Path(cfg.training._edge_losses_dir)/'test'/('model_epoch_'+str(cfg.training.num_epochs-1))).glob('*.csv'),key=lambda p:int(p.stem.rsplit('_b',1)[1]))
assert [int(p.stem.rsplit('_b',1)[1]) for p in losses] == list(range(len(losses)))
vectors = {}
hist = defaultdict(Counter)
exact = defaultdict(Counter)
label_inputs = defaultdict(set)
global_freq = Counter()
def vector_id(array):
    key = array.tobytes()
    return vectors.setdefault(key,len(vectors))
event_path = out/'test_events.csv.gz'
for split, dataset in [('train',train),('test',test)]:
    assert len(dataset) == 1
    batches = dataset[0]
    ledger = sorted((x for x in windows if x['split']==split),key=lambda x:x['window'])
    bi = total = 0
    max_error = 0
    if split == 'test':
        stream = gzip.open(event_path,'wt',compresslevel=1)
        writer = csv.writer(stream)
        writer.writerow('batch row u v event_uuid y src dst op loss guess p_true p_guess vec_src vec_dst'.split())
    for window in ledger:
        name = window['window']
        graph = torch.load(Path(cfg.transformation._graphs_dir)/('graph_'+window['date'])/name,map_location='cpu')
        edges = iter(graph.edges(data=True))
        labels = {int(k):v['label'] for k,v in graph.nodes(data=True)}
        assert graph.number_of_edges()==window['edges']
        for start in range(0,window['edges'],batch_size):
            b = batches[bi]
            u,v = b.original_edge_index.cpu().tolist()
            truth = b.edge_type.argmax(1).cpu().tolist()
            times = b.t.cpu().tolist()
            flags = b.y.cpu().reshape(-1).tolist()
            assert len(u)==min(window['edges']-start,batch_size)
            b = b.to(device)
            src,dst = b.edge_index
            features = b.x.detach().cpu().numpy()
            ids = [vector_id(x) for x in features]
            pairs = [(ids[i],ids[j]) for i,j in zip(src.cpu().tolist(),dst.cpu().tolist())]
            if split == 'test':
                frame = pd.read_csv(losses[bi],float_precision='round_trip')
                assert list(frame.srcnode)==u and list(frame.dstnode)==v and list(frame.time)==times and list(frame.edge_type)==[x+1 for x in truth]
                with torch.no_grad():
                    capture.clear()
                    replay = model(b,inference=True,validation=False)['loss'].reshape(-1)
                    encoder_input,encoded,hs,hd,logits = capture
                    assert torch.equal(encoder_input,b.x)
                    assert torch.equal(hs,encoded[src]) and torch.equal(hd,encoded[dst])
                    prob = logits.softmax(1)
                    guesses = prob.argmax(1).cpu().tolist()
                    true_probs = prob.gather(1,b.edge_type.argmax(1)[:,None]).flatten().cpu().tolist()
                    top_probs = prob.max(1).values.cpu().tolist()
                error = float(np.max(np.abs(replay.cpu().numpy()-frame.loss.values)))
                assert error<1e-4,(bi,error)
                max_error=max(max_error,error)
                loss = frame.loss.tolist()
            for j,(a,z,opid,pair) in enumerate(zip(u,v,truth,pairs)):
                gu,gv,e = next(edges)
                op = ops[opid]
                assert (int(gu),int(gv),e['label'],int(e['time']),int(e['y'])) == (a,z,op,times[j],flags[j])
                c = labels[a],labels[z]
                if split == 'train':
                    assert flags[j]==0
                    hist[c][op]+=1
                    exact[pair][op]+=1
                    label_inputs[c].add(pair)
                    global_freq[op]+=1
                else:
                    writer.writerow((bi,j,a,z,e['event_uuid'],flags[j],*c,op,loss[j],ops[guesses[j]],true_probs[j],top_probs[j],*pair))
            total+=len(u)
            bi+=1
        assert next(edges,None) is None
        if bi%100<5:
            print(split,name,bi,total,flush=True)
    assert bi==len(batches)==summary['coverage'][split]['batches'] and total==summary['coverage'][split]['edges']
    if split=='train':
        rows=[]
        distribution=Counter()
        for (s,d),h in hist.items():
            m=max(h.values())
            modes='|'.join(sorted(o for o,n in h.items() if n==m))
            for o,n in h.items():
                rows.append((s,d,o,n,sum(h.values()),m,modes,n/sum(h.values()),n/m,len(h),global_freq[o],len(label_inputs[s,d])))
                if len(h)>1:
                    distribution[bucket(n,m)]+=n
        table('condition_operations.csv','src dst op n total maximum modes p q operations global_n exact_pairs'.split(),rows)
        table('input_operations.csv',['vec_src','vec_dst','op','n'],((*pair,o,n) for pair,h in exact.items() for o,n in h.items()))
        save('training.json',dict(windows=len(ledger),batches=bi,edges=total,conditions=len(hist),single=sum(len(h)==1 for h in hist.values()),
             single_edges=sum(sum(h.values()) for h in hist.values() if len(h)==1),multi_distribution=distribution,global_operations=global_freq,
             conditions_with_multiple_final_inputs=sum(len(p)>1 for p in label_inputs.values()),exact_pairs=len(exact),
             multi_operation_exact_pairs=sum(len(h)>1 for h in exact.values()),multi_operation_exact_edges=sum(sum(h.values()) for h in exact.values() if len(h)>1)))
    else:
        stream.close()
        save('replay.json',dict(windows=len(ledger),batches=bi,edges=total,max_loss_error=max_error))
    print('Saved',split,flush=True)
