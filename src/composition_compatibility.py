"""Verify first-path native identities against the original path extraction, read-only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from action_composition import ro, sha

THRESHOLD = 5.9490790367126465


def predictions(cell):
    path = cell / 'knee/objects.jsonl'
    corrected = cell.parent / 'verify/corrected_export/knee/objects.jsonl'
    digest = sha(path)
    assert digest == sha(corrected)
    metrics = Counter()
    labels = {}
    for line in path.open():
        r = json.loads(line)
        n = r['node_id']
        assert n not in labels and r['scope'] == 'retrospective_test'
        assert r['threshold'] == THRESHOLD and r['prediction'] == (r['score'] > THRESHOLD)
        labels[n] = r['gt']
        metrics[['TN', 'FP', 'FN', 'TP'][2*r['gt'] + r['prediction']]] += 1
    assert metrics == dict(TN=286686, FP=172, TP=35, FN=37)
    return labels, dict(sha256=digest, threshold=THRESHOLD, scope='retrospective_test', metrics=metrics)


def audit(cell, old, out):
    import torch
    import yaml
    sys.path.insert(0, str(cell.parents[1] / 'code'))
    from pidsmaker.utils.feature_cache import load_feature_data
    pred, report = predictions(cell)
    db = ro(old / 'native.sqlite')
    db.execute('attach database ? as sem', ((old/'semantic.sqlite').as_uri()+'?mode=ro',))
    cfg = yaml.safe_load((cell/'resolved_config.yml').read_text())
    mapping = json.loads((cell/'uuid_mapping.json').read_text())
    nodes = {n:(kind,label) for n,kind,label,uuid in db.execute('select * from nodes')
             if mapping[str(n)] == uuid}
    assert len(nodes) == db.execute('select count(*) from nodes').fetchone()[0]
    assert pred == dict(db.execute('select id,gt from pred'))
    changed = {}
    totals = Counter()
    errors = Counter()
    examples = {}
    summary = dict(predictions=report, mapped_nodes=len(nodes), checked=totals,
                   discrepancies=errors, examples=examples)
    for split in ('train','test'):
        fs = sorted((Path(cfg['feat_inference']['_edge_embeds_dir'])/split).iterdir())
        windows = list(db.execute('select w,name,n from windows where split=? order by w',(split,)))
        assert [p.name.removesuffix('.TemporalData.simple') for p in fs] == [name for _,name,_ in windows]
        for (w,name,n), fp in zip(windows,fs):
            graph = Path(cfg['transformation']['_graphs_dir'])/('graph_'+name[:10])/name
            g = torch.load(graph,map_location='cpu')
            d = load_feature_data(fp)
            src,dst,t,y = (z.tolist() for z in (d.src,d.dst,d.t,d.y))
            assert len(src)==len(dst)==len(t)==len(y)==g.number_of_edges()==n
            q = db.execute('select e.row,e.u,e.v,e.ns,e.op,e.uuid,e.y,s.seq from edges e join sem.source s using(w,row) where w=? order by e.row',(w,))
            checked = 0
            for i,((u,v,_,e),ref) in enumerate(zip(g.edges(keys=True,data=True),q)):
                row = (i,int(u),int(v),int(e['time']),e['label'],e['event_uuid'],int(e['y']))
                assert (row[1],row[2],row[3],row[6]) == (src[i],dst[i],t[i],y[i])
                if row != ref[:-1] or not isinstance(ref[-1],int):
                    errors[split]+=1
                    examples.setdefault(split, [w,row,ref])
                checked+=1
            assert checked==n and q.fetchone() is None
            for node,a in g.nodes(data=True):
                node=int(node)
                if nodes[node] != (a['node_type'],a['label']):
                    changed[node]=a['node_type']
            totals[split+'_events']+=n
            totals[split+'_windows']+=1
            if w%100==0:
                print(w,dict(totals),flush=True)
    summary['changed_native_labels_by_kind']=Counter(changed.values())
    summary['topology_reusable']=not errors
    out.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('cell','old','out'):
        p.add_argument('--'+field,required=True,type=Path)
    a=p.parse_args()
    audit(a.cell,a.old,a.out)
