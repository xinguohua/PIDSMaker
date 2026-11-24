"""Reopen the frozen Velox source graph and training tensor for each published example."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audit(cell, root):
    import torch
    import yaml
    from yacs.config import CfgNode

    sys.path.insert(0,str(cell.parents[1]/'code'))
    from pidsmaker.utils.data_utils import custom_temporal_data_loader
    from pidsmaker.utils.dataset_utils import edge_reversed, get_rel2id
    from pidsmaker.utils.feature_cache import load_feature_data

    cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    rel=get_rel2id(cfg)
    dates=set(json.loads((cell/'manifest.json').read_text())['protocol']['train_dates'])
    train=Path(cfg.feat_inference._edge_embeds_dir)/'train'
    inputs=sorted(train.iterdir())
    bs=cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
    width=cfg.dataset.num_node_types+cfg.featurization.emb_dim
    windows=records(root/'NATIVE_WINDOWS.csv')
    if not inputs or len(windows)!=len(inputs):
        raise ValueError('Window count differs from native training inputs')
    examples=[json.loads(line) for line in (root/'COMPOSITION_EXAMPLES.jsonl').read_text().splitlines()]
    if not examples:
        raise ValueError('No examples to validate')
    results=[]
    for case in examples:
        cid=case['composition_id']
        part=root/'COMPOSITION_ASSIGNMENTS'/f"{case['order']}_{case['hop']}.csv"
        with part.open() as stream:
            members=Counter(int(r['unit_id']) for r in csv.DictReader(stream) if int(r['composition_id'])==cid)
        if members!=Counter(u['uid'] for u in case['units']):
            raise ValueError((cid,'member set'))
        wid=int(case['native_window']['window_id'])
        if any(u['segment']!=wid for u in case['units']):
            raise ValueError((cid,'window membership'))
        tensor_path=inputs[wid]
        name=tensor_path.name.removesuffix('.TemporalData.simple')
        date=name[:10]
        assert date in dates
        window=windows[wid]
        relative=Path('graph_'+date)/name
        graph_path=Path(cfg.transformation._graphs_dir)/relative
        construction_path=Path(cfg.construction._graphs_dir)/relative
        assert window==case['native_window']
        assert str(tensor_path)==window['tensor_file'] and str(graph_path)==window['transformed_file']
        assert construction_path.samefile(graph_path)
        g=torch.load(graph_path,map_location='cpu')
        data=load_feature_data(tensor_path)
        batches=list(custom_temporal_data_loader(data,bs))
        # Sum the native slice counts for earlier source windows (each independently bounded).
        first_batch=sum((int(w['edges'])+bs-1)//bs for w in windows[:wid])
        assert first_batch==int(window['first_batch_id'])
        e_by_id={e['eid']:e for e in case['events']}
        assert len(e_by_id)==len(case['events'])
        subjects={str(unit['native_nid']):unit['uid'] for unit in case['units'] if unit['subject'] is not None}
        uncertain={unit['fallback_eid']:unit['uid'] for unit in case['units'] if unit['subject'] is None}
        for u in case['units']:
            if u['subject'] is not None:
                info=g.nodes[str(u['native_nid'])]
                assert info['node_type']=='subject' and info['label']==u['native_label']
        owners={}
        for i,(src,dst,key,edge) in enumerate(g.edges(keys=True,data=True)):
            eid=(wid<<32)|i
            subjects_on_edge=[s for s in (src,dst) if g.nodes[s]['node_type']=='subject']
            if len(subjects_on_edge)==1:
                owner=subjects.get(subjects_on_edge[0])
            elif len(subjects_on_edge)==2:
                owner=subjects.get(dst if edge['label'] in edge_reversed else src)
            else:
                owner=uncertain.get(eid)
            if owner is None:
                continue
            owners[eid]=owner
            event=e_by_id.get(eid)
            if event is None:
                raise ValueError((cid,eid,'owned native edge missing'))
            expected=(wid,i,int(src),int(dst),edge['label'],edge['time'],edge['event_uuid'],key,edge['y'],owner)
            actual=tuple(event[k] for k in ('window_id','tensor_row','src','dst','op','ts','edge_uuid','nx_key','y','owner'))
            if expected!=actual:
                raise ValueError((cid,eid,'source graph and case differ'))
            batch_idx,batch_row=divmod(i,bs)
            assert (first_batch+batch_idx,batch_row)==(event['batch_id'],event['batch_row'])
            batch=batches[batch_idx]
            for frame,row in ((data,i),(batch,batch_row)):
                assert (int(frame.src[row]),int(frame.dst[row]),int(frame.t[row]),int(frame.y[row]))==(
                    int(src),int(dst),edge['time'],edge['y'])
                onehot=frame.msg[row,width:width+cfg.dataset.num_edge_types]
                assert int(onehot.sum())==1 and float(onehot[rel[edge['label']]-1])==1
            assert torch.equal(data.msg[i],batch.msg[batch_row])
        if owners!={eid:e['owner'] for eid,e in e_by_id.items()}:
            raise ValueError((cid,'extraneous or missing edge in complete native ownership set'))
        results.append(dict(order=case['order'],hop=case['hop'],composition_id=cid,window_id=wid,
            members_verified=len(members),all_owned_source_edges=len(owners),example_rows=len(e_by_id),
            distinct_batches=len({e['batch_id'] for e in case['events']}),graph_tensor_batch_verified=len(owners),
            missing_members=0,missing_edges=0,extra_edges=0,field_mismatches=0))
    with (root/'EXAMPLE_SOURCE_AUDIT.partial').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    (root/'EXAMPLE_SOURCE_AUDIT.partial').replace(root/'EXAMPLE_SOURCE_AUDIT.csv')
    print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell',type=Path)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    audit(args.cell.resolve(),args.root.resolve())
