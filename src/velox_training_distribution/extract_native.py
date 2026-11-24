"""Index the frozen Velox TRAIN graphs and tensors, checking every edge and loader batch."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile


EDGE_COLUMNS = ('eid','window_id','tensor_row','src','dst','ts','op','edge_uuid','nx_key',
                'y','src_type_id','op_id','dst_type_id','batch_id','batch_row')
NODE_COLUMNS = ('window_id','nid','kind','label','uuid')


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(staging, out):
    # Use the separate Arrow environment so the original PIDSMaker/Torch environment stays fixed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    for name, cols, strings in (('NATIVE_EDGES',EDGE_COLUMNS,{'op','edge_uuid'}),
                                ('NATIVE_NODES',NODE_COLUMNS,{'kind','label','uuid'})):
        destination = out/name
        destination.mkdir(exist_ok=True)
        schema = pa.schema([(c, pa.string() if c in strings else pa.int64()) for c in cols])
        for file in sorted(staging.glob(name+'*.pkl')):
            with file.open('rb') as stream:
                rows=pickle.load(stream)
            table=pa.table({c:[r[i] for r in rows] for i,c in enumerate(cols)}, schema=schema)
            target=destination/(file.stem[len(name):]+'.parquet')
            pq.write_table(table,target.with_suffix('.partial'),compression='zstd')
            target.with_suffix('.partial').replace(target)


def run(cell, out, arrow_python):
    import torch
    import yaml
    from yacs.config import CfgNode

    train_run = cell.parents[1]
    sys.path.insert(0,str(train_run/'code'))
    from pidsmaker.utils.data_utils import custom_temporal_data_loader
    from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
    from pidsmaker.utils.feature_cache import load_feature_data

    manifest=json.loads((cell/'manifest.json').read_text())
    metadata=json.loads((train_run/'meta.json').read_text())
    train_commit=subprocess.check_output(['git','-C',str(train_run/'code'),'rev-parse','HEAD'],text=True).strip()
    assert train_commit==manifest['commit']==metadata['commit']
    assert manifest['model']=='velox' and manifest['dataset']=='CADETS_E3'
    cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    assert cfg.transformation.used_methods=='none'
    assert cfg.batching.global_batching.used_method=='none' and cfg.batching.inter_graph_batching.used_method=='none'
    assert cfg.batching.intra_graph_batching.used_methods=='edges'
    bs=cfg.batching.intra_graph_batching.edges.intra_graph_batch_size
    dates=manifest['protocol']['train_dates']
    graphs=Path(cfg.transformation._graphs_dir)
    constructed=Path(cfg.construction._graphs_dir)
    train=Path(cfg.feat_inference._edge_embeds_dir)/'train'
    tensors=sorted(train.iterdir())
    names={p.name for date in dates for p in (graphs/('graph_'+date)).iterdir()}
    assert names=={p.name.removesuffix('.TemporalData.simple') for p in tensors}
    node_types=get_node_map()
    relation_ids=get_rel2id(cfg)
    width=cfg.dataset.num_node_types+cfg.featurization.emb_dim
    classes=cfg.dataset.num_edge_types
    native_uuid=json.loads((cell/'uuid_mapping.json').read_text())
    log=(train_run/'run.log').read_text()
    n_batches=int(re.search(r'Train num graphs:\s*(\d+)',log)[1])
    out.mkdir(parents=True, exist_ok=True)
    windows=[]; batches=[]; triples=Counter()
    observed_nodes=set()
    with tempfile.TemporaryDirectory(prefix='native-export-',dir=out) as scratch:
        staging=Path(scratch)
        for wid,tensor_file in enumerate(tensors):
            name=tensor_file.name.removesuffix('.TemporalData.simple')
            date=name[:10]
            assert date in dates
            rel=Path('graph_'+date)/name
            graph_file=graphs/rel
            original=constructed/rel
            graph_hash=digest(graph_file)
            assert original.samefile(graph_file) or digest(original)==graph_hash
            graph=torch.load(graph_file,map_location='cpu')
            assert graph.is_directed() and graph.is_multigraph()
            data=load_feature_data(tensor_file).to('cpu')
            assert set(data.keys())=={'src','dst','t','msg','y','tgn_mode'}
            assert not data.tgn_mode
            n=data.src.numel()
            assert n==graph.number_of_edges() and tuple(data.msg.shape)==(n,2*width+classes)
            node_rows=[]; kinds={}
            for node,attrs in graph.nodes(data=True):
                assert set(attrs)=={'node_type','label'}
                node_rows.append((wid,int(node),attrs['node_type'],attrs['label'],native_uuid.get(str(node))))
                kinds[node]=attrs['node_type']
                observed_nodes.add(int(node))
            source_type=data.msg[:,:cfg.dataset.num_node_types]
            op_type=data.msg[:,width:width+classes]
            dest_type=data.msg[:,width+classes:width+classes+cfg.dataset.num_node_types]
            type_ids=[]
            for block in (source_type,op_type,dest_type):
                indices=block.argmax(dim=1)
                assert torch.equal(torch.nn.functional.one_hot(indices,block.shape[1]).to(block.dtype),block)
                type_ids.append(indices.tolist())
            source_ids,edge_ids,dest_ids=type_ids
            sources,dests,times,ys=[getattr(data,k).tolist() for k in ('src','dst','t','y')]
            first=len(batches)
            start=0
            for batch in custom_temporal_data_loader(data,bs):
                stop=start+batch.src.numel()
                for key in ('src','dst','t','y','msg'):
                    assert torch.equal(getattr(data,key)[start:stop],getattr(batch,key))
                assert start%bs==0 and 0<stop-start<=bs
                batches.append(dict(batch_id=len(batches),window_id=wid,first_tensor_row=start,
                                    last_tensor_row=stop-1,edges=stop-start,
                                    nodes=len(set(sources[start:stop]).union(dests[start:stop]))))
                start=stop
            assert start==n and n < 2**32
            edge_rows=[]
            for row,(u,v,key,attrs) in enumerate(graph.edges(data=True,keys=True)):
                assert set(attrs)=={'event_uuid','label','time','y'}
                op=attrs['label']
                expected=(int(u),int(v),int(attrs['time']),attrs['y'],node_types[kinds[u]]-1,
                          relation_ids[op]-1,node_types[kinds[v]]-1)
                actual=(sources[row],dests[row],times[row],ys[row],source_ids[row],edge_ids[row],dest_ids[row])
                if actual!=expected:
                    raise ValueError((wid,row,'graph/tensor mismatch',expected,actual))
                triples[op,kinds[u],kinds[v]]+=1
                edge_rows.append(((wid<<32)|row,wid,row,int(u),int(v),times[row],op,attrs['event_uuid'],
                                  int(key),ys[row],source_ids[row],edge_ids[row],dest_ids[row],first+row//bs,row%bs))
            for prefix,rows in [('NATIVE_EDGES',edge_rows),('NATIVE_NODES',node_rows)]:
                with (staging/f'{prefix}{wid:04d}.pkl').open('wb') as f:
                    pickle.dump(rows,f,protocol=4)
            windows.append(dict(window_id=wid,date=date,window_name=name,construction_file=str(original),
                transformed_file=str(graph_file),tensor_file=str(tensor_file),first_batch_id=first,
                last_batch_id=len(batches)-1,edges=n,nodes=len(node_rows),batches=len(batches)-first,
                min_ts=min(times),max_ts=max(times),graph_sha256=graph_hash,tensor_sha256=digest(tensor_file)))
            if wid%100==0:
                print(f'{wid+1}/{len(tensors)} windows verified',flush=True)
        assert len(batches)==n_batches
        for attr in ('nodes','edges'):
            limits=re.search(r'Train '+attr+r' \| mean: (\d+) \| min: (\d+) \| max: (\d+)',log)
            values=[b[attr] for b in batches]
            assert limits and (int(sum(values)/len(values)),min(values),max(values))==tuple(map(int,limits.groups()))
        subprocess.run([arrow_python,__file__,'--convert',str(staging),str(out)],check=True)
    write_csv(out/'NATIVE_WINDOWS.csv',windows)
    write_csv(out/'NATIVE_BATCHES.csv',batches)
    write_csv(out/'EDGE_ENDPOINT_TYPES.csv',[dict(op=op,src_kind=u,dst_kind=v,edges=c) for (op,u,v),c in sorted(triples.items())])
    audit=dict(train_run=train_run.name,train_commit=train_commit,source_cell=str(cell),dates=dates,
               graph_class='MultiDiGraph',transformation=cfg.transformation.used_methods,windows=len(windows),
               edge_occurrences=sum(r['edges'] for r in windows),within_window_node_occurrences=sum(r['nodes'] for r in windows),
               distinct_node_ids=len(observed_nodes),unmapped_node_ids=sum(str(i) not in native_uuid for i in observed_nodes),
               node_types=cfg.dataset.num_node_types,op_types=classes,word2vec_width=cfg.featurization.emb_dim,
               msg_width=2*width+classes,feature_fields=sorted(data.keys()),tensor_row_order='original feature tensor order',
               batch_method='native custom_temporal_data_loader per window',batch_size=bs,
               native_loader_train_batches=len(batches),logged_train_batches=n_batches,
               sources=dict(graph_labels_and_event_uuid='construction/transformation graph',
                            feature_input='original training TemporalData msg,src,dst,t,y',
                            node_uuid='training run uuid_mapping.json'),
               checks=dict(construction_equals_transformation_windows=len(windows),graph_tensor_edge_rows=sum(triples.values()),
                           native_loader_edges=sum(r['edges'] for r in batches),train_log_batch_statistics_matched=True))
    report=out/'NATIVE_INPUT_AUDIT.json'
    report.with_suffix('.tmp').write_text(json.dumps(audit,indent=2))
    report.with_suffix('.tmp').replace(report)
    print(audit,flush=True)


if __name__=='__main__':
    if sys.argv[1:2]==['--convert']:
        write_parquet(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('cell',type=Path)
        p.add_argument('out',type=Path)
        p.add_argument('--arrow-python',required=True)
        a=p.parse_args()
        run(a.cell.resolve(),a.out.resolve(),a.arrow_python)
