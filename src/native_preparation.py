"""Native construction/transformation boundary with disk monitoring and measurement."""
import json,os,shutil,threading,time
from pathlib import Path
from baseline_matrix import save,now

def feature_storage_capacity(report, cache_format='dense'):
    if cache_format == 'dense':return report['native_dense_feature_tensor_bytes']
    if cache_format != 'node_table_v1':raise ValueError('Unknown cache format')
    width=report['node_types']+report['feature_dimension'];edge_width=report['edge_types']
    # Graph node count bounds the window-local table. Original int64 src/dst/t/y
    # plus three int64 lookup indices per edge; native one-hot edge type table.
    return sum(s['nodes']*width*4+s['edges']*56+s['windows']*(edge_width+1)*edge_width*4 for s in report['splits'].values())

def dense_tensor_capacity(features, total_edges, node_types, edge_types, ocr_histogram_bins=None):
    """Size the tensors saved by feat_inference, excluding file and object overhead."""
    method = features.used_method.strip()
    if method == 'ocrapt_features':
        options = features.ocrapt_features
        # Incoming/outgoing histograms, three idle statistics, and optional time columns.
        width = (2 * ocr_histogram_bins + 3
                 + int(options.use_lifespan) + int(options.use_cumulative_active_time))
    elif method in ('only_type', 'only_ones'):
        width = 0
    elif method in ('word2vec', 'hierarchical_hashing'):
        width = features.emb_dim
    else:
        width = None
    if width is None:
        return {'feature_dimension': None, 'native_dense_feature_tensor_bytes': None}
    message_width = 2 * (node_types + width) + edge_types
    # One float32 message and four int64 fields (src, dst, t, y) per prepared edge.
    return {'feature_dimension': width, 'native_message_width': message_width,
            'native_dense_feature_tensor_bytes': total_edges * (4 * message_width + 4 * 8)}


def prepare_native(cfg,cell,out,main):
    reserve=512*1024**2
    initial=shutil.disk_usage(cell).free
    if initial < reserve+1024**3: raise RuntimeError('Insufficient preparation reserve')
    ledger=cell/'windows.jsonl'
    os.environ['BASELINE_PREPARATION_LEDGER']=str(ledger)
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            free=shutil.disk_usage(cell).free
            save(cell/'disk_progress.json',{'time':now(),'free_bytes':free,'initial_free_bytes':initial,'reserve_bytes':reserve,'net_disk_consumed_bytes':initial-free})
            if free < reserve:
                report={'state':'stopped_disk_reserve','time':now(),'free_bytes':free,'reserve_bytes':reserve,'partial_artifacts_retained':True}
                save(cell/'status.json',report);save(out/'summary.json',report)
                os._exit(74)
            stop.wait(1)
    threading.Thread(target=monitor,daemon=True).start()
    save(cell/'status.json',{'state':'full_native_preparation','time':now(),'full_training_started':False,'write_budget_bytes':initial-reserve})
    try:
        main(cfg,exp=os.environ['RUN_ID'],stop_after='transformation')
        rows=[json.loads(x) for x in ledger.read_text().splitlines()]
        totals={}
        for split in ('train','val','test'):
            rr=[r for r in rows if r['split']==split]
            if not rr:raise RuntimeError('Empty split '+split)
            totals[split]={'windows':len(rr),**{k:sum(r[k] for r in rr) for k in ('input_events_after_type_filter','events_after_dedup','nodes','edges','bytes')}}
        total_edges=sum(r['edges'] for r in rows)
        # Exact native dense tensor dimensions; serialization/CSV/model overhead separate.
        from pidsmaker.utils.dataset_utils import get_rel2id,get_node_map,get_num_edge_type
        edge_types=max(v for v in get_rel2id(cfg).values() if type(v) is int);node_types=max(v for v in get_node_map().values() if type(v) is int)
        bins = get_num_edge_type(cfg) if cfg.featurization.used_method.strip() == 'ocrapt_features' else None
        capacity = dense_tensor_capacity(cfg.featurization, total_edges, node_types, edge_types, bins)
        report={'state':'preparation_complete_pending_capacity','time':now(),'cell':str(cell),'splits':totals,'windows_ledger':str(ledger),**capacity,'edge_types':edge_types,'node_types':node_types,'capacity_note':'Tensor payload uses prepared edge counts and the native feature layout. It excludes serialization overhead, fitted feature models, predictions and checkpoints. A null value denotes an unknown layout.','free_bytes':shutil.disk_usage(cell).free,'full_training_started':False,'raw_complete':False}
        save(cell/'status.json',report);save(out/'summary.json',report)
    finally:
        stop.set();os.environ.pop('BASELINE_PREPARATION_LEDGER',None)
