"""Native construction/transformation boundary with disk monitoring and measurement."""
import json,os,shutil,threading,time
from pathlib import Path
from baseline_matrix import save,now

def prepare_native(cfg,cell,out,main):
    reserve=4*1024**3
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
        from pidsmaker.utils.dataset_utils import get_rel2id,get_node_map
        edge_types=max(v for v in get_rel2id(cfg).values() if type(v) is int);node_types=max(v for v in get_node_map().values() if type(v) is int)
        tensor_bytes=(total_edges*((2*cfg.featurization.emb_dim+2*node_types+edge_types)*4+4*8)
                      if cfg.featurization.emb_dim is not None and cfg.featurization.used_method in ['word2vec','hierarchical_hashing'] else None)
        report={'state':'preparation_complete_pending_capacity','time':now(),'cell':str(cell),'splits':totals,'windows_ledger':str(ledger),'feature_dimension':cfg.featurization.emb_dim,'edge_types':edge_types,'node_types':node_types,'native_dense_feature_tensor_bytes':tensor_bytes,'capacity_note':'null tensor bytes means model-specific materialized shape is not yet measured; do not apply Velox formula. Non-null tensor bytes excludes serialization overhead, fitted feature models, full predictions and checkpoints; no smoke extrapolation','free_bytes':shutil.disk_usage(cell).free,'full_training_started':False,'raw_complete':False}
        save(cell/'status.json',report);save(out/'summary.json',report)
    finally:
        stop.set();os.environ.pop('BASELINE_PREPARATION_LEDGER',None)
