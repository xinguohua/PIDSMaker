"""Recover the completed graph stage after capacity reporting failed; no graph rebuild."""
import argparse,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from yacs.config import CfgNode
from baseline_matrix import save,now,git
from native_preparation import dense_tensor_capacity

def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);a=p.parse_args()
    cell=a.cell;cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
    from pidsmaker.utils.dataset_utils import get_rel2id,get_node_map,get_num_edge_type
    for task in ('construction','transformation'):
        assert (Path(getattr(cfg,task)._task_path)/'done.txt').is_file(), task
    rows=[json.loads(line) for line in (cell/'windows.jsonl').read_text().splitlines()]
    assert {r['date'] for r in rows}==set(cfg.dataset.train_dates+cfg.dataset.val_dates+cfg.dataset.test_dates)
    for r in rows:assert Path(r['path']).stat().st_size==r['bytes']
    totals={}
    for split in ('train','val','test'):
        rr=[r for r in rows if r['split']==split];assert rr
        totals[split]={'windows':len(rr),**{k:sum(r[k] for r in rr) for k in ('input_events_after_type_filter','events_after_dedup','nodes','edges','bytes')}}
    edges=max(v for v in get_rel2id(cfg).values() if type(v) is int)
    nodes=max(v for v in get_node_map(cfg).values() if type(v) is int)
    capacity=dense_tensor_capacity(cfg.featurization,sum(r['edges'] for r in rows),nodes,edges,get_num_edge_type(cfg))
    report={'state':'preparation_complete_pending_capacity','time':now(),'cell':str(cell),'splits':totals,**capacity,'edge_types':edges,'node_types':nodes,'free_bytes':shutil.disk_usage(cell).free,'recovery_commit':git('rev-parse','HEAD'),'original_status':json.loads((cell/'status.json').read_text()),'graphs_rebuilt':False}
    save(cell/'preparation_recovery.json',report)
    save(Path(os.environ['RUN_OUTPUT_DIR'])/'summary.json',report)
if __name__=='__main__':main()
