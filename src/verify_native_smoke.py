"""Independent native replay, extending the original real Velox smoke verifier.
Unsupported model/state/coverage protocols fail closed; no model mocks.
"""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch,yaml,wandb
from yacs.config import CfgNode
from baseline_matrix import summarize,save,digest,load_evaluation_checkpoint,export
from server_credentials import database_options

def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);p.add_argument('--server-config',type=Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
 out=Path(os.environ['RUN_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);cell=a.cell
 assert json.loads((cell/'status.json').read_text())['state']=='pipeline_finished_pending_acceptance'
 from pidsmaker.tasks.batching import get_preprocessed_graphs
 from pidsmaker.factory import build_model
 from pidsmaker.utils.utils import get_device,set_seed,ns_time_to_datetime_US
 from pidsmaker.detection.training_methods import inference_loop
 from pidsmaker.tasks import evaluation
 cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
 assert cfg.training.encoder.used_methods=='none', 'Non-Velox fitted/online-state replay not yet qualified'
 assert bool(cfg._test_mode)==(not a.full)
 assert cfg.evaluation.best_model_selection=='fixed_budget'
 if a.full:assert cfg.training.num_epochs==12
 cfg.database.password=database_options(json.loads(a.server_config.read_text()))['password']
 original_metrics=summarize(cell/'objects.jsonl')
 checkpoint=json.loads((cell/'checkpoint.json').read_text())['path']
 original=Path(cfg.training._edge_losses_dir)
 protected=[cell/f for f in ['resolved_config.yml','manifest.json','objects.jsonl','metrics.json','uuid_mapping.json','missing_gt.json','checkpoint.json','export_complete.json']]+[Path(checkpoint)]
 protected+=list(original.rglob('*.csv'))+list(Path(cfg.evaluation._precision_recall_dir).rglob('*.pth'))
 before={str(f):digest(f) for f in protected};save(out/'source_hashes_before.json',before)
 # Read frozen feature tensors, not predictions, to construct the scoring universe.
 set_seed(cfg,seed=0)
 train,val,test,max_nodes=get_preprocessed_graphs(cfg)
 coverage={};epoch=f'model_epoch_{cfg.training.num_epochs-1}'
 reuse=json.loads((cell/'prepared_reuse.json').read_text()) if a.full else None
 if reuse:
  from baseline_matrix import protocol
  expected=protocol()[json.loads((cell/'manifest.json').read_text())['dataset']]
  for key in ['train_dates','val_dates','test_dates']:
   assert list(getattr(cfg.dataset,key))==expected[key]
  assert reuse['verified']
  for f,h in reuse['shared_files_hashes'].items():assert digest(f)==h, 'Prepared input changed: '+f
  manifest=json.loads((cell/'manifest.json').read_text());assert manifest['input_sha256']==reuse['input_sha256']
  assert digest(cell/'input_manifest.json')==manifest['input_sha256'] or json.loads((cell/'input_manifest.json').read_text())==json.loads(Path(json.loads(a.server_config.read_text())['input_manifests'][manifest['dataset']]).read_text())
  assert manifest['mode']=='full' and manifest['seed']==0
  assert digest(cell/'resolved_config.yml')==manifest['config_sha256']
  root=Path(__file__).resolve().parents[1]
  for f,h in manifest['gt_sha256'].items():assert digest(root/'Ground_Truth/orthrus'/f)==h
  for f,h in reuse['native_source_hashes'].items():assert digest(root/f)==h
 for split,dataset in [('train',train),('val',val),('test',test)]:
  batches=[g for group in dataset for g in group];edges=sum(g.original_edge_index.shape[1] for g in batches)
  if reuse:assert edges==reuse['splits'][split]['edges'], 'Preprocessed edge coverage mismatch: '+split
  expected_names=[]
  for g in batches:
   if split=='train':continue
   name=ns_time_to_datetime_US(g.t[0],timezone=cfg.dataset.timezone)+'~'+ns_time_to_datetime_US(g.t.max(),timezone=cfg.dataset.timezone)+'.csv'
   expected_names.append(name);frame=pd.read_csv(original/split/epoch/name)
   assert len(frame)==g.original_edge_index.shape[1]
   for col,arr in [('srcnode',g.original_edge_index[0]),('dstnode',g.original_edge_index[1]),('time',g.t),('edge_type',g.edge_type.max(dim=1).indices+1)]:
    np.testing.assert_array_equal(frame[col].to_numpy(),arr.cpu().numpy())
  if split!='train':
   assert len(expected_names)==len(set(expected_names)), 'Native CSV name collision'
   assert set(expected_names)=={f.name for f in (original/split/epoch).glob('*.csv')}
  coverage[split]={'batches':len(batches),'edges':edges,'independent_tensor_identity_checked':True,'prepared_windows':reuse['splits'][split]['windows'] if reuse else None,'unscored_preprocessed_edges':0}
 expected_nodes={int(n) for group in test for g in group for n in g.original_edge_index.flatten().cpu().tolist()}
 scored_nodes={json.loads(line)['node_id'] for line in (cell/'objects.jsonl').open()}
 assert expected_nodes==scored_nodes, 'Scored node universe differs from independent test tensors'
 coverage['test']['scored_nodes']=len(expected_nodes)
 coverage['test']['unscored_native_nodes']=0
 save(out/'coverage.json',coverage)
 set_seed(cfg,seed=0)
 model=build_model(data_sample=train[0][0],device=get_device(cfg),cfg=cfg,max_node_num=max_nodes,all_data=(val,test))
 initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
 load_evaluation_checkpoint(model,checkpoint)
 changed=[k for k,v in model.named_parameters() if not torch.equal(initial[k],v.detach().cpu())]
 assert changed, 'No learned parameter differs from seeded initialization'
 cfg.training._edge_losses_dir=str(out/'replay_losses')
 wandb.init(mode='disabled')
 stats=inference_loop.main(cfg,model,val,test,cfg.training.num_epochs-1,'all')
 comparisons=[]
 for split in ['val','test']:
  old=original/split/epoch;new=out/'replay_losses'/split/epoch
  assert sorted(x.name for x in old.glob('*.csv'))==sorted(x.name for x in new.glob('*.csv'))
  for f in old.glob('*.csv'):
   left=pd.read_csv(f);right=pd.read_csv(new/f.name)
   pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-6,atol=1e-7)
   comparisons.append({'split':split,'file':f.name,'rows':len(left),'max_loss_error':float(np.max(np.abs(left.loss-right.loss)))})
 assert comparisons
 # Redirect every evaluation/cache write; original artifacts remain read-only inputs.
 for k,v in list(cfg.evaluation.items()):
  if k.startswith('_') and isinstance(v,str) and v.startswith(str(cell)):
   setattr(cfg.evaluation,k,str(out/'evaluation'/k.lstrip('_')))
 cfg.construction._tw_labels=str(out/'evaluation/tw_labels')
 cfg.construction._node_id_to_path=str(out/'evaluation/node_id_to_path')
 evaluation.main(cfg)
 replay=out/'export';replay.mkdir();export(cfg,replay)
 replay_metrics=summarize(replay/'objects.jsonl');assert replay_metrics==original_metrics
 left=pd.read_json(cell/'objects.jsonl',lines=True).sort_values('node_id').reset_index(drop=True)
 right=pd.read_json(replay/'objects.jsonl',lines=True).sort_values('node_id').reset_index(drop=True)
 pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-6,atol=1e-7)
 assert digest(cell/'uuid_mapping.json')==digest(replay/'uuid_mapping.json')
 assert digest(cell/'missing_gt.json')==digest(replay/'missing_gt.json')
 after={str(f):digest(f) for f in protected};save(out/'source_hashes_after.json',after);assert before==after
 checks={k:True for k in ['independent_coverage','model_rebuilt','scores_replayed','decisions_recomputed','source_unchanged']};checks['full_budget']=a.full
 save(out/'summary.json',{'state':'reconstruction_replay_verified','source_cell':str(cell),'checks':checks,'coverage':coverage,'metrics_recomputed':replay_metrics,'comparisons':comparisons,'inference':stats,'changed_parameters':changed,'full_accepted':a.full})
if __name__=='__main__':main()
