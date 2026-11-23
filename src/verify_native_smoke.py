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
 p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);p.add_argument('--server-config',type=Path,required=True);p.add_argument('--full',action='store_true');p.add_argument('--repair-predictions',action='store_true');a=p.parse_args()
 out=Path(os.environ['RUN_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);cell=a.cell
 assert json.loads((cell/'status.json').read_text())['state']=='pipeline_finished_pending_acceptance'
 from pidsmaker.tasks.batching import get_preprocessed_graphs
 from pidsmaker.factory import build_model
 from pidsmaker.utils.utils import get_device,set_seed,ns_time_to_datetime_US
 from pidsmaker.detection.training_methods import inference_loop
 from pidsmaker.tasks import evaluation
 cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
 manifest=json.loads((cell/'manifest.json').read_text());model_name=manifest['model']
 from replay_profiles import validate_profile
 profile=validate_profile(model_name,cfg)
 from pidsmaker.utils.replay_state import restore_replay_state
 assert bool(cfg._test_mode)==(not a.full)
 assert cfg.evaluation.best_model_selection=='fixed_budget'
 if a.full:assert cfg.training.num_epochs==profile['epochs']
 cfg.database.password=database_options(json.loads(a.server_config.read_text()))['password']
 original_metrics=summarize(cell/'objects.jsonl')
 checkpoint=json.loads((cell/'checkpoint.json').read_text())['path']
 original=Path(cfg.training._edge_losses_dir)
 protected=[cell/f for f in ['resolved_config.yml','manifest.json','objects.jsonl','metrics.json','uuid_mapping.json','missing_gt.json','checkpoint.json','export_complete.json']]+[Path(checkpoint)]
 protected+=list(Path(checkpoint).parent.glob('replay_state.pt'))
 protected += [f for f in (cell/'artifacts/featurization').rglob('*') if f.is_file()]
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
  allowed={'pidsmaker/detection/training_methods/inference_loop.py','pidsmaker/detection/evaluation_methods/evaluation_utils.py','pidsmaker/utils/data_utils.py'} if a.repair_predictions else set()
  if a.repair_predictions:
   import ast
   old_code=cell.parent.parent/'code'
   old=ast.parse((old_code/'pidsmaker/utils/data_utils.py').read_text());new=ast.parse((root/'pidsmaker/utils/data_utils.py').read_text())
   for tree in (old,new):tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='save_model')]
   assert ast.dump(old)==ast.dump(new), 'Data loading/training semantics changed'
  for f,h in reuse['native_source_hashes'].items():
   if f not in allowed:assert digest(root/f)==h, f
  if a.repair_predictions:save(out/'repair_lineage.json',{'original_training_commit':manifest['commit'],'original_config_sha256':manifest['config_sha256'],'original_checkpoint_sha256':digest(checkpoint),'inference_commit':__import__('baseline_matrix').git('rev-parse','HEAD'),'changed_native_files':{f:digest(root/f) for f in allowed},'training_repeated':False,'reason':'Proven timestamp filename collision: test batches 1408/1409; no metric selection','original_resource_limitation':'Original full process-tree RSS and device VRAM were not sampled; cannot reconstruct retrospectively'})
 for split,dataset in [('train',train),('val',val),('test',test)]:
  batches=[g for group in dataset for g in group];edges=sum(g.original_edge_index.shape[1] for g in batches)
  if reuse:assert edges==reuse['splits'][split]['edges'], 'Preprocessed edge coverage mismatch: '+split
  coverage[split]={'batches':len(batches),'edges':edges,'independent_tensor_identity_checked':True,'prepared_windows':reuse['splits'][split]['windows'] if reuse else None,'unscored_preprocessed_edges':0}
 def native_ids(g):
  return getattr(g,'original_n_id_tgn',g.original_n_id) if profile['scoring']=='node' else g.original_edge_index.flatten()
 expected_nodes={int(n) for group in test for g in group for n in native_ids(g).cpu().tolist()}
 scored_nodes={json.loads(line)['node_id'] for line in (cell/'objects.jsonl').open()}
 if not a.repair_predictions:assert expected_nodes==scored_nodes, 'Scored node universe differs from independent test tensors'
 coverage['test']['scored_nodes']=len(expected_nodes)
 coverage['test']['unscored_native_nodes']=0
 save(out/'coverage.json',coverage)
 set_seed(cfg,seed=0)
 model=build_model(data_sample=train[0][0],device=get_device(cfg),cfg=cfg,max_node_num=max_nodes,all_data=(val,test))
 initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
 load_evaluation_checkpoint(model,checkpoint)
 restored=restore_replay_state(model,Path(checkpoint).parent/'replay_state.pt',required=model_name!='velox')
 changed=[k for k,v in model.named_parameters() if not torch.equal(initial[k],v.detach().cpu())]
 assert changed, 'No learned parameter differs from seeded initialization'
 original_magic=Path(cfg.training._magic_dir)/'train_distance.txt' if model_name=='magic' else None
 if model_name=='magic':cfg.training._magic_dir=str(out/'magic_replay')
 cfg.training._edge_losses_dir=str(out/'replay_losses')
 wandb.init(mode='disabled')
 stats=inference_loop.main(cfg,model,val,test,cfg.training.num_epochs-1,'all')
 from pidsmaker.utils.prediction_identity import prediction_name
 def compare_inputs(directory):
  evidence={}
  for split,dataset in [('val',val),('test',test)]:
   names=[];rows=0
   for gi,group in enumerate(dataset):
    for bi,g in enumerate(group):
     end=g.t[-1] if profile['scoring']=='node' else g.t.max()
     interval=ns_time_to_datetime_US(g.t[0],timezone=cfg.dataset.timezone)+'~'+ns_time_to_datetime_US(end,timezone=cfg.dataset.timezone)
     name=prediction_name(interval,gi,bi)+'.csv';names.append(name)
     path=directory/split/epoch/name
     if not path.exists():raise ValueError('Missing stable batch identity '+name+'; legacy timestamp-only outputs require collision audit and explicit repair')
     frame=pd.read_csv(path)
     if profile['scoring']=='node':expected=pd.DataFrame({'node':native_ids(g).cpu().numpy()})
     else:expected=pd.DataFrame({'srcnode':g.original_edge_index[0].cpu().numpy(),'dstnode':g.original_edge_index[1].cpu().numpy(),'time':g.t.cpu().numpy(),'edge_type':(g.edge_type.max(dim=1).indices+1).cpu().numpy()})
     keys=list(expected.columns)
     pd.testing.assert_frame_equal(frame[keys].sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False)
     rows+=len(frame)
   assert len(names)==len(set(names))
   assert set(names)=={f.name for f in (directory/split/epoch).glob('*.csv')}
   evidence[split]={'batches':len(names),'rows':rows,'missing_identities':0,'extra_identities':0}
  return evidence
 coverage['prediction_identity']=compare_inputs(out/'replay_losses')
 if not a.repair_predictions:compare_inputs(original)
 comparisons=[]
 if not a.repair_predictions:
  for split in ['val','test']:
   old=original/split/epoch;new=out/'replay_losses'/split/epoch
   assert sorted(x.name for x in old.glob('*.csv'))==sorted(x.name for x in new.glob('*.csv'))
   for f in old.glob('*.csv'):
    left=pd.read_csv(f);right=pd.read_csv(new/f.name)
    pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-6,atol=1e-7)
    comparisons.append({'split':split,'file':f.name,'rows':len(left)})
 def evaluate_at(losses,label):
  target=out/label;target.mkdir()
  local=cfg.clone();local.training._edge_losses_dir=str(losses)
  for k,v in list(local.evaluation.items()):
   if k.startswith('_') and isinstance(v,str) and v.startswith(str(cell)):
    setattr(local.evaluation,k,str(target/'evaluation'/k.lstrip('_')))
  local.construction._tw_labels=str(target/'evaluation/tw_labels')
  local.construction._node_id_to_path=str(target/'evaluation/node_id_to_path')
  evaluation.main(local);export(local,target)
  return target,summarize(target/'objects.jsonl')
 replay,replay_metrics=evaluate_at(out/'replay_losses','corrected_export')
 assert summarize(replay/'objects.jsonl')==replay_metrics
 right=pd.read_json(replay/'objects.jsonl',lines=True)
 assert set(right.node_id)==expected_nodes
 if not a.repair_predictions:assert replay_metrics==original_metrics
 save(out/'coverage.json',coverage)
 if model_name=='magic':
  generated=out/'magic_replay/train_distance.txt'
  reference=generated if a.repair_predictions else original_magic
  np.testing.assert_allclose(np.loadtxt(generated),np.loadtxt(reference),rtol=1e-6,atol=1e-7)
 after={str(f):digest(f) for f in protected};save(out/'source_hashes_after.json',after);assert before==after
 checks={k:True for k in ['independent_coverage','model_rebuilt','scores_replayed','decisions_recomputed','source_unchanged']};checks['full_budget']=a.full
 save(out/'summary.json',{'state':'reconstruction_replay_verified','source_cell':str(cell),'checks':checks,'coverage':coverage,'metrics_recomputed':replay_metrics,'comparisons':comparisons,'inference':stats,'changed_parameters':changed,'model_profile':profile,'restored_state':restored,'full_accepted':False if a.repair_predictions else a.full,'repaired_predictions_verified':a.repair_predictions,'original_attempt_accepted':False,'original_metrics_for_lineage_only':original_metrics,'corrected_export':str(replay)})
if __name__=='__main__':main()
