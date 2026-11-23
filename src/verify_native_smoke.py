"""Rebuild a real Velox model and replay bounded validation/test without training."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml,wandb
from yacs.config import CfgNode
from baseline_matrix import summarize,save,load_evaluation_checkpoint
from server_credentials import database_options
p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);p.add_argument('--server-config',type=Path,required=True);a=p.parse_args()
out=Path(os.environ['RUN_OUTPUT_DIR']);cell=a.cell
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.factory import build_model
from pidsmaker.utils.utils import get_device,set_seed
from pidsmaker.detection.training_methods import inference_loop
cfg=CfgNode(yaml.safe_load((cell/'resolved_config.yml').read_text()))
assert cfg._test_mode and cfg.training.encoder.used_methods == 'none'
cfg.database.password=database_options(json.loads(a.server_config.read_text()))['password']
metrics=summarize(cell/'objects.jsonl')
train,val,test,max_nodes=get_preprocessed_graphs(cfg)
set_seed(cfg,seed=0)
model=build_model(data_sample=train[0][0],device=get_device(cfg),cfg=cfg,max_node_num=max_nodes,all_data=(val,test))
checkpoint=json.loads((cell/'checkpoint.json').read_text())['path']
initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
load_evaluation_checkpoint(model,checkpoint)
changed=[k for k,v in model.named_parameters() if not torch.equal(initial[k],v.detach().cpu())]
assert changed, 'No learned parameter differs from seeded initialization'
original=Path(cfg.training._edge_losses_dir)
cfg.training._edge_losses_dir=str(out/'replay_losses')
wandb.init(mode='disabled')
stats=inference_loop.main(cfg,model,val,test,cfg.training.num_epochs-1,'all')
comparisons=[]
for split in ['val','test']:
 old=original/split/'model_epoch_0';new=out/'replay_losses'/split/'model_epoch_0'
 assert sorted(x.name for x in old.glob('*.csv')) == sorted(x.name for x in new.glob('*.csv'))
 for f in old.glob('*.csv'):
  left=pd.read_csv(f);right=pd.read_csv(new/f.name)
  pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-6,atol=1e-7)
  comparisons.append({'split':split,'file':f.name,'rows':len(left),'max_loss_error':float(np.max(np.abs(left.loss-right.loss)))})
assert comparisons
save(out/'summary.json',{'state':'reconstruction_replay_verified','source_cell':str(cell),'metrics_recomputed':metrics,'comparisons':comparisons,'inference':stats,'changed_parameters':changed,'full_accepted':False})
