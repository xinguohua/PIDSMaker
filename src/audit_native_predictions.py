"""Read-only input-batch/CSV collision audit. No model inference or training."""
import argparse,json,os,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml,pandas as pd
from yacs.config import CfgNode
from baseline_matrix import save,digest
from server_credentials import database_options
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import ns_time_to_datetime_US,set_seed
p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);p.add_argument('--server-config',type=Path,required=True);a=p.parse_args()
cfg=CfgNode(yaml.safe_load((a.cell/'resolved_config.yml').read_text()));cfg.database.password=database_options(json.loads(a.server_config.read_text()))['password']
set_seed(cfg,seed=0);train,val,test,_=get_preprocessed_graphs(cfg)
out=Path(os.environ['RUN_OUTPUT_DIR']);report={'source':str(a.cell),'splits':{}}
for split,data in [('val',val),('test',test)]:
 groups=collections.defaultdict(list)
 for gi,gs in enumerate(data):
  for bi,g in enumerate(gs):
   name=ns_time_to_datetime_US(g.t[0],timezone=cfg.dataset.timezone)+'~'+ns_time_to_datetime_US(g.t.max(),timezone=cfg.dataset.timezone)+'.csv'
   keys=list(zip(g.original_edge_index[0].tolist(),g.original_edge_index[1].tolist(),g.t.tolist(),(g.edge_type.max(dim=1).indices+1).tolist()))
   groups[name].append((gi,bi,keys))
 directory=Path(cfg.training._edge_losses_dir)/split/f'model_epoch_{cfg.training.num_epochs-1}'
 anomalies=[];expected_count=actual_count=missing=extra=0
 for name,batches in groups.items():
  f=directory/name;expected=collections.Counter(k for _,_,keys in batches for k in keys)
  actual=collections.Counter(map(tuple,pd.read_csv(f)[['srcnode','dstnode','time','edge_type']].itertuples(index=False,name=None))) if f.exists() else collections.Counter()
  nmiss=sum((expected-actual).values());nextra=sum((actual-expected).values());expected_count+=sum(expected.values());actual_count+=sum(actual.values());missing+=nmiss;extra+=nextra
  if nmiss or nextra or len(batches)>1:
   anomalies.append({'file':name,'batches':[{'group':gi,'batch':bi,'rows':len(keys)} for gi,bi,keys in batches],'missing_identities':nmiss,'extra_identities':nextra,'csv_sha256':digest(f) if f.exists() else None,'equals_last_batch':actual==collections.Counter(batches[-1][2]),'missing_examples':list((expected-actual).items())[:3]})
 report['splits'][split]={'native_batches':sum(len(bs) for bs in groups.values()),'distinct_names':len(groups),'csv_files':len(list(directory.glob('*.csv'))),'expected_rows':expected_count,'actual_rows':actual_count,'missing_identities':missing,'extra_identities':extra,'anomalies':anomalies}
 save(out/'summary.json',report)
