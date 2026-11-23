"""Server-only native imports/configuration and real TGN regression; no training."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
out=Path(os.environ['RUN_OUTPUT_DIR'])
os.environ.update(TMPDIR=str(out),XDG_CACHE_HOME=str(out/'cache'),WANDB_MODE='disabled')
r={'python':sys.executable,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'imports':{},'configs':[]}
for name in ['torch','torch_geometric','sklearn','yacs','psycopg2','pidsmaker.main']:
 try:
  m=importlib.import_module(name);r['imports'][name]={'ok':True,'version':getattr(m,'__version__',None)}
  if name=='torch':r['cuda']={'available':m.cuda.is_available(),'version':m.version.cuda,'devices':[m.cuda.get_device_name(i) for i in range(m.cuda.device_count())]}
 except Exception as e:r['imports'][name]={'ok':False,'error':str(e),'type':type(e).__name__}
try:
 from pidsmaker.config import get_runtime_required_args,get_yml_cfg
 for model in ['ocrapt','velox','orthrus','magic','kairos']:
  try:
   args,unknown=get_runtime_required_args(True,[model,'CADETS_E3','--artifact_dir',str(out/'config_check'), '--evaluation.best_model_selection','fixed_budget','--featurization.training_split','train'])
   cfg=get_yml_cfg(args)
   r['configs'].append({'model':model,'ok':not bool(unknown),'evaluation':cfg.evaluation.used_method,'epochs':cfg.training.num_epochs})
  except Exception as e:r['configs'].append({'model':model,'ok':False,'error':str(e)})
except Exception as e:r['config_import_error']=str(e)
t=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_baseline_tgn_checkpoint.py','-v'],capture_output=True,text=True,timeout=90)
(out/'tgn_test.txt').write_text(t.stdout+t.stderr)
r['tgn_test']={'exit':t.returncode,'skipped':'skipped' in t.stderr.lower(),'log':str(out/'tgn_test.txt')}
r['passed']=all(x['ok'] for x in r['imports'].values()) and len(r['configs'])==5 and all(x['ok'] for x in r['configs']) and t.returncode==0 and not r['tgn_test']['skipped']
(out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
sys.exit(0 if r['passed'] else 1)
