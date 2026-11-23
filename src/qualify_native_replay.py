"""Serial real-data smoke/reconstruction qualification, never formal completion."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from baseline_matrix import ROOT,save,now,git
p=argparse.ArgumentParser();p.add_argument('--server-config',type=Path,required=True);p.add_argument('--wait-run',type=Path,required=True);a=p.parse_args()
out=Path(os.environ['RUN_OUTPUT_DIR']);summary={'state':'waiting_for_existing_repair','commit':git('rev-parse','HEAD'),'formal_completion':False,'models':{}}
save(out/'summary.json',summary)
while True:
 s=json.loads((a.wait_run/'status.json').read_text())
 if 'exit_code' in s:break
 time.sleep(15)
for model in ['ocrapt','orthrus','magic','kairos']:
 record=summary['models'].setdefault(model,{})
 for phase in ['smoke','verify']:
  phase_out=out/model/phase;phase_out.mkdir(parents=True,exist_ok=False)
  env=os.environ.copy();env.update(RUN_ID=os.environ['RUN_ID']+'-'+model+'-'+phase,RUN_OUTPUT_DIR=str(phase_out),RUN_DIR=str(phase_out))
  if phase=='smoke':command=[sys.executable,str(ROOT/'src/baseline_server.py'),'run','--dataset','CADETS_E3','--model',model,'--mode','smoke','--server-config',str(a.server_config)]
  else:command=[sys.executable,str(ROOT/'src/verify_native_smoke.py'),'--cell',str(out/model/'smoke'/('CADETS_E3-'+model+'-smoke')),'--server-config',str(a.server_config)]
  record.update(state=phase+'_running',time=now(),output=str(phase_out));summary['state']='qualifying';save(out/'summary.json',summary)
  with (phase_out/'run.log').open('x') as log:
   result=subprocess.run([sys.executable,str(ROOT/'src/process_resources.py'),str(phase_out/'resources'),'--',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  record[phase]={'exit_code':result.returncode,'output':str(phase_out),'time':now()}
  if result.returncode:
   record['state']=phase+'_failed';break
  if phase=='verify':
   evidence=json.loads((phase_out/'summary.json').read_text());record['state']='bounded_replay_qualified' if evidence['state']=='reconstruction_replay_verified' else 'verification_failed';record['formal_accepted']=False
  save(out/'summary.json',summary)
 save(out/'summary.json',summary)
summary['state']='qualification_finished';save(out/'summary.json',summary)
