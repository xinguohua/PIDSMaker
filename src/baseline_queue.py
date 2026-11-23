"""Single-writer server ledger. No acceptance from process exit or test ranking."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from baseline_matrix import matrix,save,now,digest,ROOT,git

REQUIRED=('full_budget','independent_coverage','model_rebuilt','scores_replayed','decisions_recomputed','source_unchanged')
def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default

def initialize(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return read(path)
    rows=matrix()
    for r in rows:r.update(attempt_ids=[],attempts=[],active_run_id=None,accepted_run_id=None)
    doc={'schema':1,'canonical':str(path.resolve()),'created_at':now(),'rows':rows}
    save(path,doc);return doc

def register(row,run):
    run=Path(run);rid=run.name
    if row['accepted_run_id']:raise ValueError('Already accepted')
    if row['active_run_id'] not in (None,rid):raise ValueError('Already active')
    if rid not in row['attempt_ids']:
        meta=read(run/'meta.json',{})
        row['attempt_ids'].append(rid);row['attempts'].append({'run_id':rid,'path':str(run),'meta':meta,'registered_at':now()})
    row.update(active_run_id=rid,state='active')

def accept(row,attempt,evidence):
    if evidence.get('full_accepted') is not True or not all(evidence.get('checks',{}).get(k) is True for k in REQUIRED):
        raise ValueError('Incomplete full verification evidence')
    if row['accepted_run_id'] and row['accepted_run_id']!=attempt['run_id']:raise ValueError('First accepted attempt is immutable')
    attempt['verification']=evidence
    row.update(state='completed',accepted_run_id=attempt['run_id'],active_run_id=None)

def probe(run):
    run=Path(run);status=read(run/'status.json',{})
    # Existing controller session naming is read from its runner metadata/script.
    sessions=subprocess.run(['tmux','list-panes','-a','-F','#{pane_current_path}'],capture_output=True,text=True).stdout.splitlines()
    alive=any(Path(s)==run/'code' for s in sessions)
    return {'status':status,'tmux_alive':alive,'observed_at':now()}

def persist(path,doc):
    doc['updated_at']=now();save(path,doc)
    out=Path(os.environ.get('RUN_OUTPUT_DIR',str(path.parent)));out.mkdir(parents=True,exist_ok=True)
    save(out/'summary.json',{'state':'queue_monitoring','canonical':str(path),'canonical_sha256':digest(path),'time':now(),'completed':sum(r['state']=='completed' for r in doc['rows']),'rows':doc['rows']})

def launch_verify(run,config,cell):
    destination=run/'verification'/('verify-'+git('rev-parse','--short','HEAD'))
    if not destination.exists():
        destination.mkdir(parents=True)
        spec={'kind':'verify','cell':str(cell),'run':str(run),'config':str(config),'output':str(destination),'code':str(ROOT),'commit':git('rev-parse','HEAD')}
        save(destination/'spec.json',spec)
        cmd=[sys.executable,str(ROOT/'src/baseline_queue.py'),'worker',str(destination/'spec.json')]
        import shlex
        subprocess.run(['tmux','new-session','-d','-s','verify-'+run.name,'-c',str(ROOT),shlex.join(cmd)],check=True)
    return destination

def worker(spec_path):
    spec=read(spec_path);out=Path(spec['output']);env=os.environ.copy();env['RUN_OUTPUT_DIR']=str(out)
    cmd=spec.get('command') or [sys.executable,str(ROOT/'src/verify_native_smoke.py'),'--cell',spec['cell'],'--server-config',spec['config'],'--full']
    env.update(spec.get('environment',{}))
    save(out/'status.json',{'state':'running','pid':os.getpid(),'time':now(),'commit':spec['commit']})
    cmd=[sys.executable,str(ROOT/'src/process_resources.py'),str(out/'resource_evidence'),'--',*cmd]
    with (out/'run.log').open('a') as f:
        p=subprocess.Popen(cmd,env=env,cwd=spec['code'],stdout=f,stderr=subprocess.STDOUT)
        while p.poll() is None:
            free=shutil.disk_usage(out).free
            save(out/'disk.json',{'time':now(),'free_bytes':free,'reserve':4*1024**3})
            if free<4*1024**3:
                save(out/'disk_guard_waiting.json',{'time':now(),'owner':'process_resources supervisor','free_bytes':free})
            time.sleep(2)
        code=p.returncode
    save(out/'status.json',{'state':'finished' if code==0 else 'failed','exit_code':code,'time':now()})

def launch_native(row,path,config,phase,prepared=None):
    import shlex
    root=path.parent.parent
    rid=f"{row['dataset'].lower()}-{row['model']}-{phase}-"+str(time.time_ns())
    run=root/rid;run.mkdir();(run/'output').mkdir()
    # Independent fixed snapshot, never update a running checkout.
    subprocess.run(['git','clone','--quiet','--no-hardlinks',str(ROOT),str(run/'code')],check=True)
    commit=git('rev-parse','HEAD')
    subprocess.run(['git','-C',str(run/'code'),'checkout','--quiet','--detach',commit],check=True)
    command=[sys.executable,'src/baseline_server.py','run','--dataset',row['dataset'],'--model',row['model'],'--mode','full','--server-config',str(config)]
    if phase=='prepare':command+=['--stop-after-preparation']
    if prepared:command+=['--prepared-cell',str(prepared)]
    meta={'run_id':rid,'commit':commit,'command':command,'phase':phase,'created_at':now(),'seed':0}
    save(run/'meta.json',meta)
    spec={'kind':phase,'run':str(run),'config':str(config),'output':str(run),'code':str(run/'code'),'commit':commit,'command':command,'environment':{'RUN_ID':rid,'RUN_OUTPUT_DIR':str(run/'output'),'RUN_DIR':str(run/'output'),'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1'}}
    save(run/'spec.json',spec);save(run/'status.json',{'state':'queued','time':now()})
    register(row,run)
    return run,spec

def start_native(run,spec):
    import shlex
    cmd=[sys.executable,str(Path(spec['code'])/'src/baseline_queue.py'),'worker',str(run/'spec.json')]
    subprocess.run(['tmux','new-session','-d','-s','baseline-'+run.name,'-c',spec['code'],shlex.join(cmd)],check=True)

def queue(args):
    path=args.canonical.resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        doc=initialize(path)
        if args.adopt_run:
            row=next(r for r in doc['rows'] if r['dataset']=='CADETS_E3' and r['model']=='velox')
            if not row['accepted_run_id']:register(row,args.adopt_run)
        while True:
            control=read(path.parent/'control.json',{})
            doc['scheduler']={'paused':control.get('paused',True),'time':now(),'reason':control.get('reason','Fail closed: explicit resume required')}
            for request in control.get('repair_attempts',[]):
                row=next(r for r in doc['rows'] if r['dataset']==request['dataset'] and r['model']==request['model'])
                run=Path(request['path'])
                if run.name not in row['attempt_ids']:
                    register(row,run);row['attempts'][-1]['kind']='prediction_repair'
                    row['attempts'][-1]['original_training_run_id']=request['original_training_run_id']
            for row in doc['rows']:
                if not row['active_run_id']:continue
                attempt=next(a for a in row['attempts'] if a['run_id']==row['active_run_id'])
                run=Path(attempt['path']);observed=probe(run);attempt['observation']=observed
                status=observed['status'];cell=run/'output'/f"{row['dataset']}-{row['model']}-full"
                attempt['pipeline']=read(cell/'status.json',{})
                terminal='exit_code' in status
                if not terminal:
                    if not observed['tmux_alive']:row['state']='active_liveness_unknown'
                    continue
                if attempt.get('kind')=='prediction_repair' and status.get('exit_code')==0:
                    evidence=read(run/'output/summary.json',{})
                    resources=read(run/'output/resources/process_resources_summary.json',{})
                    if evidence.get('repaired_predictions_verified') and resources.get('exit_code')==0 and not resources.get('disk_guard_triggered') and resources.get('sampled_peak_tree_rss_bytes',0)>0:
                        evidence=dict(evidence,full_accepted=True,acceptance_scope='Derived corrected inference with immutable original full training lineage',resource_evidence=resources)
                        try:
                            accept(row,attempt,evidence)
                            attempt['evidence_sha256']=digest(run/'output/summary.json')
                        except ValueError as e:
                            attempt['failure']=str(e);row.update(state='acceptance_failed',active_run_id=None)
                    else:
                        attempt['failure']='Repair replay or real process-resource evidence incomplete';row.update(state='acceptance_failed',active_run_id=None)
                    continue
                if status['exit_code']!=0:
                    attempt['failure']=status;row.update(state='failed',active_run_id=None);continue
                if attempt['meta'].get('phase')=='prepare':
                    if attempt['pipeline'].get('state')=='preparation_complete_pending_capacity':
                        row.update(state='prepared',prepared_cell=str(cell),active_run_id=None)
                    else:
                        attempt['failure']='Preparation did not complete';row.update(state='failed',active_run_id=None)
                    continue
                if attempt['pipeline'].get('state')!='pipeline_finished_pending_acceptance':
                    attempt['failure']='Not a complete native full pipeline';row.update(state='failed',active_run_id=None);continue
                row['state']='pipeline_finished_pending_acceptance'
                if row['model']!='velox':
                    row.update(state='acceptance_blocked',active_run_id=None,blocker='Independent verifier currently supports Velox only');continue
                dest=launch_verify(run,args.server_config,cell);attempt['verification_directory']=str(dest)
                vstatus=read(dest/'status.json',{});attempt['verification_status']=vstatus
                if vstatus.get('exit_code')==0:
                    try:accept(row,attempt,read(dest/'summary.json',{}))
                    except ValueError as e:attempt['failure']=str(e);row.update(state='acceptance_failed',active_run_id=None)
                elif 'exit_code' in vstatus:
                    attempt['failure']='Verification failed; inspect '+str(dest/'run.log');row.update(state='acceptance_failed',active_run_id=None)
            # One native worker at a time; independent missing inputs do not block later pairs.
            if not doc['scheduler']['paused'] and not any(r['active_run_id'] for r in doc['rows']):
                free=shutil.disk_usage(path.parent).free
                server=read(args.server_config)
                for row in doc['rows']:
                    if row['state'] not in ('not_started','prepared','paused_capacity'):continue
                    evidence=Path(server['input_manifests'].get(row['dataset'],'/missing'))
                    manifest=read(evidence,{})
                    if not manifest or 'full' not in manifest.get('allowed_modes',[]):
                        row.update(state='blocked_input',blocker='Missing full-mode input manifest: '+str(evidence));continue
                    if free<5*1024**3:
                        row.update(state='paused_capacity',free_bytes=free,blocker='4GiB reserve plus 1GiB minimum preparation headroom');continue
                    prepared=row.get('prepared_cell')
                    if prepared:
                        report=read(Path(prepared)/'status.json')
                        tensor=report['native_dense_feature_tensor_bytes']
                        eval_edges=sum(report['splits'][s]['edges'] for s in ('val','test'))
                        required=tensor+max(64*1024**2,int(tensor*.02))+eval_edges*160+3*1024**3+64*1024**2+4*1024**3
                        row['capacity']={'free_bytes':free,'estimated_required_bytes':required,'basis':'measured prepared edges and native feature dimension; explicit serialization/fitted/output allowances'}
                        if required>free:
                            row.update(state='paused_capacity',blocker='Measured preparation capacity budget exceeds free bytes');continue
                    if prepared and row['model']!='velox':
                        row.update(state='verification_unsupported',blocker='Full launch disabled until model-specific replay is qualified');continue
                    phase='full' if prepared else 'prepare'
                    run,spec=launch_native(row,path,args.server_config,phase,prepared)
                    # Publish ownership before tmux launch. A crash here is unresolved, never duplicated.
                    persist(path,doc)
                    start_native(run,spec)
                    break
            persist(path,doc)
            if args.once or control.get('stop_controller',False):return
            time.sleep(30)

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(Path(sys.argv[2]))
