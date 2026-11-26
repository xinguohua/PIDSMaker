#!/usr/bin/env python3
"""Auditable E3 baseline jobs. Run training only through the existing remote controller."""
import argparse
import ast
import csv
import hashlib
import json
import math
import numbers
import shutil
import tempfile
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('CADETS_E3', 'THEIA_E3', 'CLEARSCOPE_E3')
MODELS = ('ocrapt', 'velox', 'orthrus', 'magic', 'kairos')
EXPORT_FILES = ('uuid_mapping.json', 'objects.jsonl', 'missing_gt.json', 'metrics.json', 'checkpoint.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def protocol():
    tree = ast.parse((ROOT / 'pidsmaker/config/config.py').read_text())
    data = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'DATASET_DEFAULT_CONFIG' for t in n.targets))
    return {key.value: ast.literal_eval(value) for key, value in zip(data.keys, data.values)
            if key.value in DATASETS}


def matrix():
    rows = [{'dataset': d, 'model': m, 'seed': 0, 'mode': 'full', 'state': 'not_started'}
            for d in DATASETS for m in MODELS]
    rows.sort(key=lambda r: (r['dataset'] != 'CADETS_E3' or r['model'] != 'velox',
                             DATASETS.index(r['dataset']), MODELS.index(r['model'])))
    return rows


def metrics(rows):
    counts = dict(TP=0, FP=0, TN=0, FN=0)
    seen = set()
    for row in rows:
        key = row['node_id']
        if key in seen:
            raise ValueError('Duplicate node in scored universe')
        seen.add(key)
        y, p = int(row['gt']), int(row['prediction'])
        if y not in (0, 1) or p not in (0, 1):
            raise ValueError('Labels must be binary')
        counts[('T' if y == p else 'F') + ('P' if p else 'N')] += 1
    t, f, n, miss = (counts[k] for k in ('TP', 'FP', 'TN', 'FN'))
    div = lambda a, b: a / b if b else None
    return dict(unit='node', positive='malicious', n=len(seen), **counts,
                precision=div(t,t+f), recall=div(t,t+miss), f1=div(2*t,2*t+f+miss),
                fpr=div(f,f+n), fnr=div(miss,t+miss))


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'matrix.json').exists():
        return  # Existing ledger is never reset.
    save(output/'matrix.json', matrix())
    p = protocol()
    for d, v in p.items():
        v['gt_version'] = 'orthrus'
        v['gt_sha256'] = {f: digest(ROOT/'Ground_Truth/orthrus'/f) for f in v['ground_truth_relative_path']}
        v['server_integrity'] = 'unverified'
    save(output/'protocol.json', p)
    save(output/'source.json', {'commit': git('rev-parse','HEAD'), 'created_at': now(),
                               'dirty': bool(git('status','--porcelain')), 'upstream_latest_verified': False})


def run(args):
    # No laptop processing. remote.py supplies these paths on the server data disk.
    if platform.system() != 'Linux' or not os.environ.get('RUN_OUTPUT_DIR') or not os.environ.get('RUN_ID'):
        raise RuntimeError('Run only in a Linux job deployed by remote.py')
    if git('status', '--porcelain'):
        raise RuntimeError('Commit and publish code before deployment')
    out = Path(os.environ['RUN_OUTPUT_DIR']).resolve()
    server = json.loads(args.server_config.read_text())
    disk = Path(server['data_disk_root']).resolve()
    if disk == Path('/') or not out.is_relative_to(disk):
        raise ValueError('RUN_OUTPUT_DIR must be on the configured data disk')
    cell = out / (args.dataset + '-' + args.model + '-' + args.mode)
    cell.mkdir(exist_ok=False)  # no ambiguous restart, cache reuse, or smoke/full collision
    os.chmod(cell, 0o700)
    status = cell/'status.json'
    save(status, {'state':'preparing', 'time':now()})
    try:
        evidence_path = Path(server['input_manifests'][args.dataset])
        evidence = json.loads(evidence_path.read_text())
        db_input = evidence.get('input_kind') == 'existing_postgresql' and evidence.get('database_verified') is True
        if evidence.get('dataset') != args.dataset or not (db_input or evidence.get('integrity_verified') is True or (args.mode == 'smoke' and evidence.get('database_smoke_verified') is True)):
            raise ValueError('A server-verified input/DB manifest is required')
        if args.mode not in evidence.get('allowed_modes', ['smoke', 'full']):
            raise ValueError('Input evidence is not accepted for this run mode')
        for key in (('raw_files', 'database_snapshot', 'date_coverage', 'uuid_mapping_sha256', 'boundary_counts') if args.mode == 'full' and not db_input else ('database_snapshot', 'date_coverage', 'uuid_mapping_sha256', 'boundary_counts')):
            if not evidence.get(key):
                raise ValueError('Missing input evidence: ' + key)
        save(cell/'input_manifest.json', evidence)
        for name in ('tmp','cache','wandb','artifacts'):
            (cell/name).mkdir()
        os.environ.update(TMPDIR=str(cell/'tmp'), XDG_CACHE_HOME=str(cell/'cache'),
                          WANDB_DIR=str(cell/'wandb'), WANDB_MODE='disabled',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if os.environ.get('PYTHONHASHSEED') != '0':
            raise ValueError('Launch with PYTHONHASHSEED=0 (see README)')
        sys.path.insert(0, str(ROOT))
        from pidsmaker.config import get_runtime_required_args, get_yml_cfg
        import torch
        import wandb
        from pidsmaker.main import main
        command = [args.model,args.dataset,'--artifact_dir',str(cell/'artifacts'),
                   '--evaluation.best_model_selection','fixed_budget','--training.seed','0',
                   '--featurization.seed','0','--featurization.training_split','train',
                   '--training_loop.run_evaluation','each_epoch']
        if args.mode == 'smoke':
            command += ['--test_mode','--training.num_epochs','1','--featurization.epochs','1']
        if server['device'] == 'cpu':
            command += ['--cpu']
        elif server['device'] != 'cuda':
            raise ValueError('device must be cpu or cuda')
        parsed, unknown = get_runtime_required_args(True, command)
        if unknown:
            raise ValueError('Unknown framework arguments')
        db = server['database']
        parsed.database_host, parsed.database_user = db['host'], db['user']
        parsed.database_port = db['port']
        parsed.database_password = os.environ[db['password_env']]
        cfg = get_yml_cfg(parsed)
        cfg._feature_cache_format = server.get('feature_cache_formats',{}).get(args.dataset+'/'+args.model,'dense')
        if cfg._feature_cache_format != 'dense' and cfg.featurization.used_method != 'word2vec':
            raise ValueError('Compact baseline cache is qualified only for static word2vec')
        if args.prepared_cell:
            from prepared_reuse import attach_prepared
            attach_prepared(cfg, args.prepared_cell, cell, evidence_path)
        resolved = cfg.clone()
        resolved.database.password = '<environment>'
        (cell/'resolved_config.yml').write_text(resolved.dump())
        save(cell/'manifest.json', {'commit':git('rev-parse','HEAD'), 'task_id':os.environ['RUN_ID'],
             'model':args.model,'dataset':args.dataset,'mode':args.mode,'seed':0,'command':command,
             'input_sha256':digest(evidence_path),'config_sha256':digest(cell/'resolved_config.yml'),
             'protocol':protocol()[args.dataset], 'gt_version':'orthrus',
             'gt_sha256':{f:digest(ROOT/'Ground_Truth/orthrus'/f) for f in cfg.dataset.ground_truth_relative_path},
             'environment':{'pidsmaker_source':__import__('pidsmaker').__file__,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                'device':server['device'],'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None},
             'output':str(cell),'cache_policy':('validated read-only prepared artifacts shared; fresh fitted states and outputs; legacy cache reused bytes=0' if args.prepared_cell else 'fresh isolated artifacts; legacy cache reused bytes=0'),
             'recovery':'retain failed job; deploy a new run_id after fixing cause','created_at':now()})
        with (cell/'packages.txt').open('w') as f:
            subprocess.run([sys.executable,'-m','pip','list','--format=json'],stdout=f,check=True)
        save(status, {'state':'running','time':now(),'pid':os.getpid()})
        wandb.init(mode='disabled')
        if args.stop_after_preparation:
            if args.mode != 'full':
                raise ValueError('Preparation boundary requires unchanged full configuration')
            from native_preparation import prepare_native
            prepare_native(cfg, cell, out, main)
            return
        main(cfg, exp=os.environ['RUN_ID'])
        export(cfg, cell)
        from baseline_knee import DEFAULT_FP_PER_TP, knee_summary, publish_knee
        manifest = json.loads((cell/'manifest.json').read_text())
        calibration = server.get('knee_calibration_manifests', {}).get(f'{args.dataset}/{args.model}')
        knee = publish_knee(cell, manifest, calibration,
                            fp_per_tp=server.get('knee_fp_per_tp', DEFAULT_FP_PER_TP))
        native_metrics = json.loads((cell/'metrics.json').read_text())
        save(status, {'state':'pipeline_finished_pending_acceptance','time':now(),
             'remaining':'verify complete input/split coverage and reconstructed-model reload; smoke is never full'})
        save(out/'summary.json', {'run_id':os.environ['RUN_ID'],'cell':str(cell),
             'status':json.loads(status.read_text()),'native_metrics':native_metrics,
             **knee_summary(knee, cell/'knee')})
    except BaseException as exc:
        # Avoid writing credential-bearing exception messages to small returned results.
        save(status, {'state':'failed','time':now(),'error_type':type(exc).__name__,
                      'recovery':'Inspect server run.log; preserve artifacts and use a new run_id'})
        raise


def load_evaluation_checkpoint(model, checkpoint):
    """Use a freshly reconstructed model; flush before loading saved eval buffers."""
    import torch
    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False))
    return model


def json_value(value, issues, field):
    """Normalize export values; unavailable metadata is null with an audit reason."""
    if value is None:
        issues.append({'field': field, 'reason': 'missing'})
        return None
    # NumPy scalar .item() returns the equivalent Python scalar, without coercing arrays.
    if type(value).__module__.startswith('numpy') and hasattr(value, 'item') and value.ndim == 0:
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        issues.append({'field': field, 'reason': 'non_finite'})
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(v, issues, f'{field}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {str(k): json_value(v, issues, f'{field}.{k}') for k, v in value.items()}
    raise TypeError('Unsupported export type at ' + field)


def object_record(nid, r, n2u, cfg):
    issues = []
    record = json_value({'node_id': nid, 'uuid': n2u.get(int(nid)),
        'score': r.get('score'), 'prediction': r.get('y_hat'), 'gt': r.get('y_true'),
        'score_direction': 'higher_is_anomalous', 'threshold': r.get('decision_threshold'),
        'decision_rule': cfg.evaluation.node_evaluation.threshold_method,
        'post_clustering': cfg.evaluation.node_evaluation.use_kmeans,
        'time_range': r.get('time_range'), 'max_loss_window': r.get('tw_with_max_loss')}, issues, 'object')
    for key in ('node_id', 'prediction', 'gt'):
        if isinstance(record[key], bool) or not isinstance(record[key], numbers.Integral):
            raise ValueError('Missing or invalid ' + key)
        record[key] = int(record[key])
    if record['prediction'] not in (0, 1) or record['gt'] not in (0, 1):
        raise ValueError('Non-binary label')
    if record['score'] is None or isinstance(record['score'], bool) or not isinstance(record['score'], numbers.Real):
        raise ValueError('Missing or non-finite score')
    window = record['max_loss_window']
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, numbers.Real) or int(window) != window:
            raise ValueError('Invalid max_loss_window')
        if window == -1:  # node_evaluation's missing-window sentinel
            record['max_loss_window'] = None
            issues.append({'field': 'object.max_loss_window', 'reason': 'missing_sentinel'})
        elif window < 0:
            raise ValueError('Invalid negative max_loss_window')
        else:
            record['max_loss_window'] = int(window)
    record['export_issues'] = issues
    return record


def export(cfg, cell):
    """Publish only a fully serialized export, with a completion marker written last."""
    names = EXPORT_FILES
    if any((cell / name).exists() for name in (*names, 'export_complete.json')):
        raise FileExistsError('Export already exists; preserve it and use a new run_id')
    stage = Path(tempfile.mkdtemp(prefix='.export-', dir=cell))
    try:
        _export(cfg, stage)
        hashes = {name: digest(stage / name) for name in names}
        for name in names:
            (stage / name).replace(cell / name)
        save(cell / 'export_complete.json', {'files': hashes, 'time': now(),
             'acceptance': 'pending coverage, reconstructed-model predictions and metric checks'})
    except BaseException as exc:
        for name in (*names, 'export_complete.json'):
            (cell / name).unlink(missing_ok=True)
        save(cell / 'export_failed.json', {'error_type': type(exc).__name__, 'time': now(),
             'reason': 'export rejected; no complete object export published'})
        raise
    finally:
        shutil.rmtree(stage)


def _export(cfg, cell):
    import torch
    from pidsmaker.utils.dataset_utils import get_uuid_to_index_id
    from pidsmaker.utils.labelling import ground_truth_files, read_ground_truth_uuids
    epoch = f'model_epoch_{cfg.training.num_epochs-1}'
    results = torch.load(Path(cfg.evaluation._precision_recall_dir)/f'result_{epoch}.pth',weights_only=False)
    checkpoint = Path(cfg.training._trained_models_dir)/epoch/'state_dict.pkl'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not state or not results:
        raise ValueError('Empty checkpoint or scored universe')
    u2n,n2u = get_uuid_to_index_id(cfg)
    save(cell/'uuid_mapping.json',{str(k):v for k,v in n2u.items()})
    records=[]
    for nid,r in results.items():
        records.append(object_record(nid, r, n2u, cfg))
    with (cell/'objects.jsonl').open('w') as f:
        for r in records: f.write(json.dumps(r,allow_nan=False)+'\n')
    ids={r['node_id'] for r in records}
    missing=[]
    for file in ground_truth_files(cfg):
        for uuid,label in read_ground_truth_uuids(cfg,file):
            if uuid not in u2n or u2n[uuid] not in ids:
                missing.append({'uuid':uuid,'gt_file':file,'reason':'absent_from_database' if uuid not in u2n else 'not_scored'})
    save(cell/'missing_gt.json',missing)
    save(cell/'metrics.json',metrics(records))
    save(cell/'checkpoint.json',{'path':str(checkpoint),'sha256':digest(checkpoint),
         'state_dict_deserialized':True,'reconstructed_model_reload_verified':False})
    # Full CSV losses, graph mappings, and preprocessing remain under artifacts/.


def summarize(objects):
    """Reject interrupted or altered publications before computing any metrics."""
    objects = Path(objects)
    cell = objects.parent
    if objects.name != 'objects.jsonl':
        raise ValueError('Summarize requires the published objects.jsonl')
    if (cell / 'export_failed.json').exists():
        raise ValueError('Failed export is not usable')
    status = cell / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('state') == 'failed':
        raise ValueError('Failed cell is not usable')
    marker = cell / 'export_complete.json'
    if not marker.is_file():
        raise ValueError('Missing export completion marker')
    hashes = json.loads(marker.read_text()).get('files')
    if not isinstance(hashes, dict) or set(hashes) != set(EXPORT_FILES):
        raise ValueError('Completion marker must cover every expected export file')
    object_bytes = None
    for name in EXPORT_FILES:
        path = cell / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or indirect export file: ' + name)
        if name == 'objects.jsonl':
            object_bytes = path.read_bytes()
            actual = hashlib.sha256(object_bytes).hexdigest()
        else:
            actual = digest(path)
        if hashes[name] != actual:
            raise ValueError('Export hash mismatch: ' + name)
    # Parse the exact bytes verified above, not a second potentially changed read.
    return metrics(json.loads(line) for line in object_bytes.splitlines())


def cli():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--dataset',choices=DATASETS,required=True)
    r.add_argument('--model',choices=MODELS,required=True);r.add_argument('--mode',choices=['smoke','full'],required=True)
    r.add_argument('--server-config',type=Path,required=True)
    r.add_argument('--stop-after-preparation',action='store_true')
    r.add_argument('--prepared-cell',type=Path)
    s=sub.add_parser('status');s.add_argument('directory',type=Path)
    m=sub.add_parser('summarize');m.add_argument('objects',type=Path)
    for command in ['queue','resume']:
        q=sub.add_parser(command);q.add_argument('--canonical',type=Path,required=True);q.add_argument('--server-config',type=Path,required=True);q.add_argument('--adopt-run',type=Path);q.add_argument('--once',action='store_true')
    q=sub.add_parser('query');q.add_argument('--canonical',type=Path,required=True)
    q=sub.add_parser('verify');q.add_argument('--cell',type=Path,required=True);q.add_argument('--server-config',type=Path,required=True);q.add_argument('--full',action='store_true');q.add_argument('--repair-predictions',action='store_true')
    q=sub.add_parser('control');q.add_argument('--canonical',type=Path,required=True);q.add_argument('--paused',choices=['true','false'],required=True);q.add_argument('--reason',required=True);q.add_argument('--stop-controller',action='store_true')
    a=p.parse_args()
    if a.action=='control':
        path=a.canonical.parent/'control.json'
        value=json.loads(path.read_text()) if path.exists() else {}
        value.update(paused=a.paused=='true',reason=a.reason,stop_controller=a.stop_controller,time=now())
        save(path,value);print(json.dumps(value));return
    if a.action in ['queue','resume']:
        from baseline_queue import queue
        return queue(a)
    if a.action=='query':
        print(a.canonical.read_text());return
    if a.action=='verify':
        cmd=[sys.executable,str(ROOT/'src/verify_native_smoke.py'),'--cell',str(a.cell),'--server-config',str(a.server_config)]
        if a.full:cmd.append('--full')
        if a.repair_predictions:cmd.append('--repair-predictions')
        raise SystemExit(subprocess.call(cmd))
    if a.action=='prepare': prepare(a.output)
    elif a.action=='run': run(a)
    elif a.action=='status':
        print(json.dumps([{'file':str(f),**json.loads(f.read_text())} for f in sorted(a.directory.rglob('status.json'))],indent=2))
    else:
        print(json.dumps(summarize(a.objects),indent=2))

if __name__=='__main__': cli()
