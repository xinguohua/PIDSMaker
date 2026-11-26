"""Verify and share this project's completed native full preparation, never legacy caches."""
import json,os,shutil,threading,time
from pathlib import Path
import yaml
from baseline_matrix import ROOT,digest,save,now

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def graph_source_hashes(root):
    package = root/'pidsmaker'
    mandatory = (
        'tasks/construction.py', 'tasks/transformation.py',
        'preprocessing/build_graph_methods/build_default_graphs.py',
        'preprocessing/build_graph_methods/build_magic_graphs.py',
        'preprocessing/cleaning.py', 'utils/utils.py', 'utils/dataset_utils.py',
        'config/pipeline.py', 'config/config.py', 'mimicry.py',
    )
    paths = {package/name for name in mandatory}
    for folder in ('preprocessing/build_graph_methods', 'preprocessing/transformation_methods'):
        if not any((package/folder).glob('*.py')):
            raise ValueError(f'Missing graph source directory: {package/folder}')
    paths.update(p for p in (package/'preprocessing').rglob('*')
                 if p.is_file() and p.suffix in {'.py', '.yml', '.yaml'})
    return {str(p.relative_to(root)):digest(p) for p in sorted(paths)}


def verify_graph_sources(source, current):
    previous = graph_source_hashes(source)
    expected = graph_source_hashes(current)
    changed = sorted(k for k in previous.keys() | expected.keys() if previous.get(k) != expected.get(k))
    if changed:
        raise ValueError('Native graph source differs: ' + ', '.join(changed))
    return previous


def attach_prepared(cfg,source,cell,evidence_path):
    source=source.resolve()
    if cfg._test_mode:raise ValueError('Full preparation cannot be attached to smoke')
    status=json.loads((source/('preparation_recovery.json' if (source/'preparation_recovery.json').exists() else 'status.json')).read_text())
    if status['state']!='preparation_complete_pending_capacity':raise ValueError('Preparation is not complete')
    meta=json.loads((source/'manifest.json').read_text())
    previous=yaml.safe_load((source/'resolved_config.yml').read_text())
    assert meta['input_sha256']==digest(evidence_path), 'Input manifest differs'
    assert not previous['_test_mode']
    for name in ['dataset','construction','transformation']:
        old, current = clean(previous[name]), clean(dict(getattr(cfg,name)))
        if name == 'construction':
            if Path(previous[name]['_task_path']).name != Path(cfg.construction._task_path).name:
                raise ValueError('Prepared graph input fingerprint differs')
            if old.get('file_path_evidence') is None:
                old.pop('file_path_evidence', None)
        assert old==current, 'Effective config differs: '+name
    code=source.parent.parent/'code'
    checked_source=verify_graph_sources(code, ROOT)
    rows=[json.loads(x) for x in (source/'windows.jsonl').read_text().splitlines()]
    bydate={r['date'] for r in rows}
    expected=set(cfg.dataset.train_dates+cfg.dataset.val_dates+cfg.dataset.test_dates)
    # Native construction emits no graph for an empty configured date.
    assert bydate<=expected, 'Unexpected graph date outside configured splits'
    files={};totals={}
    for r in rows:
        f=Path(r['path']).resolve();assert f.is_relative_to(source)
        assert f.stat().st_size==r['bytes']
        files[str(f)]=digest(f)
    for split in ['train','val','test']:
        rr=[r for r in rows if r['split']==split]
        totals[split]={k:sum(r[k] for r in rr) for k in ['edges','nodes','bytes','input_events_after_type_filter','events_after_dedup']}
        totals[split]['windows']=len(rr)
        assert totals[split]==status['splits'][split]
    # Native payload is calculated from actual fused edges and float32/int64 shapes.
    from native_preparation import feature_storage_capacity
    cache_format=getattr(cfg,'_feature_cache_format','dense')
    tensor=feature_storage_capacity(status,cache_format)
    if tensor is None:raise ValueError('Native model-specific feature shape not measured; no Velox capacity extrapolation')
    eval_edges=sum(totals[k]['edges'] for k in ['val','test'])
    # Explicit estimated allowances, not claimed measurements: CSV lines, object exports,
    # fitted word2vec vocabulary/state and native serialization overhead.
    estimate={'tensor_payload_upper_bound':tensor,'tensor_serialization_allowance':max(64*1024**2,int(tensor*.02)),
      'val_test_csv_allowance':eval_edges*160,'fitted_feature_states_allowance':128*1024**2,
      'mapping_object_metrics_allowance':256*1024**2,'checkpoint_allowance':64*1024**2,
      'reserve':512*1024**2}
    free=shutil.disk_usage(cell).free
    report={'time':now(),'source':str(source),'source_commit':meta['commit'],'input_sha256':meta['input_sha256'],'feature_cache_format':cache_format,
      'native_source_hashes':checked_source,'graph_hashes':files,'splits':totals,'capacity':estimate,
      'free_bytes':free,'required_additional_with_reserve':sum(estimate.values()),
      'configured_dates_without_graphs':sorted(expected-bydate),'links':[]}
    save(cell/'prepared_reuse.json',report)
    if sum(estimate.values())>free:raise RuntimeError('Measured fused payload plus explicit allowances exceeds free space; see prepared_reuse.json')
    for task,children in [('construction',['nx','indexid2msg']),('transformation',['nx'])]:
        old=Path(previous[task]['_task_path']);new=Path(getattr(cfg,task)._task_path)
        assert old.is_relative_to(source) and (old/'done.txt').is_file()
        for child in children:
            src=(old/child).resolve();assert src.is_relative_to(source)
            dst=new/child
            if dst.exists() or dst.is_symlink():raise FileExistsError(str(dst))
            for f in src.rglob('*'):
                if f.is_file() and not f.is_symlink():
                    files.setdefault(str(f),digest(f) if str(f) not in files else files[str(f)]);f.chmod(0o444)
            for d in sorted((d for d in src.rglob('*') if d.is_dir()),reverse=True):d.chmod(0o555)
            src.chmod(0o555)
            dst.symlink_to(src,target_is_directory=True)
            report['links'].append({'source':str(src),'target':str(dst),'mode':'read-only shared'})
        shutil.copyfile(old/'done.txt',new/'done.txt')
    report['shared_files_hashes']=files;report['verified']=True
    save(cell/'prepared_reuse.json',report)

    def watch_disk():
        while True:
            free=shutil.disk_usage(cell).free
            save(cell/'disk_progress.json',{'time':now(),'free_bytes':free,'reserve_bytes':estimate['reserve'],'phase':'full native pipeline; fitted states and outputs retained'})
            if free < estimate['reserve']:
                failure={'state':'stopped_disk_reserve','time':now(),'free_bytes':free,'no_deletions':True}
                save(cell/'status.json',failure);save(cell.parent/'summary.json',failure);os._exit(74)
            time.sleep(2)
    threading.Thread(target=watch_disk,daemon=True).start()
