"""Default operating-point export for project baseline runs and their replays."""
import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

from baseline_matrix import digest, save
from threshold_knee import DEFAULT_FP_PER_TP, METHOD, apply_knee, fit_knee


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def calibration_source(path, manifest, checkpoint, test, test_hash):
    path = Path(path)
    meta = json.loads(path.read_text())
    if (meta['split'] != 'calibration' or meta['model'] != manifest['model']
            or meta['dataset'] != manifest['dataset'] or meta['checkpoint_sha256'] != checkpoint):
        raise ValueError('Calibration split, model, dataset or checkpoint mismatch')
    dates = meta['dates']
    if not isinstance(dates, list) or not dates or any(date.fromisoformat(d).isoformat() != d for d in dates):
        raise ValueError('Calibration dates must be nonempty ISO calendar dates')
    protocol = manifest['protocol']
    if set(dates).intersection(protocol['train_dates'] + protocol['test_dates']):
        raise ValueError('Calibration overlaps training or test dates')
    source = (path.parent / meta['objects']).resolve()
    checksum = digest(source)
    if source.samefile(test) or checksum == test_hash or checksum != meta['objects_sha256']:
        raise ValueError('Calibration must match its digest and be independent of test data')
    return source, dict(manifest_sha256=digest(path), objects_sha256=checksum, dates=dates)


def publish_knee(cell, manifest, calibration_manifest=None, *, frozen=None, fp_per_tp=DEFAULT_FP_PER_TP, output=None):
    """Fit once, publish separate decisions, and reuse the frozen policy on replay."""
    target = cell / 'knee' if output is None else Path(output).resolve()
    if output is not None and target.is_relative_to(cell.resolve()):
        raise ValueError('Use a separate output directory for an existing export')
    if target.exists():
        raise FileExistsError(target)
    test = cell / 'objects.jsonl'
    native = json.loads((cell / 'export_complete.json').read_text())['files']
    test_hash = digest(test)
    checkpoint_file = cell / 'checkpoint.json'
    if test_hash != native['objects.jsonl'] or digest(checkpoint_file) != native['checkpoint.json']:
        raise ValueError('Native score export or checkpoint record hash mismatch')
    checkpoint = json.loads(checkpoint_file.read_text())['sha256']
    identity = {key: manifest[key] for key in ('model', 'dataset', 'task_id')}
    identity['checkpoint_sha256'] = checkpoint
    if frozen is not None:
        if frozen['identity'] != identity:
            raise ValueError('Frozen policy does not belong to this model run')
        policy, scope, source = frozen['policy'], frozen['scope'], frozen['source']
    elif calibration_manifest is not None:
        calibration, source = calibration_source(calibration_manifest, manifest, checkpoint, test, test_hash)
        policy = fit_knee(rows(calibration), fp_per_tp=fp_per_tp)
        scope = 'independent_calibration'
    else:
        # Annotated test data can illustrate a working point but cannot validate it.
        policy = fit_knee(rows(test), fp_per_tp=fp_per_tp)
        scope = 'retrospective_test'
        source = dict(objects_sha256=test_hash, dates=manifest['protocol']['test_dates'])

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.knee-', dir=target.parent))
    try:
        evaluated = decision_hash = None
        if policy['status'] == 'selected':
            threshold = policy['threshold']
            decisions = {}
            with (stage / 'objects.jsonl').open('w') as output:
                def records():
                    for row in rows(test):
                        label = int(row['score'] > threshold)
                        decisions[row['node_id']] = label
                        record = dict(node_id=row['node_id'], uuid=row.get('uuid'), score=row['score'],
                                      gt=row['gt'], prediction=label, threshold=threshold,
                                      decision_rule=METHOD, scope=scope)
                        output.write(json.dumps(record, allow_nan=False) + '\n')
                        yield row
                evaluated = apply_knee(records(), threshold)
            h = hashlib.sha256()
            for node, prediction in sorted(decisions.items()):
                h.update(f'{node}\t{prediction}\n'.encode())
            decision_hash = h.hexdigest()
        formal = scope == 'independent_calibration'
        report = dict(default_policy=METHOD, scope=scope, identity=identity, source=source,
                      policy=policy, test_objects_sha256=test_hash, decisions_sha256=decision_hash,
                      metrics=evaluated if formal else None, exploratory_metrics=None if formal else evaluated)
        save(stage/'report.json', report)
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True, help='Existing published node export')
    parser.add_argument('--manifest', type=Path, required=True, help='Training cell manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New derived result directory')
    parser.add_argument('--calibration-manifest', type=Path, help='Independent labeled scores manifest')
    parser.add_argument('--fp-per-tp', default=DEFAULT_FP_PER_TP, help='False-alert cost allowance')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    training = Path(manifest['output']).resolve()
    checkpoint = json.loads((args.cell/'checkpoint.json').read_text())
    if not Path(checkpoint['path']).resolve().is_relative_to(training):
        raise ValueError('Node export checkpoint does not belong to the supplied training run')
    if args.output.resolve().is_relative_to(training):
        raise ValueError('Derived output must be separate from the training run')
    result = publish_knee(args.cell, manifest, args.calibration_manifest,
                          fp_per_tp=args.fp_per_tp, output=args.output)
    print(json.dumps({'report':str(args.output/'report.json'), 'scope':result['scope'],
                      'status':result['policy']['status'], 'threshold':result['policy']['threshold']}))


if __name__ == '__main__':
    main()
