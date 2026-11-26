"""Aggregate frozen native node flags; never enumerate paths or change decisions."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(k, profile):
    def row(group, predicate):
        n = sum(v[0] for b, v in profile.items() if predicate(b))
        fp = sum(v[1] for b, v in profile.items() if predicate(b))
        return dict(length=k, group=group, nodes=n, FP=fp, TN=n-fp,
                    FPR_pct=100*fp/n if n else None)
    main = [row(name, lambda b, bit=bit: bool(b & bit)) for bit, name in
            [(1, 'Seen Composition'), (2, 'Novel Composition'),
             (4, 'Novel Action Composition')]]
    overlaps = [row(name, lambda b, mask=mask: b & mask == mask) for mask, name in
                [(3, 'Seen & Novel'), (5, 'Seen & Novel Action'),
                 (6, 'Novel & Novel Action'), (7, 'All three')]]
    exclusive = [row(f'complete_membership_{mask}', lambda b, mask=mask: b & 7 == mask)
                 for mask in range(8)]
    incomplete = [row('Incomplete participation', lambda b: bool(b & 8)),
                  row('Incomplete only', lambda b: b == 8),
                  row('Incomplete & any complete', lambda b: bool(b & 8 and b & 7))]
    raw = [row(f'raw_bits_{mask}', lambda b, mask=mask: b == mask) for mask in range(1, 16)]
    union = row('Complete union', lambda b: bool(b & 7))
    assert union['nodes'] == sum(r['nodes'] for r in main)-sum(r['nodes'] for r in overlaps[:3])+overlaps[3]['nodes']
    assert union['FP'] == sum(r['FP'] for r in main)-sum(r['FP'] for r in overlaps[:3])+overlaps[3]['FP']
    assert sum(r['nodes'] for r in exclusive) == sum(v[0] for v in profile.values())
    return dict(main=main, intersections=overlaps, exclusive=exclusive,
                incomplete=incomplete, raw_flags=raw, union=[union])


def run(args):
    old = json.loads(args.previous.read_text())
    paths = {k: Path(old['paths'][k]) for k in ('compositions', 'predictions', 'cell')}
    comp, pred, cell = (paths[k] for k in ('compositions', 'predictions', 'cell'))
    report = json.loads((pred/'report.json').read_text())
    for key in ('identity', 'policy', 'scope', 'decisions_sha256'):
        assert old[key] == report[key], key
    assert report['identity']['dataset'] == 'CADETS_E3' and report['identity']['model'] == 'velox'
    assert report['scope'] == 'retrospective_test'
    assert report['policy']['threshold'] == 4.268208026885985
    assert report['policy']['comparison'] == 'score > threshold'
    db = sqlite3.connect(f'file:{comp}/counts.sqlite?mode=ro', uri=True)
    binding = json.loads(db.execute('select binding from protocol').fetchone()[0])
    assert binding == json.loads((comp/'PROTOCOL.json').read_text())
    assert binding['protocol'] == 'native_paths_len3_v2'
    assert binding['decision_hash'] == report['decisions_sha256']
    for name in ('action_window.py', 'action_index.py'):
        assert sha(Path(__file__).with_name(name)) == binding['code'][name]
    for path in (pred/'report.json', pred/'objects.jsonl', cell/'manifest.json'):
        assert sha(path) == binding['inputs'][str(path)]
    manifest = json.loads((cell/'manifest.json').read_text())
    assert manifest['task_id'] == report['identity']['task_id']
    assert json.loads((cell/'checkpoint.json').read_text())['sha256'] == report['identity']['checkpoint_sha256']
    decisions = {}
    with (pred/'objects.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            n = r['node_id']
            assert n not in decisions
            assert r['threshold'] == report['policy']['threshold']
            assert r['prediction'] == int(r['score'] > r['threshold'])
            assert r['gt'] in (0, 1) and r['prediction'] in (0, 1)
            decisions[n] = (r['gt'], r['prediction'])
    digest = hashlib.sha256()
    for n, (_, p) in sorted(decisions.items()):
        digest.update(f'{n}\t{p}\n'.encode())
    assert digest.hexdigest() == binding['decision_hash']
    matrix = Counter(decisions.values())
    assert [matrix[x] for x in [(1, 1), (0, 1), (0, 0), (1, 0)]] == [32, 1174, 285684, 40]
    result = {}
    flag_hashes = {}
    for k in (2, 3):
        profile = {}
        digest = hashlib.sha256()
        previous = None
        for n, b in db.execute('select id,bits from node_flags where k=? order by id', (k,)):
            assert previous is None or n > previous
            previous = n
            assert 1 <= b <= 15 and decisions[n][0] == 0
            digest.update(f'{k}\t{n}\t{b}\n'.encode())
            v = profile.setdefault(b, [0, 0])
            v[0] += 1
            v[1] += decisions[n][1]
        result[k] = rows_for(k, profile)
        flag_hashes[k] = digest.hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in result[2]:
        rows = result[2][name] + result[3][name]
        with (args.output/f'{name}.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    summary = dict(results=result, operating_point=report, source_paths={k:str(v) for k,v in paths.items()},
                   previous_experiment=str(args.previous), protocol=binding, train_manifest=manifest,
                   flag_sha256=flag_hashes, code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   script_sha256=sha(Path(__file__)), run_id=os.environ.get('RUN_ID'),
                   checks='unique native IDs; benign labels; frozen hashes and decisions; inclusion-exclusion; partition totals passed',
                   semantics={'1':'complete; train>0','2':'complete; train=0; all actions train>0',
                              '4':'complete; train=0; at least one action train=0','8':'incomplete'},
                   note='Main node sets overlap. Incomplete paths add no main bit; nodes also participating in complete paths retain those valid memberships. Exclusive masks use bits & 7; mask 0 is incomplete-only within the participation universe.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(os.environ.get('RUN_OUTPUT_DIR', '.')))
    # Small independent mask check, including incomplete and overlapping membership.
    check = rows_for(2, {1:[1,0], 3:[1,1], 7:[1,0], 8:[1,1], 12:[1,1]})
    assert [r['nodes'] for r in check['main']] == [3,2,2]
    assert [r['FP'] for r in check['main']] == [1,1,1]
    run(p.parse_args())
