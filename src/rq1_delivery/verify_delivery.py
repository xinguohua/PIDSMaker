"""Audit the delivered CSVs without enumerating event paths a second time."""
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent

def rows(name):
    with (root/name).open() as stream:
        yield from csv.DictReader(stream)

def as_int(row, key):
    return int(row[key])

summary = {int(r['length']): r for r in rows('COMPOSITION_SUMMARY.csv')}
assert set(summary) == {1, 2, 3}
audit = json.loads((root/'INPUT_AUDIT.json').read_text())
assert audit['native_rows_verified']
by_split = {s: (w,n) for s,w,n in audit['window_counts']}
window_stats = list(rows('WINDOW_AUDIT.csv'))
for split, (nwin, nevent) in by_split.items():
    windows = [r for r in window_stats if r['split'] == split]
    assert len(windows) == len({r['window'] for r in windows}) == nwin
    assert sum(as_int(r,'native_edges') for r in windows) == nevent
    for r in windows:
        assert as_int(r,'native_edges') == sum(as_int(r,x) for x in ('attack_edges','unknown_label_edges','normal_edges'))
    for k,r in summary.items():
        assert as_int(r,split+'_instances') == sum(as_int(w,'instances_'+str(k)) for w in windows)
actions = {int(r['action_id']): (int(r['train_frequency']),int(r['test_frequency'])) for r in rows('ACTION_COUNTS.csv')}
for i, split in enumerate(('train','test')):
    assert sum(n[i] for n in actions.values()) == as_int(summary[1],split+'_instances')
for k,r in summary.items():
    for suffix in ('known_semantics','observable_signatures'):
        candidate = as_int(r,'candidate_'+suffix)
        assert all(as_int(r,part+'_'+suffix)>=0 for part in ('Seen','Zero_shot','Potential'))
        assert candidate == sum(as_int(r,part+'_'+suffix) for part in ('Seen','Zero_shot','Potential'))
    if k==1:
        continue
    totals=Counter();all_actions=Counter();zeroshot=Counter();prev=None
    for row in rows(f'COMPOSITION_COUNTS/length_{k}.csv'):
        seq=json.loads(row['action_ids']);train=int(row['train_frequency']);test=int(row['test_frequency'])
        assert len(seq)==k and (prev is None or prev>=train)
        prev=train
        assert [actions[t][0] for t in seq]==json.loads(row['action_train_frequencies'])
        totals.update(dict(train_instances=train,test_instances=test,train_observed_types=int(train>0),
                           test_observed_types=int(test>0)))
        if test:
            for t in seq:all_actions[t]+=test
        if test and not train:
            key='semantic_unknown' if not int(row['semantic_complete']) else 'all_actions_seen' if all(actions[t][0] for t in seq) else 'unseen_action'
            totals[key+'_instances']+=test;totals[key+'_types']+=1
            for t in seq:zeroshot[t]+=test
    for name in ('train_instances','test_instances','train_observed_types','test_observed_types'):
        assert totals[name]==as_int(r,name)
    dist=list(rows(f'ACTION_DISTRIBUTIONS/length_{k}_full.csv'))
    assert dist[:25]==list(rows(f'ACTION_DISTRIBUTIONS/length_{k}_top25.csv'))
    assert {int(v['action_id']):int(v['All_count']) for v in dist}==dict(all_actions)
    assert all(int(v['Zero_shot_count'])==zeroshot[int(v['action_id'])] for v in dist)
    for v in rows(f'ACTION_DISTRIBUTIONS/length_{k}_zero_shot_groups.csv'):
        for name in ('types','instances'):assert int(v[name])==totals[v['group']+'_'+name]
baseline=json.loads((root/'OPERATING_POINT.json').read_text())['exploratory_metrics']
assert [baseline[key] for key in ('TP','FP','TN','FN')]==[32,1174,285684,40]
import sqlite3
with sqlite3.connect('file:'+str(root.parent/'counts.sqlite')+'?mode=ro',uri=True) as db:
    record=json.loads(db.execute('select audit from window_done where w=0').fetchone()[0])
    assert json.loads((root/'REAL_PATH_CHECK.json').read_text())==dict(window=0,**record['reference'])
assert {(r['split'],r['window']) for r in window_stats}=={(r['split'],r['w']) for r in rows('NATIVE_WINDOWS.csv')}
for k in (2,3):
    groups={r['group']: r for r in rows('DETECTION_PERFORMANCE.csv') if int(r['length'])==k}
    assert int(groups['All']['FP'])==baseline['FP'] and int(groups['All']['TN'])==baseline['TN']
    for field in ('nodes','FP','TN'):
        assert int(groups['All'][field])==sum(int(groups[g][field]) for g in ('Seen','Zero-shot','Other'))
    assert (int(groups['attack_context_unresolved']['TP']),int(groups['attack_context_unresolved']['FN']))==(baseline['TP'],baseline['FN'])
print('Delivered counts, ranks, both candidate spaces, constituent frequencies and detection groups verified.')

# Validate every scored node against the frozen knee decision digest at each length.
import hashlib
knee=json.loads((root/'OPERATING_POINT.json').read_text())
from collections import defaultdict
labels=defaultdict(dict)
normal_counts=Counter()
for row in rows('NODE_SCORES.csv'):
    k,n=int(row['length']),int(row['node_id'])
    assert n not in labels[k]
    pred=int(row['prediction']);gt=int(row['gt'])
    assert float(row['threshold'])==knee['policy']['threshold']
    assert pred==int(float(row['score'])>float(row['threshold']))
    labels[k][n]=pred
    if not gt:
        bits=int(row['membership_bits'])
        assert row['group']==('Zero-shot' if bits&2 else 'Seen' if bits==1 else 'Other')
        normal_counts[k,row['group'],pred]+=1
for k, mapping in labels.items():
    assert len(mapping)==baseline['n']
    h=hashlib.sha256()
    for n in sorted(mapping):h.update(f'{n}\t{mapping[n]}\n'.encode())
    assert h.hexdigest()==knee['decisions_sha256']
    for row in rows('DETECTION_PERFORMANCE.csv'):
        if int(row['length'])==k and row['group'] in ('Seen','Zero-shot','Other'):
            assert normal_counts[k,row['group'],1]==int(row['FP'])
            assert normal_counts[k,row['group'],0]==int(row['TN'])
for k in (1,2,3):
    source={r['group']:int(r['instances']) for r in rows(f'ACTION_DISTRIBUTIONS/length_{k}_zero_shot_groups.csv')}
    actual=Counter()
    for r in rows(f'ACTION_DISTRIBUTIONS/length_{k}_source_actions.csv'):actual[r['group']]+=int(r['position_count'])
    assert all(actual[g]==k*n for g,n in source.items())
    ds=list(rows(f'ACTION_DISTRIBUTIONS/length_{k}_full.csv'))
    for col,total in [('All',k*int(summary[k]['test_instances'])),('Zero_shot',k*sum(source.values()))]:
        assert sum(int(r[col+'_count']) for r in ds)==total
        if total:assert math.isclose(math.fsum(float(r[col+'_fraction']) for r in ds),1.0)
print('Knee decisions, disjoint membership, weighted distributions and recoverable reference verified.')

native=sqlite3.connect('file:'+str(root.parent.parent/'native.sqlite')+'?mode=ro&immutable=1',uri=True)
expected_normal={n for n, in native.execute('select id from pred where gt=0')}
expected_gt={n:gt for n,gt in native.execute('select id,gt from pred')}
matched=defaultdict(set)
for row in rows('NODE_SCORES.csv'):
    n=int(row['node_id']);gt=int(row['gt'])
    assert expected_gt[n]==gt
    if gt==0:matched[int(row['length'])].add(n)
assert set(matched)=={2,3} and all(group==expected_normal for group in matched.values())
native.close()
cert=json.loads((root/'NATIVE_INPUT_AUDIT.json').read_text())
assert (cert['native_loader_train_batches'],cert['logged_train_batches'])==(3665,3665)
print('Exact normal node sets and original loader batches verified.')
