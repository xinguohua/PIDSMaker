"""Check whether any tied peak row participates in an all-actions-seen zero-shot path."""
import csv
from collections import defaultdict
from pathlib import Path
from action_index import ActionIndex
from action_paths import unpack


def check(peak_rows, memberships, out, counts, test_frequency, train_actions, groups):
    target=defaultdict(set)
    with memberships.open() as f:
        for r in csv.DictReader(f):
            if r['group']=='all_actions_seen':
                target[int(r['node_id'])].add(int(r['length']))
    peaks=defaultdict(lambda:defaultdict(set))
    with peak_rows.open() as f:
        for r in csv.DictReader(f):
            n=int(r['node_id'])
            if n in target:
                peaks[int(r['w'])][int(r['row'])].add(n)
    assert target.keys()=={n for rows in peaks.values() for ns in rows.values() for n in ns}
    candidates=defaultdict(list)
    for k in sorted(set.union(*target.values())):
        for blob, in counts.execute("select seq from counts where k=? and train='0' and test!='0'",(k,)):
            seq=unpack(blob)
            if all(a in train_actions for a in seq):
                candidates[min(seq,key=test_frequency.__getitem__)].append(seq)
    hit=defaultdict(set)
    for w,rows in sorted(peaks.items()):
        batches=groups(w)
        idx=ActionIndex(batches)
        locations=defaultdict(list)
        labels=set()
        for label,variants in idx.variants.items():
            for edge in variants:
                for j,ids in enumerate(idx.events[edge]):
                    for row in ids:
                        if row in rows:
                            locations[edge].append((j,rows[row]))
                            labels.add(label)
        for anchor in idx.variants.keys() & candidates.keys():
            for seq in candidates[anchor]:
                if labels.isdisjoint(seq) or not all(a in idx.variants for a in seq):
                    continue
                for edge,lo,hi in idx.bounds(seq):
                    for j,ns in locations.get(edge,()):
                        if lo<=j<hi:
                            hit[len(seq)].update(n for n in ns if len(seq) in target[n])
    with (out/'PEAK_ON_FOCAL_PATH.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(('length','fp','peak_on_one_or_more_focal_paths','all_peaks_outside_focal_paths'))
        for k in sorted(set.union(*target.values())):
            total=sum(k in ks for ks in target.values())
            wr.writerow((k,total,len(hit[k]),total-len(hit[k])))
