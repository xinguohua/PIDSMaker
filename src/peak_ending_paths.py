"""Classify normal test paths ending on a tied peak event of a false-positive node."""
import csv
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache

from action_paths import pack


def ending_types(incoming, time, source, action):
    """One latest admissible occurrence per predecessor variant suffices for existence."""
    types = {2:set(),3:set()}
    for (middle,b),times in incoming.get(source,{}).items():
        j = bisect_left(times,time)
        if j:
            types[2].add((b,action))
            latest = times[j-1]
            for (_,a),earlier in incoming.get(middle,{}).items():
                if earlier[0] < latest:
                    types[3].add((a,b,action))
    return types


def classify(peak_path,preds,groups,counts,db,out):
    peaks=defaultdict(lambda:defaultdict(set))
    target={n for n,v in preds.items() if v==(0,1)}
    pairs=0
    with peak_path.open() as stream:
        for r in csv.DictReader(stream):
            w,row,n=map(int,(r['w'],r['row'],r['node_id']))
            assert n in target and n not in peaks[w][row]
            peaks[w][row].add(n)
            pairs+=1
    assert target==set.union(*(ns for p in peaks.values() for ns in p.values()))

    @lru_cache(maxsize=100000)
    def bit(seq):
        result=counts.execute('select train,test from counts where k=? and seq=?',(len(seq),pack(seq))).fetchone()
        assert result and int(result[1])>0, 'Extracted path absent from frozen test counts'
        return 1 if int(result[0])>0 else 2

    flags={n:{2:0,3:0} for n in target}
    eligible=set()
    excluded_pairs=0
    for w,anchors in sorted(peaks.items()):
        raw={row:(u,v) for row,u,v in db.execute('select row,u,v from edges where w=?',(w,)) if row in anchors}
        assert raw.keys()==anchors.keys() and all(ns.issubset(raw[row]) for row,ns in anchors.items())
        incoming=defaultdict(lambda:defaultdict(list))
        found=set()
        for t,batch in enumerate(groups(w)):
            # Test peaks before inserting this time group so tied events never extend a path.
            for row,u,v,a in batch:
                if row not in anchors:
                    continue
                found.add(row)
                ns=anchors[row]
                eligible.update(ns)
                for k,seqs in ending_types(incoming,t,u,a).items():
                    mask=0
                    for seq in seqs:
                        mask |= bit(seq)
                        if mask==3:
                            break
                    for n in ns:
                        flags[n][k] |= mask
            for _,u,v,a in batch:
                times=incoming[v][u,a]
                if not times or times[-1]!=t:
                    times.append(t)
        excluded_pairs+=sum(len(anchors[row]) for row in anchors.keys()-found)
    with (out/'FP_AT_PEAK.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('length','group','fp_nodes','total_fp','percent'))
        for k in (2,3):
            hist=Counter(byk[k] for byk in flags.values())
            assert sum(hist.values())==len(target)
            for mask,label in [(1,'seen_only'),(2,'unseen_only'),(3,'both'),(0,'neither')]:
                writer.writerow((k,label,hist[mask],len(target),100*hist[mask]/len(target)))
    assert all(not g[3] or g[2] for g in flags.values())
    (out/'PEAK_CONTEXT_AUDIT.json').write_text(json.dumps(dict(
        fp_nodes=len(target),node_peak_pairs=pairs,eligible_nodes=len(eligible),
        filtered_node_peak_pairs=excluded_pairs,windows=len(peaks),
        rule='All tied maxima; path ends at peak; same native window and strictly ordered groups; train count > 0 means seen'
    ),indent=2))
