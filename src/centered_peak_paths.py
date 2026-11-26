"""Recover exact type support for native normal paths around tied FP peak rows."""
import csv
from collections import Counter, defaultdict
from functools import lru_cache

from action_paths import pack


def centered(groups, anchors):
    incoming, outgoing = defaultdict(list), defaultdict(list)
    peaks = {}
    for t, batch in enumerate(groups):
        for row, u, v, a in batch:
            incoming[v].append((t, u, a))
            outgoing[u].append((t, v, a))
            if row in anchors:
                assert anchors[row] <= {u, v}
                peaks[row] = t, u, v, a
    for row, (t, u, v, a) in peaks.items():
        # Latest predecessor and earliest successor preserve all second-hop types.
        left = {(x,b):s for s,x,b in incoming[u] if s<t}
        right = {(y,c):s for s,y,c in reversed(outgoing[v]) if s>t}
        before = {b for _,b in left}
        after = {c for _,c in right}
        pairs = {(b,a) for b in before} | {(a,c) for c in after}
        if left and right:
            triples = {(b,a,c) for b in before for c in after}
        elif left:
            triples = {(d,b,a) for (x,b),s in left.items() for z,_,d in incoming[x] if z<s}
        else:
            triples = {(a,c,d) for (y,c),s in right.items() for z,_,d in outgoing[y] if z>s}
        yield row, pairs, triples


def classify(path, preds, groups, counts, db, out, results):
    nodes = {n for n,v in preds.items() if v==(0,1)}
    anchors = defaultdict(lambda:defaultdict(set))
    with path.open() as f:
        for r in csv.DictReader(f):
            anchors[int(r['w'])][int(r['row'])].add(int(r['node_id']))
    assert nodes == {n for rows in anchors.values() for ns in rows.values() for n in ns}
    node_types = {n:{2:set(),3:set()} for n in nodes}
    eligible = set()
    @lru_cache(None)
    def support(seq):
        r = counts.execute('select train,test from counts where k=? and seq=?',(len(seq),pack(seq))).fetchone()
        assert r and int(r[1])>0
        return int(r[0])
    for w,rows in sorted(anchors.items()):
        for row,pairs,triples in centered(groups(w),rows):
            eligible.update(rows[row])
            for n in rows[row]:
                node_types[n][2].update(pairs)
                node_types[n][3].update(triples)
    all_types = set()
    node_summary = Counter()
    bins = Counter()
    with (out/'NODE_CLASS.csv').open('w',newline='') as f, (out/'NODE_TYPES.csv').open('w',newline='') as detail:
        wr,dw = csv.writer(f),csv.writer(detail)
        wr.writerow(('node_id','length','class','seen_types','unseen_types','min_seen_frequency','max_seen_frequency'))
        dw.writerow(('node_id','length','action_id_sequence','train_frequency'))
        for n in sorted(nodes):
            for k, seqs in node_types[n].items():
                for seq in sorted(seqs):
                    dw.writerow((n,k,' '.join(map(str,seq)),support(seq)))
                all_types.update(seqs)
                seen = [support(s) for s in seqs if support(s)>0]
                unseen = len(seqs)-len(seen)
                label = ('both' if unseen else 'seen_only') if seen else ('unseen_only' if unseen else ('no_path' if n in eligible else 'excluded'))
                wr.writerow((n,k,label,len(seen),unseen,min(seen,default=''),max(seen,default='')))
                node_summary[k,label]+=1
    for seq in all_types:
        n=support(seq)
        bins[len(seq),10*(n//10) if n else -1]+=1
    ids = {a for seq in all_types for a in seq}
    with (results/'ACTION_COUNTS.csv').open() as src,(out/'ACTIONS.csv').open('w',newline='') as dst:
        rd=csv.DictReader(src); wr=csv.DictWriter(dst,fieldnames=rd.fieldnames); wr.writeheader()
        for r in rd:
            if int(r['action_id']) in ids: wr.writerow(r)
    import json
    summary = {k:{'nodes':{label:node_summary[k,label] for label in ('seen_only','unseen_only','both','no_path','excluded')},
                  'type_frequency':dict(sorted((b,v) for (length,b),v in bins.items() if length==k))} for k in (2,3)}
    assert all(sum(row['nodes'].values())==len(nodes) for row in summary.values())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
