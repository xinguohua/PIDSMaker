"""Classify exact-length native paths containing a false-positive node in its peak windows."""
import csv
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from functools import lru_cache

from action_paths import pack


def index_window(groups):
    incoming, outgoing = defaultdict(dict), defaultdict(dict)
    for t,batch in enumerate(groups):
        for _,u,v,a in batch:
            times=outgoing[u].get((v,a))
            if times is None:
                times=[]
                outgoing[u][v,a]=incoming[v][u,a]=times
            if not times or times[-1]!=t:
                times.append(t)
    return incoming,outgoing


def types_touching(n,k,incoming,outgoing):
    # Every path touching n contains at least one incident edge variant.
    anchors={(u,n,a):ts for (u,a),ts in incoming.get(n,{}).items()}
    anchors.update({(n,v,a):ts for (v,a),ts in outgoing.get(n,{}).items()})
    for (u,v,a),ts in anchors.items():
        prior=incoming.get(u,{})
        later=outgoing.get(v,{})
        for (x,b),bs in prior.items():
            j=bisect_left(bs,ts[-1])
            if not j:
                continue
            if k==2:
                yield (b,a)
            else:
                for (_,c),cs in incoming.get(x,{}).items():
                    if cs[0]<bs[j-1]:
                        yield (c,b,a)
                # Anchor in the middle: earliest feasible anchor after first b.
                mid=ts[bisect_right(ts,bs[0])]
                for (_,c),cs in later.items():
                    if mid<cs[-1]:
                        yield (b,a,c)
        for (y,b),bs in later.items():
            j=bisect_right(bs,ts[0])
            if j==len(bs):
                continue
            if k==2:
                yield (a,b)
            else:
                for (_,c),cs in outgoing.get(y,{}).items():
                    if bs[j]<cs[-1]:
                        yield (a,b,c)


def classify(path,preds,groups,counts,db,out):
    targets={n for n,v in preds.items() if v==(0,1)}
    peaks=defaultdict(lambda:defaultdict(set))
    with path.open() as f:
        for r in csv.DictReader(f):
            n,w,row=map(int,(r['node_id'],r['w'],r['row']))
            assert n in targets and n not in peaks[w][row]
            peaks[w][row].add(n)
    assert set.union(*(ns for p in peaks.values() for ns in p.values()))==targets

    @lru_cache(maxsize=100000)
    def support(seq):
        found=counts.execute('select train,test from counts where k=? and seq=?',(len(seq),pack(seq))).fetchone()
        assert found and int(found[1])>0
        return 1 if int(found[0]) else 2

    masks={k:defaultdict(int) for k in (2,3)}
    for w,anchors in peaks.items():
        rows={row:{u,v} for row,u,v in db.execute('select row,u,v from edges where w=?',(w,)) if row in anchors}
        assert rows.keys()==anchors.keys() and all(ns<=rows[row] for row,ns in anchors.items())
        idx=index_window(groups(w))
        for n in set.union(*anchors.values()):
            for k,bynode in masks.items():
                if bynode[n]!=3:
                    for seq in types_touching(n,k,*idx):
                        bynode[n]|=support(seq)
                        if bynode[n]==3:
                            break
    assert masks[2].keys()==masks[3].keys()==targets
    assert all(not masks[3][n] or masks[2][n] for n in targets)
    with (out/'PEAK_WINDOW_NODE_COMPOSITIONS.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('length','group','count','all_fp','percent_all_fp','fp_with_path','percent_with_path'))
        for k,by_node in masks.items():
            hist=Counter(by_node.values())
            covered=len(targets)-hist[0]
            for i,label in ((1,'seen_only'),(2,'unseen_only'),(3,'both'),(0,'no_path_of_length')):
                writer.writerow((k,label,hist[i],len(targets),100*hist[i]/len(targets),covered,
                                 100*hist[i]/covered if i and covered else ''))
