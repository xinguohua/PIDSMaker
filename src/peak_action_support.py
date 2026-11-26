"""Compare peak edge Action signatures to the frozen normal training split."""
import csv
from collections import Counter, defaultdict
from action_paths import pack


def summarize(path,preds,attrs,keys,counts,db,attack,out):
    fps={n for n,v in preds.items() if v==(0,1)}
    peaks=defaultdict(set)
    with path.open() as stream:
        for r in csv.DictReader(stream):
            edge=(int(r['w']),int(r['row']))
            n=int(r['node_id'])
            assert n in fps and n not in peaks[edge]
            peaks[edge].add(n)
    assert set.union(*peaks.values())==fps
    edge_type={}
    excluded=set()
    for (w,row),ns in peaks.items():
        u,op,v,label,split=db.execute('select u,op,v,y,split from edges join windows using(w) where w=? and row=?',(w,row)).fetchone()
        assert split=='test' and ns<={u,v}
        edge_type[w,row]=attrs[u][0],op,attrs[v][0]
        if label or u in attack or v in attack or u not in preds or v not in preds:
            excluded.add((w,row))

    # Directly count the requested complete signatures in the normal training edges;
    # this also handles a peak filtered from the normal test path export.
    training=Counter(dict.fromkeys(edge_type.values(),0))
    for u,op,v in db.execute("select u,op,v from windows join edges using(w) where split='train' and y=0"):
        key=(attrs[u][0],op,attrs[v][0])
        if key in training:
            training[key]+=1
    for key,n in training.items():
        aid=keys.get(key)
        stored=counts.execute('select train from counts where k=1 and seq=?',(pack((aid,)),)).fetchone() if aid is not None else None
        assert (int(stored[0]) if stored else 0)==n
    masks=defaultdict(int)
    units=defaultdict(Counter)
    by_op=defaultdict(Counter)
    for key,n in training.items():
        units['distinct_action_types']['seen' if n else 'unseen']+=1
    for edge,ns in peaks.items():
        key=edge_type[edge]
        seen=bool(training[key])
        label='seen' if seen else 'unseen'
        units['distinct_peak_edges'][label]+=1
        units['node_peak_pairs'][label]+=len(ns)
        if edge in excluded:
            units['peak_edges_filtered_from_path_analysis'][label]+=1
        by_op[key[1]][label]+=1
        for n in ns:
            masks[n] |= 1 if seen else 2
    assert set(masks)==fps
    for mask in masks.values():
        units['fp_nodes'][{1:'seen',2:'unseen',3:'mixed'}[mask]]+=1
    with (out/'PEAK_ACTION_SUPPORT.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['unit','seen_only','unseen_only','mixed','total'])
        for unit,c in units.items():
            wr.writerow([unit,c['seen'],c['unseen'],c['mixed'],sum(c.values())])
    with (out/'PEAK_ACTION_OPERATIONS.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['operation','seen_peak_edges','unseen_peak_edges'])
        for op,c in sorted(by_op.items()):wr.writerow([op,c['seen'],c['unseen']])
