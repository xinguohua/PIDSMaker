"""Match verified peak edges to actual length-two paths in the frozen native windows."""
import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from action_composition import ro
from action_index import signature
from action_paths import pack
from action_window import read_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('peak_file',type=Path)
    parser.add_argument('compositions',type=Path)
    args=parser.parse_args()
    result=args.compositions/'results'
    root=args.compositions.parent
    db=ro(root/'native.sqlite')
    db.execute('attach database ? as sem',(f'file:{root}/semantic.sqlite?mode=ro&immutable=1',))
    counts=sqlite3.connect(f'file:{args.compositions}/counts.sqlite?mode=ro',uri=True)
    with args.peak_file.open() as f:
        peaks=defaultdict(dict)
        for r in csv.DictReader(f):
            w,row,n=map(int,(r['w'],r['row'],r['node_id']))
            peaks[w][row]=n
    target={n for p in peaks.values() for n in p.values()}
    assert len(target)==555
    attrs={n:(signature(raw),) for n,raw in db.execute('select id,semantic from sem.attributes')}
    labels=dict(db.execute('select id,label from nodes'))
    keys={}
    csv.field_size_limit(sys.maxsize)
    with (result/'ACTION_COUNTS.csv').open() as f:
        for r in csv.DictReader(f):
            keys[r['source_semantics'],r['operation'],r['destination_semantics']]=int(r['action_id'])
    cell=Path(json.loads((result/'INPUT_AUDIT.json').read_text())['cell'])
    pred=dict(db.execute('select id,gt from pred'))
    attack={n for n,gt in pred.items() if gt}
    mapping={u:n for n,u in db.execute('select id,uuid from nodes')}
    for rel in json.loads((cell/'manifest.json').read_text())['protocol']['ground_truth_relative_path']:
        for line in (cell.parents[1]/'code/Ground_Truth/orthrus'/rel).read_text().splitlines():
            attack.update(mapping[u] for u in line.replace('"','').strip().split(',') if u in mapping)
    output=Path(os.environ['RUN_OUTPUT_DIR'])
    members=defaultdict(set)
    witnesses={}
    support={}
    included=set()
    for w,peak in sorted(peaks.items()):
        groups,_,_,_=read_window(db,w,True,attrs,keys,pred,attack)
        incoming=defaultdict(list)
        outgoing=defaultdict(list)
        anchors=[]
        for t,batch in enumerate(groups):
            for e in batch:
                incoming[e[2]].append((t,e))
                outgoing[e[1]].append((t,e))
                if e[0] in peak:
                    assert e[2]==peak[e[0]]
                    anchors.append((t,e))
                    included.add((w,e[0]))
        for t,e in anchors:
            n=e[2]
            def record(left,right,pos):
                seq=(left[3],right[3])
                if seq not in support:
                    row=counts.execute('select train,test from counts where k=2 and seq=?',(pack(seq),)).fetchone()
                    support[seq]=tuple(map(int,row))
                assert support[seq][0]>0
                key=(*seq,pos)
                members[key].add(n)
                if key not in witnesses:
                    ids=[left[1],left[2],right[2]]
                    ops=[db.execute('select op from edges where w=? and row=?',(w,r[0])).fetchone()[0] for r in (left,right)]
                    witnesses[key]=dict(w=w,first_row=left[0],second_row=right[0],node_id=n,
                        source=labels[ids[0]],op1=ops[0],middle=labels[ids[1]],op2=ops[1],destination=labels[ids[2]])
            for earlier,left in incoming[e[1]]:
                if earlier<t: record(left,e,'second')
            for later,right in outgoing[e[2]]:
                if later>t: record(e,right,'first')
    participating=set().union(*members.values())
    audit=dict(target_nodes=len(target),peak_edges=sum(map(len,peaks.values())),eligible_peak_edges=len(included),
        nodes_with_peak_on_two_step_path=len(participating),nodes_without_such_path=sorted(target-participating))
    (output/'SH_PEAK_AUDIT.json').write_text(json.dumps(audit,indent=2))
    rows=[]
    for key,ns in sorted(members.items(),key=lambda item:-len(item[1])):
        a,b,pos=key
        train,test=support[a,b]
        rows.append(dict(action1=a,action2=b,peak_position=pos,nodes=len(ns),nodes_ids='|'.join(map(str,sorted(ns))),
            train_instances=train,test_instances=test,**witnesses[key]))
    if rows:
        with (output/'SH_PEAK_COMPOSITIONS.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(audit,flush=True)


if __name__=='__main__':
    main()
