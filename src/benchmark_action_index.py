"""Finite synthetic benchmark; each measurement has a 45-second subprocess limit."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
from action_index import ActionIndex, LENGTHS, Participation

BASELINE = 'db672ff9d985cc698eb66cb945bd03bbdba9eba4'
CASES = [(50,20,1),(20,4,4),(8,8,8)]


def worker(case, method):
    groups_n,copies,actions_n=CASES[case]
    groups=[[(g*copies+i,0,0,i%actions_n) for i in range(copies)] for g in range(groups_n)]
    endpoints={e[0]:(2*e[0],2*e[0]+1) for batch in groups for e in batch}
    if method=='reference':
        source=subprocess.check_output(['git','show',BASELINE+':src/action_paths.py'],text=True,
                                       cwd=Path(__file__).resolve().parents[1])
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='paths')
        scope=dict(defaultdict=defaultdict,LENGTHS=LENGTHS)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-content-reference>','exec'),scope)
    tracemalloc.start()
    start=time.perf_counter()
    digest=types=instances=0
    phases={}
    def consume(k,seq,n):
        nonlocal digest,types,instances
        digest=(digest+int.from_bytes(hashlib.sha256(repr((k,seq,n)).encode()).digest(),'big'))%(1<<256)
        types+=1;instances+=n
    if method=='reference':
        totals=scope['paths']([e for batch in groups for e in batch],True,groups,endpoints)
        for k, table in totals.items():
            for seq,(n,_,_) in table.items():consume(k,seq,n)
        nodes={k:set().union(*(r[2] for r in table.values())) if table else set() for k,table in totals.items()}
    else:
        index=ActionIndex(groups)
        phases['index_s']=time.perf_counter()-start
        nodes={}
        for k in LENGTHS:
            tick=time.perf_counter();participation=Participation()
            for seq,n in index.sequences(k):
                consume(k,seq,n)
                participation.add(index,seq,1)
            nodes[k]=set(participation.flags(index,endpoints))
            phases['length_'+str(k)+'_s']=time.perf_counter()-tick
        phases.update(prefix_visits=index.branches,operations=index.operations)
    seconds=time.perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(method=method,seconds=seconds,traced_peak_bytes=peak,peak_rss_bytes=rss,
                digest=hex(digest),types=types,instances=str(instances),
                nodes={k:sorted(ns) for k,ns in nodes.items()},phases=phases)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['reference','stream']);p.add_argument('--case',type=int);a=p.parse_args()
    if a.worker:
        print(json.dumps(worker(a.case,a.worker)));return
    report=dict(baseline=BASELINE,scope='synthetic; identical content joins and actual endpoint membership; not window 437',
                timing='tracemalloc enabled for both; isolated processes; 45 second limit each',cases=[])
    for i,(gs,copies,actions) in enumerate(CASES):
        results=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker',m,'--case',str(i)],timeout=45))
                 for m in ('reference','stream')]
        for field in ('digest','types','instances','nodes'):assert results[0][field]==results[1][field],field
        for r in results:r['node_counts']={k:len(v) for k,v in r.pop('nodes').items()}
        report['cases'].append(dict(events=gs*copies,groups=gs,actions=actions,equal=True,results=results))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
