"""Recall one short native witness per selected pair and link family, and audit its ports."""
import csv
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import duckdb

src,res,audit,out=map(Path,sys.argv[1:])
db=duckdb.connect()
db.execute("set threads=3")
db.execute("set memory_limit='3GB'")
db.execute(f"attach '{src}/native_units.duckdb' as g (read_only)")
db.execute(f"attach '{src}/native_types.duckdb' as t (read_only)")
def rows(root,name):
    with (root/name).open() as f: return list(csv.DictReader(f))
rank={(int(r['hop']),r['behavior_type']):int(r['frequency']) for r in rows(res,'SELECTED_BEHAVIORS.csv')}
bound={(int(r['hop']),r['A'],r['B']):r for r in rows(audit,'NATIVE_CONNECTIVITY_AUDIT.csv')}
embedded={(int(r['hop']),r['A'],r['B']):r for r in rows(audit,'EMBEDDED_NATIVE_CONNECTIVITY_AUDIT.csv')}
assert all(int(v['native_connected_pairs'])==0 for v in bound.values())
assert all(int(v['native_connected_pairs'])==0 for v in embedded.values())
best={};counts=Counter()
with (res/'CONNECTION_EVIDENCE.jsonl').open() as f:
    for line in f:
        r=json.loads(line);key=(r['hop'],r['A'],r['B'])
        if r['A']==r['B']:continue
        family=r['definition'].split(':',1)[0]
        k=(*key,family)
        w=r['witness'];size=(len(w.get('path',[])),sum(map(len,w.get('event_bundles',[]))),r['structure'])
        counts[k]+=1
        if k not in best or size<best[k][0]:best[k]=(size,r)
def one(query,params):
    result=db.execute(query,params).fetchall()
    assert len(result)==1,(query,params,result)
    return result[0]
def port(witness_uid,other,donor_uid,edge_ids):
    if other>=0:
        parent,child,op,eid=one("select src_uid,dst_uid,op,eid from g.proc_edges where eid=?",[edge_ids[0]])
        assert op=='EVENT_FORK' and {parent,child}=={other,witness_uid}
        if parent==witness_uid:
            assert donor_uid==parent
            return dict(kind='fork_parent',eid=eid,donor_uid=donor_uid)
        sid1,sid2=one("select a.native_nid,b.native_nid from g.units a,g.units b where a.uid=? and b.uid=?",[child,donor_uid])
        assert sid1==sid2
        return dict(kind='fork_child',eid=eid,donor_uid=donor_uid,child_native_id=sid1)
    local=-1-other
    rid,du_resource,signature=one("""select n.native_nid,dn.nid,b.h from t.bundles b
        join g.nodes n on b.a=n.nid join g.units w on w.uid=b.uid
        join g.units d on d.uid=? and d.program=w.program
        join g.nodes dn on dn.window_id=d.segment and dn.native_nid=n.native_nid
        join t.bundles db on db.uid=d.uid and db.a=dn.nid and db.b is null and db.h=b.h
        where b.uid=? and b.a=? and b.b is null""",[donor_uid,witness_uid,local])
    events=db.execute('select eid,op,direction,ord from t.actor where uid=? and a=? order by ts,eid',[donor_uid,du_resource]).fetchall()
    assert events
    return dict(kind='resource',native_id=rid,bundle=signature,donor_uid=donor_uid,donor_events=events)
examples={}
with (out/'PAIR_CONNECTION_EVIDENCE.jsonl').open('w') as f:
    for k,(_,r) in sorted(best.items()):
        h,a,b,family=k;w=r['witness'];key=k[:3]
        r.update(f_A=rank[h,a],f_B=rank[h,b],matching_frequency=0,
            structure_count_in_family=counts[k],same_window_pairs=int(bound[key]['same_window_pairs']),
            connected_pairs=int(bound[key]['native_connected_pairs']),
            embedded_connected_pairs=int(embedded[key]['native_connected_pairs']))
        if family=='native_chain':
            p=w['path'];groups=w['event_bundles'];du=w['donor_uids']
            w['terminal_checks']=[port(p[0],p[1],du[0],groups[0]),port(p[-1],p[-2],du[1],groups[-1])]
        elif family=='shared_resource':
            uses=db.execute("""select a.eid,a.uid,n.native_nid,a.op,a.direction,a.ts,u.native_nid
                from t.actor a join g.nodes n on n.nid=a.a join g.units u using(uid) where eid in (?,?) order by eid""",
                [w['port_a_eid'],w['port_b_eid']]).fetchall()
            assert len(uses)==2 and len({x[2] for x in uses})==1 and len({x[6] for x in uses})==2
            assert uses[0][2]==w['anchor_native_id']
            w['native_access_records']=uses
        elif family=='fork':
            rec=one('select src_uid,dst_uid,ts,op from g.proc_edges where eid=?',[w['bridge_eid']])
            assert rec[-1]=='EVENT_FORK'
            w['native_fork']=rec
        else:raise ValueError(family)
        f.write(json.dumps(r,separators=(',',':'))+'\n')
        examples.setdefault((h,family),r)
assert len({k[:3] for k in best})==225
with (out/'COMPOSITION_EXAMPLES.jsonl').open('w') as f:
    for k,r in sorted(examples.items()):
        f.write(json.dumps(dict(r,category='zero',actual_joint=None),separators=(',',':'))+'\n')
with (out/'PAIR_EVIDENCE_AUDIT.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['hop','family','pairs_with_witness','structures'])
    for h,family in sorted(examples):
        keys=[k for k in counts if k[0]==h and k[-1]==family]
        wr.writerow((h,family,len(keys),sum(counts[k] for k in keys)))
print('audited representatives',len(best),'distinct pair-hops',len({k[:3] for k in best}),flush=True)
