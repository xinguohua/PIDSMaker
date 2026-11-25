"""Independent per-hop ranking, exact embedded behaviours, witnessed full joints."""
import argparse
import csv
import hashlib
import itertools as it
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import duckdb
import networkx as nx
from connectable import SOURCE, clique, output, setup
from connectable_chains import WitnessSearch
from types_native import exact_graph, node_meta

HOPS=(1,3,5)

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))

class Matcher:
    def __init__(self,db):
        self.db=db
        self.units={u:(p,s,t) for u,p,s,t in db.execute('select uid,program,subject,unit_type from g.units join t.ut using(uid)').fetchall()}
        self.windows=dict(db.execute('select uid,segment from g.units').fetchall())
        self.meta={n:(k,r or 'unknown',p or '') for n,k,r,p in db.execute('select n.nid,n.kind,n.role,u.program from g.nodes n left join g.units u on n.nid=u.subject').fetchall()}
        self.bb=defaultdict(list)
        for u,a,b,h in db.execute('select * from t.bundles').fetchall():self.bb[u].append((a,b,h))
        self.window_units=defaultdict(set)
        self.relations=defaultdict(list)
        self.native=defaultdict(lambda:defaultdict(set))
        self.edges=defaultdict(list)
        self.event_owner=dict(db.execute('select eid,uid from t.actor').fetchall())
        for u in self.units:self.window_units[self.windows[u]].add(u)
        # Necessary matching graph: all shared endpoints and original process links.
        for obj,group in it.groupby(db.execute('select obj,uid from t.endpoint_use order by obj,uid').fetchall(),lambda r:r[0]):
            us=[u for _,u in group]
            if us:self.relations[self.windows[us[0]]].append(us)
        for a,b in db.execute('select src_uid,dst_uid from g.proc_edges').fetchall():self.relations[self.windows[a]].append([a,b])
        for w,u,v,e in db.execute('''select r.segment,r.uid,-1-r.resource,r.eid from g.resource_events r
            union all select w,src_uid,dst_uid,eid from forks''').fetchall():
            if u==v:continue
            self.native[w][u].add(v);self.native[w][v].add(u)
            self.edges[w,min(u,v),max(u,v)].append(e)
        self.cert_cache={}

    def graph(self,w,us):
        us=set(us);g=nx.Graph()
        for u in us:g.add_node(u,tag=self.units[u][2])
        for group in self.relations[w]:
            g.add_edges_from(it.combinations(us.intersection(group),2))
        return g

    def certificate(self,us):
        us=tuple(sorted(us))
        if us in self.cert_cache:return self.cert_cache[us]
        d=self.db
        d.execute('create or replace temp table match_members as select unnest(?::bigint[]) uid',[list(us)])
        timelines=d.execute('''with shared as (select obj from t.endpoint_use semi join match_members using(uid)
             group by obj having count(distinct uid)>1), ranked as (
             select uid,obj,ord,dense_rank() over(partition by obj order by ts nulls last) rr
             from t.steps semi join match_members using(uid) semi join shared using(obj)), chunks as (
             select uid,obj,ord//128 chunk,sha256(string_agg(concat(ord,':',rr),',' order by ord,rr)) h
             from ranked group by uid,obj,chunk)
             select uid,obj,sha256(string_agg(h,'' order by chunk)) h from chunks group by uid,obj''').fetchall()
        links=d.execute('''select eid,src,dst,op,owner,ord,a,b from t.process
             where src in (select uid from match_members) and dst in (select uid from match_members)''').fetchall()
        cert=exact_graph(us,self.bb,timelines,links,self.meta,self.units)
        self.cert_cache[us]=cert
        return cert

    def occurrences(self,representative):
        rep=tuple(sorted(representative));w=self.windows[rep[0]]
        if len(rep)==1:
            tag=self.units[rep[0]][2]
            # S- types have no internal process self-loop; don't silently add one.
            loops={u for u, in self.db.execute('select distinct src from t.process where src=dst').fetchall()}
            return [(u,) for u in self.units if self.units[u][2]==tag and u not in loops]
        pattern=self.graph(w,rep)
        target=self.certificate(rep);wanted=Counter(self.units[u][2] for u in rep)
        found=set();tested=set()
        for ww,window_us in self.window_units.items():
            counts=Counter(self.units[u][2] for u in window_us)
            if any(counts[t]<n for t,n in wanted.items()):continue
            candidates=tuple(sorted(u for u in window_us if self.units[u][2] in wanted))
            # A single possible member set needs one certificate, not factorial automorphisms.
            if len(candidates)==len(rep):
                if self.certificate(candidates)==target:found.add(candidates)
                print('embedded unique subset',ww,len(rep),flush=True)
                continue
            h=self.graph(ww,candidates)
            gm=nx.algorithms.isomorphism.GraphMatcher(h,pattern,node_match=lambda a,b:a['tag']==b['tag'])
            for mapping in gm.subgraph_monomorphisms_iter():
                us=tuple(sorted(mapping))
                if us in tested:continue
                tested.add(us)
                if self.certificate(us)==target:found.add(us)
            print('embedded window',ww,'members',len(rep),'tested',len(tested),'found',len(found),flush=True)
        assert rep in found
        return sorted(found)

    def joints(self,aa,bb):
        """Each count binds two complete disjoint behaviours plus real connector events.

        Enumerate all shortest paths for each concrete terminal interface after
        excluding all other A/B subjects; extra-event sets identify realizations.
        """
        bywindow=defaultdict(list)
        for b in bb:bywindow[self.windows[b[0]]].append(b)
        for a in aa:
            w=self.windows[a[0]];graph=self.native[w]
            for b in bywindow[w]:
                if set(a)&set(b):continue
                members=set(a)|set(b);seen=set()
                for u in a:
                    for v in b:
                        paths=[]
                        if v in graph[u]:paths.append((u,v))
                        for x in graph[u]-members:
                            for y in graph[v]-members:
                                paths.extend((u,)+p+(v,) for p in WitnessSearch.middle(graph,x,y,members,True))
                        for path in paths:
                            bundles=[sorted(self.edges[w,min(x,y),max(x,y)]) for x,y in zip(path,path[1:])]
                            assert all(bundles)
                            ev=sorted({e for group in bundles for e in group})
                            extra=tuple(e for e in ev if self.event_owner[e] not in members)
                            if extra in seen:continue
                            seen.add(extra)
                            yield dict(window=w,A_uids=a,B_uids=b,path=path,event_bundles=bundles,extra_events=extra)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(SOURCE));p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(os.environ.get('RUN_OUTPUT_DIR','.')));args=p.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((args.source/'NATIVE_INPUT_AUDIT.json').read_text())
    assert (audit['windows'],audit['edge_occurrences'],audit['native_loader_train_batches'])==(581,3426589,3665)
    for r in csv.DictReader((args.source/'INTEGRITY_AUDIT.csv').open()):
        assert all(int(r[k])==0 for k in ('missing_nodes','missing_events','duplicate_event_owners','duplicate_unit_owners','event_ids_or_field_mismatches'))
    for name in ('NATIVE_INPUT_AUDIT.json','INTEGRITY_AUDIT.csv'):shutil.copyfile(args.source/name,out/name)
    dump(out/'RUN_CONFIG.json',dict(source=str(args.source),screen=str(args.screen),run_id=os.getenv('RUN_ID'),hops=HOPS,N=100,n=10,source_run=audit['train_run'],source_commit=audit['train_commit'],selection='per-hop lexicographically earliest frequency-ranked clique; no joint counts',count='unique (window, A members, B members, extra connector event set)',partition='forward'))
    db=duckdb.connect();setup(db,args.source,100)
    ranks=db.execute('select hop,rk,tag,f from ranks where hop in (1,3,5) order by hop,rk').fetchall()
    output(out/'ALL_BEHAVIOR_FREQUENCIES.csv','hop rank behavior_type partition_frequency',ranks)
    output(out/'TOP_N_BEHAVIORS.csv','hop rank behavior_type partition_frequency',[r for r in ranks if r[1]<=100])
    screen=list(csv.DictReader(args.screen.open()));chosen={};evidence={}
    for h in HOPS:
        g=defaultdict(set)
        for r in screen:
            if int(r['hop'])==h and r['supported']=='1':
                a,b=r['A'],r['B'];g[a].add(b);g[b].add(a);evidence[h,a,b]=json.loads(r['witness'])
        order=[t for hh,r,t,f in ranks if hh==h and r<=100]
        chosen[h]=next(c for n in range(10,0,-1) if (c:=clique(order,g,n)) is not None)
    selected=[(h,i,r,t,f) for h,ts in chosen.items() for i,t in enumerate(ts,1) for hh,r,tt,f in ranks if hh==h and tt==t]
    output(out/'SELECTED_BEHAVIORS.csv','hop position rank behavior_type partition_frequency',selected)
    print('selected',[(h,[r for hh,i,r,t,f in selected if hh==h]) for h in HOPS],flush=True)
    m=Matcher(db)
    # Real-data positive controls use a known native fork and a known shared object.
    controls=[]
    queries=[('fork','select src_uid,dst_uid from forks order by eid limit 1'),('resource','select a.uid,b.uid from g.incidence a join g.incidence b on a.resource=b.resource and a.uid<b.uid order by a.uid,b.uid limit 1')]
    for kind,q in queries:
        u,v=db.execute(q).fetchone();aa=m.occurrences([u]);bb=m.occurrences([v])
        hits=list(m.joints([tuple([u])],[tuple([v])]))
        assert (u,) in aa and (v,) in bb and hits
        controls.append(dict(kind=kind,known_uids=[u,v],actual_frequency=len(hits),matches=hits))
    dump(out/'POSITIVE_CONTROL.json',dict(passed=True,controls=controls))
    print('positive controls passed',[(c['kind'],c['actual_frequency']) for c in controls],flush=True)
    occurrences={};reps={};ownfreq={}
    for h,ts in chosen.items():
        for t in ts:
            if t not in occurrences:
                cid=db.execute('select min(cid) from inst where hop=? and tag=?',[h,t]).fetchone()[0]
                rep=[u for u, in db.execute('select uid from assignment where hop=? and cid=? order by uid',[h,cid]).fetchall()]
                reps[t]=rep;occurrences[t]=m.occurrences(rep)
                print('behaviour',h,t,'size',len(rep),'full frequency',len(occurrences[t]),flush=True)
            ownfreq[h,t]=len(occurrences[t])
            # Every original placement must be included, regardless of larger partitions.
            actual=set(occurrences[t]);original=defaultdict(list)
            for c,u in db.execute('select cid,uid from assignment join inst using(hop,cid) where hop=? and tag=?',[h,t]).fetchall():original[c].append(u)
            assert all(tuple(sorted(us)) in actual for us in original.values())
    with (out/'BEHAVIOR_MATCHES.jsonl').open('w') as f:
        for t,instances in occurrences.items():
            for us in instances:f.write(json.dumps(dict(type=t,window=m.windows[us[0]],uids=us))+'\n')
    results=[];event_ids=set();counts={}
    with (out/'CONNECTION_EVIDENCE.jsonl').open('w') as f:
        for h,ts in chosen.items():
            for a,b in it.combinations(sorted(ts),2):
                wit=evidence[h,a,b]
                # Keep exact donor nodes and actual edge bundles, not program reachability.
                if wit['kind']=='chain':
                    path=wit['path'];w=wit['window']
                    wit['event_bundles']=[sorted(m.edges[w,min(x,y),max(x,y)]) for x,y in zip(path,path[1:])]
                    assert all(wit['event_bundles'])
                    for es in wit['event_bundles']:event_ids.update(es)
                elif wit['kind']=='resource':event_ids.update(wit['port_eids'])
                else:event_ids.add(wit['eid'])
                f.write(json.dumps(dict(hop=h,A=a,B=b,witness=wit))+'\n')
    with (out/'JOINT_MATCHES.jsonl').open('w') as f:
        for h,ts in chosen.items():
            mat={}
            for a,b in it.combinations(sorted(ts),2):
                key=(a,b)
                if key not in counts:
                    hits=list(m.joints(occurrences[a],occurrences[b]));counts[key]=hits
                hits=counts[key];mat[a,b]=len(hits)
                for hit in hits:
                    for es in hit['event_bundles']:event_ids.update(es)
                    f.write(json.dumps(dict(hop=h,A=a,B=b,**hit))+'\n')
                results.append((h,a,b,len(hits),'complete'))
                print('pair',h,ts.index(a)+1,ts.index(b)+1,len(hits),flush=True)
            output(out/f'hop_{h}_matrix.csv','behavior '+' '.join(ts),[[a]+[ownfreq[h,a] if a==b else mat[tuple(sorted((a,b)))] for b in ts] for a in ts])
    output(out/'PAIR_COUNTS.csv','hop A B joint_frequency status',results)
    output(out/'BEHAVIOR_COUNTS.csv','hop position rank behavior_type partition_frequency full_graph_frequency',[(h,i,r,t,f,ownfreq[h,t]) for h,i,r,t,f in selected])
    for control in controls:
        for hit in control['matches']:
            for es in hit['event_bundles']:event_ids.update(es)
    db.execute('create temp table evidence_eids as select unnest(?::bigint[]) eid',[sorted(event_ids)])
    db.execute(f"copy (select e.* from g.events e semi join evidence_eids using(eid) order by eid) to '{out}/NATIVE_EVENT_EVIDENCE.csv' (header,delimiter ',')")
    dump(out/'summary.json',dict(status='complete',positive_controls_passed=True,hops=[dict(hop=h,selected_n=len(chosen[h]),zero_pairs=sum(n==0 for hh,a,b,n,s in results if hh==h),positive_pairs=sum(n>0 for hh,a,b,n,s in results if hh==h),joint_matches=sum(n for hh,a,b,n,s in results if hh==h)) for h in HOPS]))

if __name__=='__main__':main()
