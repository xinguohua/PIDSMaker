"""Join full partition membership certificates to witnessed native connectors."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

from connectable import output


def dumped(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'))


def run(db,args,chosen,direct,possible,graphs,subjects,anchors,ranks):
    out=args.out
    db.execute('create temp table chosen(hop int,tag varchar)')
    db.executemany('insert into chosen values (?,?)',[(h,t) for h,ts in chosen.items() for t in ts])
    db.execute("""create temp table p as select distinct hop,tag,cid,uid,w,sid,role,eid
        from sp semi join chosen using(hop,tag)""")
    db.execute("""create temp table r as select p.*,a.ord from ports p
        semi join chosen using(hop,tag) semi join shared using(rid) join t.actor a using(eid)""")
    # Only a concrete membership and a separately witnessed chain are needed for nomination.
    owners={(h,u):c for h,u,c in db.execute('select hop,uid,cid from m semi join chosen using(hop,tag)').fetchall()}
    donor=defaultdict(dict)
    actual_all=defaultdict(set)
    donor_alt=defaultdict(dict)
    for h,t,c,u,w,s,role,eid in db.execute('select * from p order by cid,uid,role,eid').fetchall():
        donor[s].setdefault((h,t,role),(c,u,w,eid))
        actual_all[w,s,h,t,role].add(c)
        slots=donor_alt[s,h,t,role]
        if len(slots)<2:slots.setdefault(c,(c,u,w,eid))
    schemas={}
    instances=defaultdict(set)
    orient_instances=defaultdict(set)
    def record(h,a,b,shape,witness,joint=None,orientation='unoriented'):
        key=(h,a,b,shape,orientation)
        if key not in schemas:schemas[key]=[witness,set(),None]
        rec=schemas[key]
        if joint is not None:
            rec[1].add(joint)
            if rec[2] is None:rec[2]=witness
            instances[h,a,b].add(joint)
            orient_instances[h,a,b,orientation].add(joint)

    # Each query produces per-structure donor evidence, then scans all same-window matches.
    common="a.hop=b.hop and (a.tag<b.tag or (a.tag=b.tag and a.cid<b.cid)) and a.cid<>b.cid"
    sqls={
      'shared_subject':f"""select a.*,b.tag bt,b.cid bc,b.uid bu,b.w bw,b.eid be,
        to_json([a.role,b.role]) layout,'unoriented' ori,-1::bigint extra
        from p a join p b on {common} and a.sid=b.sid""",
      'shared_resource':f"""select a.hop,a.tag,a.cid,a.uid,a.w,a.rid sid,a.op AS role,a.eid,
        b.tag bt,b.cid bc,b.uid bu,b.w bw,b.eid be,
        to_json([a.kind,a.unit_type,a.op,a.direction,a.ord::varchar,
                 b.unit_type,b.op,b.direction,b.ord::varchar]) layout,
        case when a.ts<b.ts then 'A_to_B' when a.ts>b.ts then 'B_to_A' else 'simultaneous' end ori,
        -1::bigint extra from r a join r b on {common} and a.rid=b.rid and a.sid<>b.sid""",
      'fork':f"""select a.hop,a.tag,a.cid,a.uid,a.w,a.sid,a.role,a.eid,
        b.tag bt,b.cid bc,b.uid bu,b.w bw,b.eid be,
        to_json([a.role,f.op,b.role]) layout,
        case when f.pa=a.sid then 'A_to_B' else 'B_to_A' end ori,f.eid extra
        from p a join forks f on f.pa=a.sid or f.ch=a.sid join p b on {common}
        and b.sid=case when f.pa=a.sid then f.ch else f.pa end"""
    }
    for family,sql in sqls.items():
        print('matching',family,flush=True)
        db.execute('create or replace temp view links as '+sql)
        ev="""struct_pack(cid_a:=cid,uid_a:=uid,window_a:=w,port_a_eid:=eid,
            cid_b:=bc,uid_b:=bu,window_b:=bw,port_b_eid:=be,anchor_native_id:=sid,
            bridge_eid:=extra)"""
        for h,a,b,layout,ori,wit in db.execute(f"""select hop,tag,bt,layout,ori,first({ev} order by
            w,cid,bc,uid,bu,eid,be,extra) from links group by all""").fetchall():
            record(h,a,b,family+':'+layout,wit,orientation=ori)
        cond='and (l.extra<0 or l.extra>>32=l.w)'
        for h,a,b,layout,ori,w,c,d,ex,wit in db.execute(f"""select hop,tag,bt,layout,ori,w,cid,bc,
            case when f.eid is not null and coalesce(z.cid,-1) not in (l.cid,l.bc)
            then l.extra else -1 end ext,first({ev}) from links l left join forks f on f.eid=l.extra
            left join assignment z on z.hop=l.hop and z.uid=f.src_uid
            where l.w=l.bw {cond} group by l.hop,l.tag,l.bt,l.layout,l.ori,l.w,l.cid,l.bc,ext""".replace(
                'select hop,tag,bt,layout,ori,w,cid,bc','select l.hop,tag,bt,layout,ori,l.w,l.cid,bc').replace(
                'cid_a:=cid','cid_a:=l.cid').replace('uid_a:=uid','uid_a:=l.uid').replace('window_a:=w','window_a:=l.w').replace('port_a_eid:=eid','port_a_eid:=l.eid')).fetchall():
            record(h,a,b,family+':'+layout,wit,(w,min(c,d),max(c,d),() if ex<0 else (ex,)),ori)

    # Shortest native chains between concrete subject anchors. One resource crossing
    # consists of two real accesses. Intermediate subjects/resources cannot repeat.
    edge_table={}
    candidate_sids=list(donor)
    candidate_nodes={u for u,s in subjects.items() if s in donor}
    plans=defaultdict(list)
    for w,(graph,ids,ix) in graphs.items():
        endpoints=[(ix[u],u) for u in candidate_nodes.intersection(ix)]
        if len(endpoints)<2:continue
        endpoints.sort()
        indices=[j for j,_ in endpoints]
        dist=graph.distances(source=indices,target=indices)
        for i,(_,u) in enumerate(endpoints):
            for j in range(i+1,len(endpoints)):
                if dist[i][j]==float('inf'):continue
                v=endpoints[j][1]
                plans[w].append((u,v))
    print('anchor pairs',sum(map(len,plans.values())),flush=True)
    # Preload only edges of retained shortest paths, one source window at a time.
    for w,pairs in plans.items():
        graph,ids,ix=graphs[w]
        needed={}
        lookup={}
        for u,v in pairs:
            options=[]
            for (h,a,ra),da in donor[subjects[u]].items():
                for (hh,b,rb),dbb in donor[subjects[v]].items():
                    if h!=hh:continue
                    # For identical types, require distinct donor groups somewhere.
                    if a==b and da[0]==dbb[0]:
                        alt=next((d for d in donor_alt[subjects[v],h,b,rb].values() if d[0]!=da[0]),None)
                        if alt is None:
                            alt=next((d for d in donor_alt[subjects[u],h,a,ra].values() if d[0]!=dbb[0]),None)
                            if alt is None:continue
                            options.append((h,a,b,ra,rb,alt,dbb));continue
                        options.append((h,a,b,ra,rb,da,alt));continue
                    options.append((h,a,b,ra,rb,da,dbb))
            if not options:continue
            paths={tuple(ids[i] for i in path) for path in graph.get_all_shortest_paths(ix[u],to=ix[v])}
            lookup[u,v]=(options,paths)
            for path in paths:
                for x,y in zip(path,path[1:]):needed[tuple(sorted((x,y)))]=[]
        if not needed:continue
        # The indexed window and native event IDs identify every original tensor row.
        lo=w<<32; hi=(w+1)<<32
        sql="""select r.uid,-1-r.resource,e.eid,e.ts,e.op,e.direction,e.uid,
            n.kind,n.native_nid from g.resource_events r join t.actor e using(eid)
            join g.nodes n on n.nid=r.resource where r.segment=?
            union all select f.src_uid,f.dst_uid,f.eid,f.ts,f.op,'actor_to_object',f.src_uid,
            'subject',f.ch from forks f where f.w=?"""
        for x,y,e,ts,op,d,owner,kind,nid in db.execute(sql,[w,w]).fetchall():
            k=tuple(sorted((x,y)))
            if k in needed:needed[k].append((e,ts,op,d,owner,kind,nid))
        programs=dict(db.execute('select uid,program from g.units where segment=?',[w]).fetchall())
        for (u,v),(options,paths) in lookup.items():
            for path in paths:
                bundles=[sorted(needed[tuple(sorted(edge))]) for edge in zip(path,path[1:])]
                assert all(bundles)
                def describe(reverse):
                    seq=path[::-1] if reverse else path
                    bb=bundles[::-1] if reverse else bundles
                    times=sorted({ev[1] for grp in bb for ev in grp})
                    tick={ts:i for i,ts in enumerate(times)}
                    nodes=[programs[x] if x>=0 else bb[i-1][0][5] for i,x in enumerate(seq)]
                    links=[sorted((op,d,owner==seq[i],tick[ts]) for e,ts,op,d,owner,kind,nid in grp)
                           for i,grp in enumerate(bb)]
                    left=min(ev[1] for ev in bb[0]);right=min(ev[1] for ev in bb[-1])
                    direction='A_to_B' if left<right else 'B_to_A' if left>right else 'simultaneous'
                    return [nodes,links],direction
                layouts={rev:describe(rev) for rev in (False,True)}
                for h,a,b,ra,rb,da,dbb in options:
                    rev=a>b or (a==b and u>v)
                    aa,bb=(b,a) if rev else (a,b)
                    shape,ori=layouts[rev]
                    roles=[rb,ra] if rev else [ra,rb]
                    # Direct native fork is already covered by the fork query.
                    if len(path)==2:continue
                    layout='native_chain:'+dumped([roles,shape])
                    d1,d2=(dbb,da) if rev else (da,dbb)
                    wits={'window':w,'path':path[::-1] if rev else path,
                          'donor_a':d1,'donor_b':d2,
                          'events':[[ev[0] for ev in group] for group in (bundles[::-1] if rev else bundles)]}
                    record(h,aa,bb,layout,wits,orientation=ori)
                    # A given native subject may belong to multiple certified subgraphs
                    # as a non-owner port. Iterate all such placements below.
                    for ca in actual_all.get((w,subjects[u],h,a,ra),()):
                        for cb in actual_all.get((w,subjects[v],h,b,rb),()):
                            if ca==cb:continue
                            if any(owners.get((h,x)) in (ca,cb) for x in path[1:-1] if x>=0):continue
                            extra=tuple(sorted(ev[0] for group in bundles for ev in group if owners.get((h,ev[4])) not in (ca,cb)))
                            joint=(w,min(ca,cb),max(ca,cb),extra)
                            wit=dict(wits,actual_cids=([cb,ca] if rev else [ca,cb]))
                            record(h,aa,bb,layout,wit,joint,ori)
    write_results(db,args,chosen,ranks,schemas,instances,orient_instances)


def write_results(db,args,chosen,ranks,schemas,instances,orient_instances):
    out=args.out
    mats=out/'COMPOSITION_MATRICES'
    mats.mkdir(exist_ok=True)
    def log(f,obj):
        f.write(dumped(obj)+'\n')
    counts=[];sumrows=[];selfrows=[]; examples={}
    supported=set()
    with (out/'CONNECTION_EVIDENCE.jsonl').open('w') as evidence,(out/'COMPOSITION_EXAMPLES.jsonl').open('w') as samples:
        for (h,a,b,shape,ori),(wit,joints,match) in sorted(schemas.items()):
            supported.add((h,a,b))
            sig=hashlib.sha256((shape+'|'+ori).encode()).hexdigest()
            freq=len(joints)
            row=dict(hop=h,A=a,B=b,structure=sig,orientation=ori,definition=shape,witness=wit)
            log(evidence,row)
            counts.append((h,a,b,'structure',sig,ori,freq))
            key=(h,'zero' if freq==0 else 'low' if freq<=3 else 'higher')
            if key not in examples or (freq>examples[key][0] if key[1]=='higher' else freq<examples[key][0]):
                examples[key]=(freq,dict(row,frequency=freq,f_A=ranks[h,a][1],f_B=ranks[h,b][1],
                    matched_example=match,category=key[1]))
        for _,example in examples.values():log(samples,example)
    for h,types in chosen.items():
        vals=[]
        for a,b in itertools.combinations_with_replacement(sorted(types),2):
            n=len(instances[h,a,b])
            if a!=b:
                if (h,a,b) not in supported:raise AssertionError(('unwitnessed',h,a,b))
                vals.append(n)
            else:selfrows.append((h,a,int((h,a,b) in supported),n))
            if (h,a,b) in supported:counts.append((h,a,b,'union','','all',n))
        def mat(ori):
            for a in types:
                line=[a]
                for b in types:
                    x,y=sorted((a,b))
                    if a==b:value=ranks[h,a][1]
                    elif ori is None:value=len(instances[h,x,y])
                    else:
                        side='A_to_B' if a<b else 'B_to_A'
                        value=len(orient_instances[h,x,y,side])
                    line.append(value)
                yield line
        for label,ori in [('union',None),('directed',True)]:
            output(mats/f'hop_{h}_{label}.csv','type '+' '.join(types),mat(ori))
        sumrows.append((h,args.top,len(types),len(vals),sum(n==0 for n in vals),sum(n>0 for n in vals),sum(vals)))
    output(out/'COMPOSITION_COUNTS.csv','hop A B level structure orientation frequency',counts)
    output(out/'SAME_TYPE_COUNTS.csv','hop type has_evidence unique_joints',selfrows)
    output(out/'HOP_SUMMARY.csv','hop initial_N selected_n distinct_pairs zero_pairs seen_pairs total_unique_pair_matches',sumrows)
    print('complete',sumrows,flush=True)
