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


def run(db,args,chosen,direct,search,ranks):
    out=args.out
    db.execute('create temp table chosen(hop int,tag varchar)')
    db.executemany('insert into chosen values (?,?)',[(h,t) for h,ts in chosen.items() for t in ts])
    db.execute("""create temp table p as select hop,tag,cid,uid,w,sid,'member' AS role,
        NULL::bigint eid from m semi join chosen using(hop,tag)""")
    db.execute("""create temp table r as select p.*,a.ord from ports p
        semi join chosen using(hop,tag) semi join shared using(rid) join t.actor a using(eid)""")
    members={(h,u):(c,t) for h,u,c,t in db.execute('select hop,uid,cid,tag from p').fetchall()}
    allcid={(h,u):c for h,u,c in db.execute('select hop,uid,cid from assignment').fetchall()}
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
        from p a join forks f on f.src_uid=a.uid or f.ch=a.sid join p b on {common}
        and ((f.src_uid=a.uid and b.sid=f.ch) or (f.ch=a.sid and b.uid=f.src_uid))"""
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

    programs=dict(db.execute('select uid,program from g.units').fetchall())
    # Materialize only witnessed paths under the fixed terminal-port rule. Store full
    # edge bundles once per window; every row points back to the original eid.
    for w,records in itertools.groupby(search.paths(chosen,direct),key=lambda item:item[0]):
        edge=defaultdict(list)
        for x,y,e,ts,op,d,owner,kind in db.execute("""select r.uid,-1-r.resource,e.eid,e.ts,e.op,e.direction,e.uid,n.kind
            from g.resource_events r join t.actor e using(eid) join g.nodes n on n.nid=r.resource
            where r.segment=? and e.eid>=? and e.eid<? union all
            select src_uid,dst_uid,eid,ts,op,'actor_to_object',src_uid,'subject' from forks where w=?""",
            [w,w<<32,(w+1)<<32,w]).fetchall():
            edge[min(x,y),max(x,y)].append((e,ts,op,d,owner,kind))
        for _,path,options in records:
            bundles=[edge[min(a,b),max(a,b)] for a,b in zip(path,path[1:])]
            assert all(bundles)
            times=sorted({ev[1] for gr in bundles for ev in gr})
            tick={ts:i for i,ts in enumerate(times)}
            layouts={}
            for reverse in {rev for _,rev,_,_ in options}:
                seq=path[::-1] if reverse else path
                bs=bundles[::-1] if reverse else bundles
                nodes=[programs[x] if x>=0 else bs[i-1][0][5] for i,x in enumerate(seq)]
                links=[sorted((op,d,owner==seq[i],tick[ts]) for e,ts,op,d,owner,kind in gr)
                        for i,gr in enumerate(bs)]
                l=min(ev[1] for ev in bs[0]);r=min(ev[1] for ev in bs[-1])
                ori='A_to_B' if l<r else 'B_to_A' if l>r else 'simultaneous'
                layouts[reverse]=('native_chain:'+dumped([nodes,links]),ori)
            for (h,a,b),rev,da,dbu in options:
                ca,ta=members[h,da];cb,tb=members[h,dbu]
                u,v=path[0],path[-1]
                acta=members.get((h,u));actb=members.get((h,v))
                if ca==cb:
                    if acta and actb and acta[1]==ta and actb[1]==tb and acta[0]!=actb[0]:
                        ca,cb=acta[0],actb[0];da,dbu=u,v
                    else:continue
                shape,ori=layouts[rev]
                wit=dict(window=w,path=path[::-1] if rev else path,
                    donor_cids=[cb,ca] if rev else [ca,cb],donor_uids=[dbu,da] if rev else [da,dbu],
                    event_bundles=[[ev[0] for ev in gr] for gr in (bundles[::-1] if rev else bundles)])
                joint=None
                if acta and actb and acta[1]==ta and actb[1]==tb and acta[0]!=actb[0]:
                    ac,bc=acta[0],actb[0]
                    if not any(allcid[h,x] in (ac,bc) for x in path[1:-1] if x>=0):
                        extra=tuple(sorted(ev[0] for gr in bundles for ev in gr if allcid[h,ev[4]] not in (ac,bc)))
                        joint=(w,min(ac,bc),max(ac,bc),extra)
                        wit['actual_cids']=[bc,ac] if rev else [ac,bc]
                record(h,a,b,shape,wit,joint,ori)
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
