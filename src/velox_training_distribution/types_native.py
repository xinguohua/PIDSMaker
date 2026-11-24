"""Count complete typed shapes on Velox native training windows, jointly over ten partitions."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import igraph

VERSION = 'velox-native-endpoint-order-v1'


def packed(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(packed(value)).hexdigest()


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def session(root):
    con = duckdb.connect(str(root/'native_types.duckdb'))
    con.execute("SET threads=4; SET memory_limit='12GB'; SET preserve_insertion_order=false")
    con.execute('ATTACH '+sql_path(root/'native_units.duckdb')+" AS native (READ_ONLY)")
    return con


def stream(cur):
    while True:
        part = cur.fetchmany(4096)
        if not part:
            break
        yield from part


def create(con, name, query):
    if con.execute("SELECT count(*) FROM duckdb_tables() WHERE database_name=current_database() AND table_name=?", [name]).fetchone()[0]:
        return
    print('materializing', name, flush=True)
    con.execute('CREATE TABLE '+name+' AS '+query)
    con.execute('CHECKPOINT')


class Colored:
    """Sparse directed gadget graph, with loops and repeated observations encoded by colours."""
    def __init__(self):
        self.vertices = {}
        self.colors = []
        self.edges = []

    def vertex(self, key, color):
        i = self.vertices.get(key)
        if i is None:
            i = self.vertices[key] = len(self.colors)
            self.colors.append(color)
        return i

    def link(self, a, b):
        self.edges.append((a,b))

    def canonical(self):
        palette = sorted(set(self.colors))
        ids = {s:i for i,s in enumerate(palette)}
        colors = [ids[c] for c in self.colors]
        g = igraph.Graph(n=len(colors), edges=self.edges, directed=True)
        p = g.canonical_permutation(color=colors)
        ordered = [None]*len(p)
        for i,j in enumerate(p):
            ordered[j] = colors[i]
        # Original node/edge/slot IDs never enter the serialization.
        arcs = sorted((p[i],p[j]) for i,j in self.edges)
        return packed([palette,ordered,arcs])


def color(*args):
    return packed(args).decode()


def endpoint_node(graph,n,meta):
    return graph.vertex(('o',n),color('object',*meta[n]))


def add_unit(graph,uid,program,subject,bundles,meta,unit_type=None):
    u = graph.vertex(('u',uid),color('unit',program,unit_type))
    if subject is not None:
        graph.link(u,endpoint_node(graph,subject,meta))
    for a,b,h in bundles:
        e = graph.vertex(('b',uid,a,b),color('bundle',h,a is not None,b is not None))
        graph.link(u,e)
        if a is not None:
            graph.link(endpoint_node(graph,a,meta),e)
        if b is not None:
            graph.link(e,endpoint_node(graph,b,meta))


def unit_certificate(uid,program,subject,bundles,meta):
    # With unary bundles, private object stars canonicalize by a multiset sort.
    if all(a is None or b is None or a==b for a,b,h in bundles):
        objects=defaultdict(list)
        empty=[]
        if subject is not None:objects[subject]
        for a,b,h in bundles:
            if a is None and b is None: empty.append(h)
            else: objects[a if a is not None else b].append((a is not None,b is not None,h))
        entries=sorted((meta[n], n==subject, sorted(values)) for n,values in objects.items())
        return packed(['star',program,sorted(empty),entries])
    g=Colored()
    add_unit(g,uid,program,subject,bundles,meta)
    return g.canonical()


def prepare(con,root):
    create(con,'actor',"""
      SELECT eid,owner uid,object1 a,object2 b,op,direction,ts,
        dense_rank() OVER (PARTITION BY owner ORDER BY ts NULLS LAST) ord
      FROM native.events
    """)
    # Encode every observation; same-nanosecond records have a multiset of op/direction.
    # Two bounded string aggregation levels prevent giant per-process string allocations.
    create(con,'bundles',"""
      WITH counts AS (SELECT uid,a,b,ord,op,direction,count(*) n FROM actor GROUP BY ALL),
      chunks AS (
       SELECT uid,a,b,ord//128 chunk,sha256(string_agg(to_json([ord::VARCHAR,op,direction,n::VARCHAR]),',' ORDER BY ord,op,direction)) h
       FROM counts GROUP BY uid,a,b,chunk
      )
      SELECT uid,a,b,sha256(string_agg(h,'' ORDER BY chunk)) h
      FROM chunks GROUP BY uid,a,b ORDER BY uid
    """)
    # A causal order witness exists wherever observations touch the same native
    # endpoint, including a process used as another process's destination.
    create(con,'steps',"""
      SELECT DISTINCT a.uid, endpoints.obj, a.ts, a.ord
      FROM actor a JOIN native.units u USING(uid),
        UNNEST([u.subject, a.a, a.b]) endpoints(obj)
      WHERE endpoints.obj IS NOT NULL
    """)
    create(con,'endpoint_use',"SELECT DISTINCT uid,obj FROM steps")
    create(con,'process',"""SELECT p.eid,p.src_uid src,p.dst_uid dst,p.op,
                    e.uid AS owner,e.ord,e.a,e.b
                    FROM native.proc_edges p JOIN actor e USING(eid)""")
    exists=con.execute("select count(*) from duckdb_tables() where database_name=current_database() and table_name='ut'").fetchone()[0]
    if exists: return
    units={u:(p,s) for u,p,s in con.execute('select uid,program,subject from native.units').fetchall()}
    meta=node_meta(con)
    groups=itertools.groupby(stream(con.execute('select uid,a,b,h from bundles order by uid')),key=lambda row:row[0])
    g=next(groups,None)
    counts=Counter()
    representatives={}
    with (root/'UNIT_TYPES_BY_INSTANCE.csv').open('w',newline='') as f,(root/'UNIT_CERTIFICATES.jsonl').open('w') as certs:
        wr=csv.writer(f);wr.writerow(['uid','unit_type'])
        for uid,(program,subject) in sorted(units.items()):
            if g and g[0]==uid:
                bb=[r[1:] for r in g[1]]
                g=next(groups,None)
            else: bb=[]
            cert=unit_certificate(uid,program,subject,bb,meta)
            ident=hashlib.sha256(cert).hexdigest()
            second=hashlib.blake2b(cert,digest_size=24).digest()
            if ident in representatives:
                if representatives[ident][0]!=second: raise ValueError('Unit certificate collision')
            else:
                representatives[ident]=(second,uid,program)
                certs.write(json.dumps([ident,json.loads(cert)])+'\n')
            counts[ident]+=1
            wr.writerow([uid,ident])
        if g is not None: raise AssertionError('Bundle without unit')
    create(con,'ut', 'select * from read_csv('+sql_path(root/'UNIT_TYPES_BY_INSTANCE.csv')+",columns={'uid':'BIGINT','unit_type':'VARCHAR'},header=true)")
    with (root/'EXECUTION_UNIT_TYPES.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['unit_type','frequency','program','example_uid'])
        for tag,n in sorted(counts.items(),key=lambda v:(-v[1],v[0])):
            _,uid,program=representatives[tag];wr.writerow([tag,n,program,uid])


def node_meta(con):
    query = """SELECT n.nid,n.kind,n.role,u.program
               FROM native.nodes n LEFT JOIN native.units u ON n.nid=u.subject"""
    return {n:(sys.intern(k),sys.intern(r or 'unknown'),sys.intern(p or ''))
            for n,k,r,p in stream(con.execute(query))}


def member_key(us):
    return hashlib.sha256(','.join(map(str,sorted(us))).encode()).hexdigest()


def compare_parts(con,root,order,hop,known,store,meta,units):
    fname=root/'COMPOSITION_ASSIGNMENTS'/f'{order}_{hop}.csv'
    con.execute('CREATE OR REPLACE TEMP TABLE assignment AS SELECT unit_id uid,composition_id cid FROM read_csv_auto('+sql_path(fname)+')')
    members=defaultdict(list)
    for cid,uid in con.execute('select cid,uid from assignment order by cid,uid').fetchall():
        members[cid].append(uid)
    ids={cid:member_key(us) for cid,us in members.items()}
    with_self={u for u, in con.execute('select distinct src from process where src=dst').fetchall()}
    todo={cid for cid,us in members.items() if (len(us)>1 or us[0] in with_self) and ids[cid] not in known}
    # An identical unit set implies identical original events, internal resource chronology and process links.
    for cid,us in members.items():
        if len(us)==1 and us[0] not in with_self:
            known[ids[cid]]=('S-'+units[us[0]][2], 0, 0)
    print(order,hop,'all',len(members),'new multi',len(todo),flush=True)
    if todo:
        con.execute('create or replace temp table todo as select unnest(?::BIGINT[]) cid', [sorted(todo)])
        con.execute('create or replace temp table m as select a.* from assignment a semi join todo using(cid)')
        # Exact resource time groups restricted to participants INSIDE each composition.
        con.execute("""create or replace temp table participants as
          select m.cid,i.uid,i.obj resource from endpoint_use i join m using(uid)""")
        con.execute("""create or replace temp table shared as
          select cid,resource from participants group by all having count(distinct uid)>=2""")
        con.execute("""create or replace temp table timelines as
          with ranked as (
            select a.cid,s.uid,s.obj,s.ord,dense_rank() over(partition by a.cid,s.obj order by s.ts nulls last) rr
            from steps s join m a using(uid) join shared z on z.cid=a.cid and z.resource=s.obj
          ), chunks as (
            select cid,uid,obj,ord//128 chunk,sha256(string_agg(concat(ord,':',rr),',' order by ord,rr)) h
            from ranked group by cid,uid,obj,chunk
          ) select cid,uid,obj,sha256(string_agg(h,'' order by chunk)) h from chunks group by cid,uid,obj""")
        con.execute("""create or replace temp table links as
            select a.cid,p.* from process p join m a on p.src=a.uid join m b on p.dst=b.uid and b.cid=a.cid""")
        comp_types=defaultdict(list)
        for cid,uid,obj,h in con.execute('select * from timelines order by cid,obj,uid').fetchall():
            comp_types[cid].append((uid,obj,h))
        links=defaultdict(list)
        for cid,eid,s,d,op,owner,ord_,a,b in con.execute('select * from links').fetchall():
            links[cid].append((eid,s,d,op,owner,ord_,a,b))
        # Full exact labelling for small components. Giant components first get a
        # necessary isomorphism invariant. Jointly resolve any distinct-member collisions after all partitions.
        for cid in sorted(todo):
            us=members[cid]; key=ids[cid]
            bound=comp_types[cid]; procs=links[cid]
            invariant=shape_invariant(us,bound,procs,units,meta)
            known[key]=(invariant,len({obj for _,obj,_ in bound if meta[obj][0].lower() != 'subject'}),len(procs))
            if invariant not in store:store[invariant]={}
            store[invariant][key]=(order,hop,cid)
        timeline_dir=root/'TYPE_STRUCTURE_INDEX';timeline_dir.mkdir(exist_ok=True)
        for table in ['timelines','links','m']:
            dest=timeline_dir/f'{order}_{hop}_{table}.parquet'
            con.execute('copy '+table+' to '+sql_path(dest)+' (format parquet, compression zstd)')
    return [(order,hop,cid,ids[cid],len(us)) for cid,us in members.items()]


def shape_invariant(us, bound, procs,units,meta):
    resource_bindings=defaultdict(list)
    for u,obj,trace in bound:
        resource_bindings[obj].append((units[u][2],trace))
    resources=Counter((meta[obj],tuple(sorted(values))) for obj,values in resource_bindings.items())
    processes=Counter((units[s][2],units[d][2],o,ord_) for eid,s,d,o,owner,ord_,a,b in procs)
    members=Counter(units[u][2] for u in us)
    return 'M-'+digest([sorted(members.items()),sorted(resources.items()),sorted(processes.items())])


def exact_graph(us,bundles,timelines,links,meta,units):
    graph=Colored()
    for u in us:
        program,subject,ut=units[u]
        add_unit(graph,u,program,subject,bundles.get(u,[]),meta,ut)
    for u,obj,h in timelines:
        edge=graph.vertex(('t',u,obj),color('endpoint_order',h))
        graph.link(graph.vertices['u',u],edge);graph.link(edge,endpoint_node(graph,obj,meta))
    for eid,s,d,op,owner,step,a,b in links:
        edge=graph.vertex(('p',eid,s,d,op),color('process',op,step))
        graph.link(graph.vertices['u',s],edge);graph.link(edge,graph.vertices['u',d])
        bundle=graph.vertices.get(('b',owner,a,b))
        if bundle is not None:graph.link(edge,bundle)
    return graph.canonical()


def resolve(con,root,store,known,meta,units):
    # Invariant groups are pooled over BOTH root orders and all five radii.
    batches=defaultdict(dict)
    for invariant,observed in store.items():
        if len(observed)>1:
            for key,(order,hop,cid) in observed.items():
                batches[order,hop][cid]=key
    exact_digests={}
    pre=len(store)
    with (root/'CANONICAL_COLLISIONS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['shape_invariant','membership_digest','exact_signature'])
        for (order,hop),lookup in batches.items():
            p=root/'TYPE_STRUCTURE_INDEX'
            def read(table):return 'read_parquet('+sql_path(p/f'{order}_{hop}_{table}.parquet')+')'
            con.execute('create or replace temp table resolve_cids as select unnest(?::BIGINT[]) cid', [sorted(lookup)])
            con.execute('create or replace temp table x as select uid,cid from '+read('m')+' semi join resolve_cids using(cid)')
            members=defaultdict(list);timelines=defaultdict(list);links=defaultdict(list)
            for u,cid in con.execute('select uid,cid from x').fetchall():members[cid].append(u)
            for cid,u,obj,h in con.execute('select * from '+read('timelines')+' semi join resolve_cids using(cid)').fetchall():timelines[cid].append((u,obj,h))
            for cid,eid,s,d,op,owner,step,a,b in con.execute('select * from '+read('links')+' semi join resolve_cids using(cid)').fetchall():
                links[cid].append((eid,s,d,op,owner,step,a,b))
            select='select cid,uid,a,b,h from x join bundles using(uid) order by cid'
            source=itertools.groupby(stream(con.execute(select)),key=lambda r:r[0]);group=next(source,None)
            for cid in sorted(lookup):
                bb=defaultdict(list)
                if group and group[0]==cid:
                    for _,u,a,b,h in group[1]:bb[u].append((a,b,h))
                    group=next(source,None)
                cert=exact_graph(members[cid],bb,timelines[cid],links[cid],meta,units)
                code=hashlib.sha256(cert).hexdigest();second=hashlib.blake2b(cert,digest_size=24).digest()
                if code in exact_digests and exact_digests[code]!=second:raise AssertionError('Canonical certificate collision')
                exact_digests[code]=second
                key=lookup[cid]
                inv,resources,procs=known[key]
                typ=inv+'.'+code
                known[key]=(typ,resources,procs)
                w.writerow([inv,key,typ])
    return dict(multiunit_invariant_buckets=pre,ambiguous_buckets=sum(len(x)>1 for x in store.values()),
                exact_membership_graphs_compared=sum(len(v) for v in batches.values()),
                exact_certificates=len(exact_digests),ambiguous_buckets_with_nonisomorphic_graphs=
                sum(len({known[k][0] for k in xs})>1 for xs in store.values()),
                count_membership_hashes=len(known),solver='igraph BLISS '+igraph.__version__)


def write_records(path,names,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(names);w.writerows(rows)


def report(con,root,placements,known,units):
    freq=Counter(t[2] for t in units.values())
    programs={t[2]:t[0] for t in units.values()}
    top=sorted(freq,key=lambda t:(-freq[t],t))[:12]
    # Publish display labels and the complete operation multiset for each ranked type.
    reps_for_top = {t:min(u for u,v in units.items() if v[2]==t) for t in top}
    operations = defaultdict(list)
    for u,op,direction,n in con.execute("""
            SELECT uid,op,direction,count(*) n FROM actor
            WHERE uid IN (SELECT unnest(?::BIGINT[]))
            GROUP BY ALL ORDER BY uid,n DESC,op,direction""", [list(reps_for_top.values())]).fetchall():
        operations[u].append((op,direction,n))
    labels = {}
    for tag,u in reps_for_top.items():
        names = [f"{op.removeprefix('EVENT_')}×{n}" for op,_,n in operations[u][:2]]
        if len(operations[u])>2:names.append('…')
        labels[tag] = programs[tag]+' · '+(', '.join(names) or 'no owned edge')

    metrics=[];f1=[];f2=[];counts=[];component=[];sizes=[];instance_rows=[];examples=[]
    def table(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    audit=table('INTEGRITY_AUDIT.csv');partition=table('PARTITION_STATISTICS.csv')
    assert {(r['order'],int(r['hops'])) for r in partition} == {(o,h) for o in ('forward','reverse') for h in range(1,6)}
    assert len(audit)==len(partition)==10
    zero=('missing_nodes','missing_events','duplicate_event_owners','unassigned_units','duplicate_unit_owners','extra_units','event_ids_or_field_mismatches')
    assert all(float(r['node_coverage'])==float(r['event_coverage'])==1 and all(int(r[k])==0 for k in zero) for r in audit)
    for (order,hop),entries in itertools.groupby(placements,key=lambda x:(x[0],x[1])):
        totals=Counter(); members=defaultdict(list)
        with (root/'COMPOSITION_ASSIGNMENTS'/f'{order}_{hop}.csv').open() as f:
            for r in csv.DictReader(f):members[int(r['composition_id'])].append(int(r['unit_id']))
        if sum(map(len,members.values()))!=len(units):raise AssertionError('Mismatched unit universe')
        reps={}; inventory={}; ps=next(r for r in partition if r['order']==order and int(r['hops'])==hop)
        pairs=Counter()
        for _,_,cid,key,n in entries:
            shape,nres,nproc=known[key];totals[shape]+=1;us=members[cid]
            instance_rows.append((order,hop,cid,key,shape,n,nres,nproc))
            if shape not in reps:
                reps[shape]=(cid,nres,nproc); inventory[shape]=Counter(units[u][2] for u in us)
            elif reps[shape][1:]!=(nres,nproc):raise AssertionError('Type signature disagrees with relation counts')
            components=inventory[shape]
            present=[t for t in top if components[t]]
            for a in present:
                for b in present:
                    if a!=b or components[a]>1:pairs[a,b]+=1
        for a in top:
            for b in top:
                f1.append((order,hop,a,b,pairs[a,b],labels[a],labels[b]))
        cells=Counter();num_multi=0
        for shape,k in totals.items():
            inv=inventory[shape]; n=sum(inv.values());cid,nres,nproc=reps[shape]
            m=min(freq[t] for t in inv); maxfreq=max(freq[t] for t in inv)
            assert all(k*v<=freq[t] for t,v in inv.items()), (order,hop,shape)
            counts.append((order,hop,shape,k,n,len(inv),m,maxfreq,cid,nres,nproc,sum(units[u][1] is None for u in members[cid])))
            component.extend((order,hop,shape,t,programs[t],v,freq[t],k) for t,v in sorted(inv.items()))
            if n>1:cells[m,k]+=1;num_multi+=1
        for (m,k),count in sorted(cells.items()):f2.append((order,hop,m,k,count,count*k))
        bysize=Counter(map(len,members.values()))
        sizes.extend((order,hop,n,q) for n,q in sorted(bysize.items()))
        metrics.append((order,hop,len(units),len(freq),len(members),len(totals),num_multi,bysize[1],len(members)-bysize[1],max(bysize),ps['internal_process_relations'],ps['cross_process_relations'],ps['internal_resource_unit_pairs'],ps['cross_resource_unit_pairs']))
        if order=='forward':
            for shape,k in totals.items():
                cid,nres,nproc=reps[shape]; us=members[cid]
                if 2<=len(us)<=5:
                    m=min(freq[units[u][2]] for u in us)
                    examples.append((hop,shape,cid,us,k,m,nres,nproc))
    write_records(root/'COMPOSITION_COUNTS.csv',['order','hop','composition_type','composition_frequency','units','distinct_unit_types','min_component_frequency','max_component_frequency','representative_composition_id','shared_resources_within','internal_process_relations','unresolved_actor_units'],counts)
    write_records(root/'COMPONENT_COMPOSITION_FREQUENCIES.csv',['order','hop','composition_type','unit_type','program','multiplicity','unit_frequency','composition_frequency'],component)
    write_records(root/'COMPOSITION_SIZE_DISTRIBUTION.csv',['order','hop','units_per_composition','composition_instances'],sizes)
    write_records(root/'COMPOSITION_TYPE_INSTANCES.csv',['order','hop','composition_id','member_key','composition_type','units','shared_resources_within','internal_process_relations'],instance_rows)
    write_records(root/'FIG1_ROWS.csv',['order','hop','type_a','type_b','composition_instances','label_a','label_b'],f1)
    write_records(root/'FIG2_ROWS.csv',['order','hop','min_component_frequency','composition_frequency','composition_types','composition_instances'],f2)
    write_records(root/'HOP_DISTRIBUTION.csv',['order','hop','unit_instances','unit_types','composition_instances','composition_types','multiunit_types','single_unit_instances','multiunit_instances','max_units','internal_process_relations','cross_process_relations','internal_resource_unit_pairs','cross_resource_unit_pairs'],metrics)
    write_records(root/'FIG1_RANKS.csv',['rank','unit_type','program','training_frequency','label','example_uid','operation_direction_counts'],
                  [(i,t,programs[t],freq[t],labels[t],reps_for_top[t],json.dumps(operations[reps_for_top[t]])) for i,t in enumerate(top,1)])
    return examples


def example_details(con, root, examples, units):
    """Extract complete small observations and their exact native input locations."""
    with (root/'NATIVE_WINDOWS.csv').open() as f:
        windows = {int(row['window_id']): row for row in csv.DictReader(f)}

    def rows(query, params=None):
        cursor = con.execute(query, params or [])
        names = [r[0] for r in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def events(us):
        return rows("""
            SELECT e.*, e.src AS native_src, e.dst AS native_dst
            FROM native.events e
            WHERE e.owner IN (SELECT unnest(?::BIGINT[]))
            ORDER BY e.ts,e.eid
            """, [us])

    n_events = dict(con.execute('SELECT uid,count(*) FROM actor GROUP BY uid').fetchall())
    freq = Counter(t[2] for t in units.values())
    by_program = defaultdict(lambda: defaultdict(list))
    for uid, (program, subject, tag) in units.items():
        if subject is not None and 0 < n_events.get(uid, 0) <= 20 and program not in ('UNKNOWN_PROGRAM', ''):
            by_program[program][tag].append(uid)
    programs = [(sum(map(len, group.values())), name) for name, group in by_program.items()
                if len(group) > 1 and max(map(len, group.values())) > 1]
    samples = []
    if programs:
        program = max(programs)[1]
        groups = by_program[program]
        tag = min((t for t in groups if len(groups[t]) >= 2), key=lambda t: (-len(groups[t]), t))
        different = min((t for t in groups if t != tag), key=lambda t: (-len(groups[t]), t))
        for uid in sorted(groups[tag])[:2] + [min(groups[different])]:
            ed = events([uid])
            samples.append(dict(uid=uid, program=program, subject=units[uid][1], unit_type=units[uid][2],
                                frequency=freq[units[uid][2]], native_window=windows[ed[0]['window_id']], events=ed))
    (root/'UNIT_TYPE_EXAMPLES.json').write_text(json.dumps(samples, indent=2))

    identified = [r for r in examples if all(units[u][1] is not None and units[u][0] != 'UNKNOWN_PROGRAM' for u in r[3])]
    examples = identified or examples
    size = lambda r: sum(n_events.get(u, 0) for u in r[3])
    choices = [r for r in examples if size(r) <= 150] or sorted(examples, key=size)[:100]
    users = defaultdict(set)
    dependencies = defaultdict(set)
    for u, resource in stream(con.execute('SELECT uid, resource FROM native.incidence')):
        users[u].add(resource)
    for a, b in stream(con.execute('SELECT DISTINCT src_uid,dst_uid FROM native.proc_edges WHERE src_uid<>dst_uid')):
        dependencies[a].add(b)
        dependencies[b].add(a)

    def branches(us):
        member_set = set(us)
        sharing = defaultdict(list)
        for u in us:
            for resource in users[u]:
                sharing[resource].append(u)
        # Degree in the unit-to-shared-resource incidence graph plus native process edges.
        return (any(len(v) >= 3 for v in sharing.values())
                or any(len(dependencies[u] & member_set) + sum(len(sharing[x]) > 1 for x in users[u]) >= 3 for u in us))

    diverse = [r for r in choices if len({units[u][2] for u in r[3]}) >= 2]
    by_gap = lambda r: (-r[5],r[4],size(r),r[0],r[2])
    selected = []
    seen_members = set()
    seen_types = set()
    selections = [
        ('distinct_activities_high_marginals', sorted((r for r in diverse if r[5]>r[4]), key=by_gap)),
        ('observed_branch', sorted((r for r in choices if len(r[3]) >= 3 and branches(r[3])), key=by_gap)),
        ('repeated_observed_shape', sorted((r for r in choices if r[4]>1), key=lambda r: (-r[4],-r[5],size(r),r[0],r[2]))),
        ('matching_marginal_and_joint', sorted((r for r in choices if r[5]==r[4]), key=lambda r: (-r[4],size(r),r[0],r[2]))),
        ('same_activity_multiple_units', sorted((r for r in choices if len({units[u][2] for u in r[3]}) == 1), key=by_gap)),
    ]
    for reason, pool in selections:
        for item in pool:
            members = tuple(sorted(item[3]))
            if members not in seen_members and item[1] not in seen_types:
                selected.append((reason, item))
                seen_members.add(members)
                seen_types.add(item[1])
                break
    tmp = root/'COMPOSITION_EXAMPLES.partial'
    with tmp.open('w') as f:
        for reason,(hop, shape, cid, us, count, minimum, nresources, nprocess) in selected:
            marks = ','.join(map(str, us))
            ed = events(us)
            unit_rows = rows(f'SELECT * FROM native.units WHERE uid IN ({marks}) ORDER BY uid')
            for u in unit_rows:
                t = units[u['uid']][2]
                u.update(unit_type=t, unit_type_frequency=freq[t])
            inc = rows(f'SELECT * FROM native.incidence WHERE uid IN ({marks}) ORDER BY uid,resource')
            proc = rows(f'SELECT * FROM native.proc_edges WHERE src_uid IN ({marks}) AND dst_uid IN ({marks}) ORDER BY ts,eid')
            ids = {v for e in ed for v in (e['subject'], e['object1'], e['object2']) if v is not None}
            ids.update(u['subject'] for u in unit_rows if u['subject'] is not None)
            nodes = rows('SELECT * FROM native.nodes WHERE nid IN (SELECT unnest(?::BIGINT[])) ORDER BY nid', [sorted(ids)])
            resource_ids = {i['resource'] for i in inc}
            window_ids = {u['segment'] for u in unit_rows}
            assert len(window_ids)==1 and len(ed)==sum(n_events.get(u,0) for u in us)
            record = dict(order='forward',hop=hop,composition_type=shape,composition_id=cid,frequency=count,
                          min_component_frequency=minimum,selection=reason,has_branch=branches(us),
                          distinct_unit_types=len({units[u][2] for u in us}), shared_resource_count=nresources,
                          units=unit_rows, events=ed, resources=[n for n in nodes if n['nid'] in resource_ids],
                          nodes=nodes, process_links=proc, resource_incidence=inc,
                          native_window=windows[next(iter(window_ids))])
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
    tmp.replace(root/'COMPOSITION_EXAMPLES.jsonl')


def run(root):
    con=session(root)
    prepare(con,root)
    print('single unit type inventory complete',flush=True)
    meta=node_meta(con)
    units={u:(p,s,t) for u,p,s,t in con.execute('select u.uid,u.program,u.subject,t.unit_type from native.units u join ut t using(uid)').fetchall()}
    known={};store={};placements=[]
    for order in ['forward','reverse']:
        for hop in range(1,6):
            a=root/'COMPOSITION_ASSIGNMENTS'/f'{order}_{hop}.csv'
            while not a.exists():time.sleep(4)
            placements.extend(compare_parts(con,root,order,hop,known,store,meta,units))
    audit=resolve(con,root,store,known,meta,units)
    audit['schema']=VERSION
    (root/'TYPE_CERTIFICATE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    # Publication waits for the independent per-event, per-node reconstruction audit.
    while True:
        p=root/'INTEGRITY_AUDIT.csv';q=root/'PARTITION_STATISTICS.csv'
        if p.exists() and q.exists():
            with p.open() as f, q.open() as g:
                if len(list(csv.DictReader(f)))==len(list(csv.DictReader(g)))==10:break
        time.sleep(4)
    examples=report(con,root,placements,known,units)
    example_details(con,root,examples,units)
    con.close()
    print('all ten distributions published',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    run(args.root.resolve())
