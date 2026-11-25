"""Independent native windows, one reader per process and one transactional result writer."""
from collections import Counter, defaultdict
import multiprocessing
import sqlite3
import time

_context = {}
_connection = None


def configure(root, attrs, keys, pred, attack, missing, train_actions, train_types, functions):
    global _context
    _context = dict(root=root, attrs=attrs, keys=keys, pred=pred, attack=attack,
                    missing=missing, train_actions=train_actions, train_types=train_types,
                    functions=functions)


def initialize():
    global _connection
    path = _context['root'] / 'native.sqlite'
    _connection = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    _connection.execute('pragma mmap_size=8589934592')
    _connection.execute('attach database ? as sem',(f'file:{path.parent / "semantic.sqlite"}?mode=ro&immutable=1',))
    _connection.execute('pragma sem.mmap_size=8589934592')


def independent_instance_counts(edges):
    prefixes = defaultdict(lambda: [0] * 5)
    result = [0] * 5
    for batch in edges:
        for offset in range(4, -1, -1):
            for _, u, v, _ in batch:
                n = prefixes[u][offset-1] if offset else 1
                prefixes[v][offset] += n
                result[offset] += n
    return result


def compute_one(job):
    w, split = job
    ctx = _context
    paths, check, pack = ctx['functions']
    begin = time.monotonic()
    test = split == 'test'
    edges = _connection.execute('''select e.row,e.u,e.v,e.ns,e.op,e.uuid,e.y,r.seq
        from edges e join sem.source r using(w,row) where e.w=? order by e.ns,r.seq''', (w,))
    good = []
    uuids = set()
    previous_order = None
    ordered_groups=[]
    excluded = Counter()
    for row, u, v, ns, op, uuid, label, seq in edges:
        assert uuid not in uuids, (w, row, 'duplicate event')
        uuids.add(uuid)
        if label or test and (u in ctx['attack'] or v in ctx['attack']):
            excluded['attack'] += 1
        elif test and (u not in ctx['pred'] or v not in ctx['pred']):
            excluded['unknown'] += 1
        else:
            action = ctx['keys'][ctx['attrs'][u][0], op, ctx['attrs'][v][0]]
            edge=(row,u,v,action)
            good.append(edge)
            if previous_order != (ns,seq):
                ordered_groups.append([])
                previous_order=(ns,seq)
            ordered_groups[-1].append(edge)
    read = time.monotonic()
    lengths = (1, 2, 3, 5)
    real = check([group[0] for group in ordered_groups], lengths, paths) if w == 0 else None
    if w == 0 and real is None:
        raise ValueError('No reference chain in validation window')
    totals = paths(good, test, ordered_groups)
    exact_total = independent_instance_counts(ordered_groups)
    dp_done = time.monotonic()
    audit = dict(window=w, split=split, native_edges=len(uuids), normal_edges=len(good),
                 attack_edges=excluded['attack'], unknown_label_edges=excluded['unknown'])
    audit['same_order_extra_events']=len(good)-len(ordered_groups)
    flags = defaultdict(int)
    rows = []
    for k, values in totals.items():
        audit['types_' + str(k)] = len(values)
        audit['instances_' + str(k)] = sum(value[0] for value in values.values())
        assert audit['instances_' + str(k)] == exact_total[k-1]
        unknown = 0
        for seq, (n, witness, nodes) in values.items():
            blob = pack(seq)
            complete = ctx['missing'].isdisjoint(seq)
            if not complete:
                unknown += n
            if test and k > 1:
                if not complete:
                    bit = 8
                elif blob in ctx['train_types'][k]:
                    bit = 1
                else:
                    bit = 2 if all(t in ctx['train_actions'] for t in seq) else 4
                for node in nodes:
                    flags[k, node] |= bit
            rows.append((k, blob, '0' if test else str(n), str(n) if test else '0', w, pack(witness)))
        audit['semantic_unknown_instances_' + str(k)] = unknown
    audit.update(read_s=round(read-begin, 4), dp_and_oracle_s=round(dp_done-read, 4),
                 reduce_s=round(time.monotonic()-dp_done, 4))
    return audit, rows, flags, real


def window_results(jobs, workers):
    if not jobs:
        return
    # A single window is validated before the first parallel batch is started.
    if jobs[0][0] == 0:
        initialize()
        yield compute_one(jobs[0])
        jobs = jobs[1:]
    if jobs:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        context = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize) as pool:
            it = iter(jobs)
            pending = {pool.submit(compute_one, job) for job in (jobs[:workers])}
            for _ in range(len(pending)):
                next(it)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                while finished:
                    yield finished.pop().result()
                    job = next(it, None)
                    if job is not None:
                        pending.add(pool.submit(compute_one, job))
