"""Independent native windows, one reader per process and one transactional result writer."""
from collections import Counter, defaultdict
import multiprocessing
import pickle
import resource
import sys
from action_index import ActionIndex, Participation, LENGTHS
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import time

_context = {}
_connection = None
_support = None


def configure(root, attrs, keys, pred, attack, missing, train_actions, train_types, functions):
    global _context
    _context = dict(root=root, attrs=attrs, keys=keys, pred=pred, attack=attack,
                    missing=missing, train_actions=train_actions, train_types=train_types,
                    functions=functions)


def initialize():
    global _connection, _support
    path = _context['root'] / 'native.sqlite'
    _connection = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    _connection.execute('pragma mmap_size=8589934592')
    _connection.execute('attach database ? as sem',(f'file:{path.parent / "semantic.sqlite"}?mode=ro&immutable=1',))
    _connection.execute('pragma sem.mmap_size=8589934592')
    if isinstance(_context['train_types'], Path):
        _support = sqlite3.connect(f"file:{_context['train_types']}?mode=ro", uri=True)


def supported_batch(k, blobs):
    """Exact indexed lookups bounded by the output batch, never all train types."""
    if not blobs:
        return set()
    placeholders=','.join('?' for _ in blobs)
    return {b for b, in _support.execute(
        f"select seq from counts where k=? and seq in ({placeholders}) and train!='0'",
        (k,*blobs))}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == 'darwin' else 1024)



def independent_instance_counts(edges):
    prefixes = defaultdict(lambda: [0] * max(LENGTHS))
    result = [0] * max(LENGTHS)
    for batch in edges:
        for offset in range(max(LENGTHS)-1, -1, -1):
            for _, u, v, _ in batch:
                n = prefixes[u][offset-1] if offset else 1
                prefixes[v][offset] += n
                result[offset] += n
    return result


def read_window(db, w, test, attrs, keys, pred, attack):
    edges = db.execute('''select e.row,e.u,e.v,e.ns,e.op,e.uuid,e.y,r.seq
        from edges e join sem.source r using(w,row) where e.w=? order by e.ns,r.seq''', (w,))
    endpoints = {}
    uuids = set()
    previous_order = None
    ordered_groups=[]
    excluded = Counter()
    for row, u, v, ns, op, uuid, label, seq in edges:
        assert uuid not in uuids, (w, row, 'duplicate event')
        uuids.add(uuid)
        if label or test and (u in attack or v in attack):
            excluded['attack'] += 1
        elif test and (u not in pred or v not in pred):
            excluded['unknown'] += 1
        else:
            action = keys[attrs[u][0], op, attrs[v][0]]
            edge=(row,u,v,action)
            endpoints[row]=(u,v)
            if previous_order != (ns,seq):
                ordered_groups.append([])
                previous_order=(ns,seq)
            ordered_groups[-1].append(edge)
    assert len(uuids) == db.execute('select count(*) from edges where w=?',(w,)).fetchone()[0], 'missing/duplicate source evidence'
    return ordered_groups, endpoints, len(uuids), excluded


def compute_one(job, spool_dir):
    w, split = job
    ctx = _context
    paths, check, pack = ctx['functions']
    begin = time.monotonic()
    test = split == 'test'
    ordered_groups, endpoints, native_count, excluded = read_window(
        _connection, w, test, ctx['attrs'], ctx['keys'], ctx['pred'], ctx['attack'])
    read = time.monotonic()
    lengths = LENGTHS
    real = check([group[0] for group in ordered_groups], lengths, paths, endpoints) if w == 0 else None
    if w == 0 and real is None:
        raise ValueError('No reference chain in validation window')
    index = ActionIndex(ordered_groups)
    indexed = time.monotonic()
    exact_total = independent_instance_counts(ordered_groups)
    oracle_done = time.monotonic()
    audit = dict(window=w, split=split, native_edges=native_count, normal_edges=len(endpoints),
                 attack_edges=excluded['attack'], unknown_label_edges=excluded['unknown'])
    audit['same_order_extra_events'] = len(endpoints)-len(ordered_groups)
    flags = {}
    spool_path = Path(spool_dir) / f'{w}.bin'
    rows = []
    write_s = membership_s = support_s = 0.0
    def progress(k, idx):
        print(dict(window=w, phase='count', length=k, seconds=round(time.monotonic()-begin, 3),
                   branches=idx.branches, operations=idx.operations, peak_rss_mb=rss_mb()), flush=True)
    with spool_path.open('xb') as spool:
        for k in lengths:
            started = time.monotonic()
            participation = Participation()
            types = instances = unknown = 0
            from itertools import islice
            stream = index.sequences(k, progress)
            while batch := list(islice(stream, 400)):
                tick = time.monotonic()
                blobs = [pack(seq) for seq, _ in batch]
                seen = supported_batch(k, blobs) if test and k > 1 else set()
                support_s += time.monotonic()-tick
                for (seq, n), blob in zip(batch, blobs):
                    types += 1
                    instances += n
                    complete = ctx['missing'].isdisjoint(seq)
                    if not complete:
                        unknown += n
                    if test and k > 1:
                        tick = time.monotonic()
                        if not complete:
                            bit = 8
                        elif blob in seen:
                            bit = 1
                        else:
                            bit = 2 if all(t in ctx['train_actions'] for t in seq) else 4
                        participation.add(index, seq, bit)
                        membership_s += time.monotonic()-tick
                    rows.append((k, blob, '0' if test else str(n), str(n) if test else '0',
                                 w))
                tick = time.monotonic()
                pickle.dump(rows, spool, protocol=5)
                rows.clear()
                write_s += time.monotonic()-tick
            tick = time.monotonic()
            if test and k > 1:
                flags.update(((k, n), b) for n, b in participation.flags(index, endpoints).items())
            membership_s += time.monotonic()-tick
            del participation
            audit['types_'+str(k)] = types
            audit['instances_'+str(k)] = instances
            audit['semantic_unknown_instances_'+str(k)] = unknown
            audit['length_'+str(k)+'_s'] = round(time.monotonic()-started, 6)
            assert instances == exact_total[k-1]
            progress(k, index)
        tick = time.monotonic()
        if rows:
            pickle.dump(rows, spool, protocol=5)
        pickle.dump(None, spool, protocol=5)
        write_s += time.monotonic()-tick
    audit.update(read_s=round(read-begin, 6), index_s=round(indexed-read, 6),
                 oracle_s=round(oracle_done-indexed, 6), support_query_s=round(support_s,6), membership_s=round(membership_s, 6),
                 spool_write_s=round(write_s, 6), total_s=round(time.monotonic()-begin, 6),
                 peak_rss_mb=rss_mb(), operations=index.operations, branches=index.branches)
    return audit, spool_path, flags, real


def window_results(jobs, workers):
    if not jobs:
        return
    with TemporaryDirectory(prefix='action-windows-', dir=_context['root']) as spool_dir:
        yield from _run_windows(jobs, workers, spool_dir)


def result_rows(rows):
    if isinstance(rows, list):
        yield from rows
        return
    with rows.open('rb') as spool:
        while (batch := pickle.load(spool)) is not None:
            yield from batch
    rows.unlink()


def _run_windows(jobs, workers, spool_dir):
    # A single window is validated before the first parallel batch is started.
    if jobs[0][0] == 0:
        initialize()
        yield compute_one(jobs[0], spool_dir)
        jobs = jobs[1:]
    if jobs:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        context = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize) as pool:
            it = iter(jobs)
            pending = {pool.submit(compute_one, job, spool_dir) for job in (jobs[:workers])}
            for _ in range(len(pending)):
                next(it)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                while finished:
                    yield finished.pop().result()
                    job = next(it, None)
                    if job is not None:
                        pending.add(pool.submit(compute_one, job, spool_dir))
