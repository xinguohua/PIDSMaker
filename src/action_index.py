"""Exact content-path counting over occupied Action/time groups, never instances."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import time

LENGTHS = (1, 2, 3)
PROTOCOL = 'native_paths_len3_v1'


def signature(value):
    """Lossless canonical JSON for the complete frozen attribute record."""
    return json.dumps(json.loads(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


class ActionIndex:
    def __init__(self, groups):
        self.times = defaultdict(list)
        self.events = defaultdict(list)
        self.counts = defaultdict(list)
        self.variants = defaultdict(list)
        self.ends = {}
        self.out = defaultdict(list)
        self.operations = 0
        self.branches = 0
        event_ids = set()
        for g, batch in enumerate(groups):
            for eid, u, v, action in batch:
                if eid in event_ids:
                    raise ValueError('duplicate event ID within window')
                event_ids.add(eid)
                label = action
                if action in self.ends and self.ends[action] != (u, v):
                    action = (u, v, label)
                if action not in self.ends:
                    self.ends[action] = (u, v)
                    self.out[u].append((label, action))
                    self.variants[label].append(action)
                if not self.times[action] or self.times[action][-1] != g:
                    self.times[action].append(g)
                    self.events[action].append([])
                    self.counts[action].append(0)
                self.events[action][-1].append(eid)
                self.counts[action][-1] += 1
        self.suffix = {}
        for a, cs in self.counts.items():
            suffix = [0] * (len(cs)+1)
            for i in range(len(cs)-1, -1, -1):
                suffix[i] = suffix[i+1] + cs[i]
            self.suffix[a] = suffix

    def extend(self, times, weights, action, terminal=False):
        """D(q+a,g) = c_a(g) * sum_{h<g} D(q,h)."""
        ats = self.times[action]
        if terminal:
            # Ordered two-pointer suffix dot product; no terminal DP vector.
            j = total = 0
            for h, weight in zip(times, weights):
                while j < len(ats) and ats[j] <= h:
                    j += 1
                    self.operations += 1
                total += weight * self.suffix[action][j]
                self.operations += 1
            return total
        result_times, result_weights = [], []
        j = cumulative = 0
        for g, count in zip(ats, self.counts[action]):
            while j < len(times) and times[j] < g:
                cumulative += weights[j]
                j += 1
                self.operations += 1
            if cumulative:
                result_times.append(g)
                result_weights.append(count*cumulative)
            self.operations += 1
        return result_times, result_weights

    def sequences(self, length, progress=None):
        if length not in LENGTHS:
            raise ValueError("unsupported path length")
        last_report = time.monotonic()
        def walk(seq, frontier):
            # frontier[node][group] counts paths ending on that exact native node.
            nonlocal last_report
            self.branches += 1
            if progress and time.monotonic() - last_report >= 5:
                progress(length, self)
                last_report = time.monotonic()
            options = defaultdict(list)
            if not seq:
                for label, variants in self.variants.items():
                    options[label] = variants
            else:
                for node, (ts, _) in frontier.items():
                    for label, edge in self.out.get(node, ()):
                        if self.times[edge][-1] > ts[0]:
                            options[label].append(edge)
            for label, variants in options.items():
                total = 0
                by_node = defaultdict(lambda: defaultdict(int))
                terminal = len(seq) + 1 == length
                for edge in variants:
                    u, v = self.ends[edge]
                    if not seq:
                        times, weights = self.times[edge], self.counts[edge]
                    elif terminal:
                        total += self.extend(*frontier[u], edge, terminal=True)
                        continue
                    else:
                        times, weights = self.extend(*frontier[u], edge)
                    if terminal:
                        total += sum(weights)
                    else:
                        for t, n in zip(times, weights):
                            by_node[v][t] += n
                if terminal and total:
                    yield seq+(label,), total
                elif by_node:
                    next_state = {}
                    for v, counts in by_node.items():
                        ts = sorted(counts)
                        if ts:
                            next_state[v] = (ts, [counts[t] for t in ts])
                    if next_state:
                        yield from walk(seq+(label,), next_state)
        yield from walk((), {})

    def _forward(self, seq):
        # Earliest reachable arrival and its event witness, per native endpoint.
        previous = {}
        layers = []
        for i, label in enumerate(seq):
            current = {}
            layers.append(previous)
            for edge in self.variants[label]:
                u, v = self.ends[edge]
                if i and u not in previous:
                    continue
                t, prefix = previous[u] if i else (-1, ())
                ts = self.times[edge]
                j = bisect_right(ts, t)
                if j < len(ts) and (v not in current or ts[j] < current[v][0]):
                    current[v] = (ts[j], prefix + (self.events[edge][j][0],))
            previous = current
        return layers, previous

    def bounds(self, seq):
        layers, _ = self._forward(seq)
        following = {}
        for i in reversed(range(len(seq))):
            current = {}
            for edge in self.variants[seq[i]]:
                u, v = self.ends[edge]
                if i+1 < len(seq) and v not in following:
                    continue
                ts = self.times[edge]
                hi = bisect_left(ts, following[v]) if i+1 < len(seq) else len(ts)
                if not hi:
                    continue
                current[u] = max(current.get(u, -1), ts[hi-1])
                if i and u not in layers[i]:
                    continue
                lo = bisect_right(ts, layers[i][u][0]) if i else 0
                if lo < hi:
                    yield edge, lo, hi
            following = current

    def witness(self, seq):
        _, final = self._forward(seq)
        return next(iter(final.values()))[1]

    def nodes(self, seq, endpoints):
        return {n for a, lo, hi in self.bounds(seq)
                for es in self.events[a][lo:hi] for eid in es for n in endpoints[eid]}


class Participation:
    """Union intervals per Action/category, bounded by occupied time groups."""
    def __init__(self):
        self.intervals = defaultdict(list)

    def add(self, index, seq, bit):
        for a, lo, hi in index.bounds(seq):
            intervals = self.intervals[a, bit]
            i = bisect_left(intervals, (lo,))
            if i and intervals[i-1][1] >= lo:
                i -= 1
            j = i
            while j < len(intervals) and intervals[j][0] <= hi:
                lo = min(lo, intervals[j][0])
                hi = max(hi, intervals[j][1])
                j += 1
            intervals[i:j] = [(lo, hi)]

    def flags(self, index, endpoints):
        result = defaultdict(int)
        for (a, bit), intervals in self.intervals.items():
            for lo, hi in intervals:
                for i in range(lo, hi):
                    for eid in index.events[a][i]:
                        for node in endpoints[eid]:
                            result[node] |= bit
        return result
