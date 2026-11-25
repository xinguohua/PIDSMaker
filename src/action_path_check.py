"""Compare path DP against ordered event combinations on an actual Velox path."""
from collections import Counter, defaultdict
from itertools import combinations


def check_real_window(groups, expected_lengths, compute, endpoints=None):
    # An event can extend only prefixes from strictly earlier groups.
    chains = {}
    chain = ()
    target = max(expected_lengths)
    for group in groups:
        pending = {}
        for event in group:
            _, u, v, _ = event
            prefix = chains.get(u, ()) + (event,)
            if len(prefix) > len(chain):
                chain = prefix
            if len(chain) == target:
                break
            if len(prefix) > len(pending.get(v, ())):
                pending[v] = prefix
        if len(chain) == target:
            break
        for v, prefix in pending.items():
            if len(prefix) > len(chains.get(v, ())):
                chains[v] = prefix

    anchors = {node for event in chain for node in event[1:3]}
    selected = {event[0] for event in chain}
    for group in groups:
        for row, u, v, _ in group:
            if u in anchors or v in anchors:
                selected.add(row)
            if len(selected) >= 18:
                break
        if len(selected) >= 18:
            break
    sample = []
    order = {}
    for i, group in enumerate(groups):
        batch = [e for e in group if e[0] in selected]
        if batch:
            sample.append(batch)
            order.update((e[0], i) for e in batch)
    events = [e for group in sample for e in group]
    observed = compute(events, True, groups=sample, endpoints=endpoints)
    witness_counts = {}
    for k in expected_lengths:
        counts = Counter()
        members = defaultdict(set)
        for path in combinations(events, k):
            if all(left[2] == right[1] and order[left[0]] < order[right[0]] for left, right in zip(path, path[1:])):
                seq = tuple(e[3] for e in path)
                counts[seq] += 1
                members[seq].update(n for e in path for n in (endpoints[e[0]] if endpoints is not None else e[1:3]))
        for seq in counts.keys() | observed[k].keys():
            got = observed[k].get(seq)
            if got is None or got[0] != counts[seq] or got[2] != members[seq]:
                raise ValueError(f"Reference mismatch: length={k}, sequence={seq}, "
                                 f"expected={(counts[seq], members[seq])}, got={got}, groups={sample}")
        witness_counts[k] = sum(counts.values())
    return dict(event_rows=[e[0] for e in events], longest_path_event_rows=[e[0] for e in chain],
                longest_requested_length_found=len(chain) == target,
                exact_instances=witness_counts, all_counts_and_node_memberships_equal=True)
