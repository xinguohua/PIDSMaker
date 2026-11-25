"""Compare path DP against ordered event combinations on an actual Velox path."""
from collections import Counter, defaultdict
from itertools import combinations


def check_real_window(edges, expected_lengths, compute):
    # Pick a real length-five chain by entity identity, without consulting type counts.
    chains = {}
    for i, (_, u, v, _) in enumerate(edges):
        chain = chains.get(u, ()) + (i,)
        if len(chain) > len(chains.get(v, ())):
            chains[v] = chain
        if len(chain) == 5:
            break
    else:
        return None
    anchors = {n for pos in chain for n in edges[pos][1:3]}
    positions = set(chain)
    for i, (_, u, v, _) in enumerate(edges):
        if u in anchors or v in anchors:
            positions.add(i)
        if len(positions) >= 18:
            break
    events = [edges[i] for i in sorted(positions)]
    observed = compute(events, True)
    witness_counts = {}
    for k in expected_lengths:
        counts = Counter()
        members = defaultdict(set)
        for path in combinations(events, k):
            if all(left[2] == right[1] for left, right in zip(path, path[1:])):
                seq = tuple(e[3] for e in path)
                counts[seq] += 1
                members[seq].update(n for e in path for n in e[1:3])
        assert {s: result[0] for s, result in observed[k].items()} == counts
        assert {s: result[2] for s, result in observed[k].items()} == members
        witness_counts[k] = sum(counts.values())
    return dict(event_rows=[e[0] for e in events], length_five_event_rows=[edges[i][0] for i in chain],
                exact_instances=witness_counts, all_counts_and_node_memberships_equal=True)
