"""Compare the two specified partitions of the same units at each radius."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def assignments(path):
    with path.open() as stream:
        ids = {}
        for row in csv.DictReader(stream):
            u, c = row['unit_id'], row['composition_id']
            if u in ids:
                raise ValueError(f'Duplicate unit {u} in {path}')
            ids[u] = c
        return ids


def pair_count(groups):
    return sum(n*(n-1)//2 for n in groups.values())


def compare(path, hop):
    left = assignments(path/f'forward_{hop}.csv')
    right = assignments(path/f'reverse_{hop}.csv')
    if not left or left.keys() != right.keys():
        raise ValueError('The two partitions must cover the same nonempty unit universe')
    joint = Counter((a, right[u]) for u, a in left.items())
    a, b = Counter(left.values()), Counter(right.values())
    pairs = len(left)*(len(left)-1)//2
    la, rb, common = pair_count(a), pair_count(b), pair_count(joint)
    union = la+rb-common
    numerator = 2*(common*pairs-la*rb)
    denominator = (la+rb)*pairs-2*la*rb
    identical = [n for (x,y),n in joint.items() if n==a[x]==b[y]]
    return dict(hop=hop, units=len(left), forward_compositions=len(a),
                reverse_compositions=len(b), identical_member_sets=len(identical),
                units_in_identical_sets=sum(identical),
                coassigned_unit_pairs_forward=la, coassigned_unit_pairs_reverse=rb,
                coassigned_unit_pairs_both=common, pair_jaccard=common/union if union else 1.0,
                adjusted_rand_index=numerator/denominator if denominator else 1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assignments', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = [compare(args.assignments, h) for h in range(1,6)]
    with args.output.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
