"""Select a node-score operating point using a fixed marginal false-alert cost."""
import argparse
import json
import math
import numbers
from fractions import Fraction
from pathlib import Path

METHOD = 'marginal_fp_per_tp_knee'
DEFAULT_FP_PER_TP = 100


def _scores(rows):
    seen = set()
    for row in rows:
        node, label, score = row['node_id'], row['gt'], row['score']
        if isinstance(node, bool) or not isinstance(node, numbers.Integral) or node in seen:
            raise ValueError('Expected a unique integer node_id')
        if isinstance(label, bool) or not isinstance(label, numbers.Integral) or label not in (0, 1):
            raise ValueError('Expected binary integer gt')
        if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score):
            raise ValueError('Expected a finite node score')
        if row.get('score_direction', 'higher_is_anomalous') != 'higher_is_anomalous':
            raise ValueError('Expected higher_is_anomalous score direction')
        seen.add(node)
        yield float(score), label


def _metrics(tp, fp, p, n):
    fn, tn = p - tp, n - fp
    return dict(unit='node', positive='malicious', n=p+n, TP=tp, FP=fp, TN=tn, FN=fn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/p if p else None,
                fpr=fp/n if n else None, fnr=fn/p if p else None,
                f1=2*tp/(p+tp+fp) if p+tp+fp else None)


def fit_knee(rows, fp_per_tp=DEFAULT_FP_PER_TP):
    """Fit only on labeled development nodes; rows must represent a single scoring model.

    The lower convex envelope in (TP, FP) coordinates lists the score cutoffs
    supported by some positive cost budget. Maximize C*TP-FP, breaking equal
    utility toward the lower FP count. Counts and slope comparisons are exact.
    """
    cost = Fraction(str(fp_per_tp))
    if cost <= 0 or not math.isfinite(float(cost)):
        raise ValueError('fp_per_tp must be positive and finite')
    groups = {}
    for score, label in _scores(rows):
        if score not in groups:
            groups[score] = [0, 0]
        groups[score][label] += 1
    n = sum(v[0] for v in groups.values())
    p = sum(v[1] for v in groups.values())
    result = dict(method=METHOD, fp_per_tp=float(cost), comparison='score > threshold',
                  calibration=dict(positives=p, negatives=n, nodes=p+n, distinct_scores=len(groups)),
                  status='unavailable', threshold=None, operating_point=None)
    if p == 0 or n == 0 or len(groups) < 2:
        reason = ('empty_calibration' if not groups else 'no_positives' if p == 0 else
                  'no_negatives' if n == 0 else 'constant_score')
        return dict(result, reason=reason)

    tp = fp = 0
    hull = [(0, 0, max(groups))]  # TP, FP, highest strict cutoff realizing this point
    for score in sorted(groups, reverse=True):
        normal, malicious = groups[score]
        fp += normal
        tp += malicious
        if not malicious:
            continue  # More FP at the previous TP count is dominated.
        threshold = math.nextafter(score, -math.inf)
        if not math.isfinite(threshold):
            result['calibration']['unreachable_minimum_score'] = score
            continue
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (b[1]-a[1])*(tp-b[0]) < (fp-b[1])*(b[0]-a[0]):
                break
            hull.pop()
        hull.append((tp, fp, threshold))

    def utility(point):
        return cost.numerator * point[0] - cost.denominator * point[1]

    selected = max(range(len(hull)), key=lambda i: (utility(hull[i]), -hull[i][1], hull[i][2]))
    tp, fp, threshold = hull[selected]
    point = _metrics(tp, fp, p, n)
    result.update(status='selected', reason='no_alerts_at_this_cost' if selected == 0 else 'maximum_net_benefit',
                  threshold=threshold, operating_point=point, supported_point_count=len(hull))
    for name, i in [('previous', selected-1), ('next', selected+1)]:
        neighbor = None
        if 0 <= i < len(hull):
            other_tp, other_fp, other_threshold = hull[i]
            extra_tp, extra_fp = abs(other_tp-tp), abs(other_fp-fp)
            neighbor = dict(threshold=other_threshold, TP=other_tp, FP=other_fp,
                            delta_TP=extra_tp, delta_FP=extra_fp, marginal_fp_per_tp=extra_fp/extra_tp)
        result[name+'_supported_point'] = neighbor
    return result


def apply_knee(rows, threshold):
    """Evaluate a frozen strict cutoff on a separate node export."""
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real) or not math.isfinite(threshold):
        raise ValueError('Expected a finite threshold')
    tp = fp = p = n = 0
    for score, label in _scores(rows):
        p += label
        n += 1-label
        if score > threshold:
            tp += label
            fp += 1-label
    return _metrics(tp, fp, p, n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', type=Path, help='Labeled development objects.jsonl')
    parser.add_argument('--fp-per-tp', default=DEFAULT_FP_PER_TP,
                        help='Maximum false alerts worth one more detected node (default 100)')
    parser.add_argument('--output', type=Path, help='Write small JSON summary (default stdout)')
    args = parser.parse_args()
    with args.objects.open() as f:
        result = fit_knee(map(json.loads, f), args.fp_per_tp)
    result['fitted_on'] = str(args.objects.resolve())
    message = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(message+'\n')
    else:
        print(message)


if __name__ == '__main__':
    main()
