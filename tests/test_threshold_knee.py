import itertools
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from threshold_knee import apply_knee, fit_knee


def rows(scores_and_labels):
    return [dict(node_id=i, score=s, gt=y) for i, (s,y) in enumerate(scores_and_labels)]


class KneeTests(unittest.TestCase):
    def test_nonconvex_candidate_is_skipped_and_ties_are_atomic(self):
        # Supporting points in increasing TP order: (0,0), (1,0), (3,2), (4,6).
        data = rows([(9,1)] + [(8,0)]*2 + [(8,1),(7,1)] +
                    [(6,0)]*4 + [(6,1)])
        result = fit_knee(reversed(data), 2)
        self.assertEqual(result['threshold'], math.nextafter(7., -math.inf))
        self.assertEqual(result['previous_supported_point']['marginal_fp_per_tp'], 1)
        self.assertEqual(result['next_supported_point']['marginal_fp_per_tp'], 4)
        self.assertEqual(apply_knee(data,result['threshold']), result['operating_point'])
        self.assertEqual(fit_knee(data, 1)['operating_point']['TP'], 1)

    def test_matches_exhaustive_utility(self):
        rng = random.Random(51)
        for _ in range(60):
            data = rows([(rng.randrange(-5, 5),y) for y in [0, 1]*12])
            thresholds = [max(r['score'] for r in data)]
            thresholds.extend(math.nextafter(s,-math.inf) for s in sorted({r['score'] for r in data}, reverse=True))
            for cost in (0.5, 1, 2, 100):
                def rank(t):
                    tp = sum(r['gt'] for r in data if r['score'] > t)
                    fp = sum(1-r['gt'] for r in data if r['score'] > t)
                    return cost*tp-fp, -fp, t
                expected = max(thresholds,key=rank)
                found = fit_knee(data,cost)
                self.assertEqual(found['threshold'],expected)
                self.assertEqual(found['operating_point'],apply_knee(data,expected))

    def test_unavailable_and_no_alerts_are_distinct(self):
        for data, reason in [([], 'empty_calibration'), ([(1,0)], 'no_positives'),
                             ([(1,1)], 'no_negatives'), ([(0,0),(0,1)], 'constant_score')]:
            result = fit_knee(rows(data))
            self.assertEqual((result['status'],result['reason']),('unavailable',reason))
        data = rows([(1,0),(0,1)])
        fit = fit_knee(data,1)
        self.assertEqual((fit['status'],fit['operating_point']['TP']),('selected',0))
        extreme = fit_knee(rows([(0,0),(-sys.float_info.max,1)]))
        self.assertTrue(math.isfinite(extreme['threshold']))
        self.assertIn('unreachable_minimum_score',extreme['calibration'])

    def test_row_validation(self):
        data = rows([(1,1),(0,0)])
        for key, bad in [('score', float('nan')), ('gt',0.5), ('score_direction','lower_is_anomalous')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_knee([dict(data[0], **{key:bad}),data[1]])
        with self.assertRaisesRegex(ValueError,'unique'):
            apply_knee(itertools.chain(data,data),1)
        with self.assertRaises(ValueError):
            fit_knee(data,0)


if __name__ == '__main__':
    unittest.main()
