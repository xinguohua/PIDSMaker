"""Synthetic serialization regressions; no dataset processing or training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from test_baseline_protocol import b

CFG = NS(evaluation=NS(node_evaluation=NS(threshold_method='max-val', use_kmeans=False)))

class ExportTests(unittest.TestCase):
    def test_real_numpy_scalars(self):
        r = b.object_record(np.int64(7), dict(score=np.float32(.5), y_hat=np.int64(1),
            y_true=np.int32(0), tw_with_max_loss=np.int64(12),
            time_range=[np.int64(1), np.int64(2)], decision_threshold=np.float64(.4)), {7:'uuid'}, CFG)
        encoded = json.loads(json.dumps(r, allow_nan=False))
        self.assertIs(type(encoded['max_loss_window']), int)
        self.assertEqual(encoded['max_loss_window'],12)
        self.assertEqual(encoded['export_issues'], [])
    def test_missing_and_nonfinite_metadata_are_audited(self):
        r=b.object_record(1, dict(score=1.,y_hat=0,y_true=0,tw_with_max_loss=np.float64('nan'),
                                 decision_threshold=np.float64('inf')), {}, CFG)
        self.assertIsNone(r['max_loss_window']);self.assertIsNone(r['threshold'])
        self.assertTrue(any(x['reason']=='non_finite' for x in r['export_issues']))
        self.assertTrue(any(x['reason']=='missing' for x in r['export_issues']))
        json.dumps(r,allow_nan=False)
    def test_missing_window_sentinel_is_null(self):
        r=b.object_record(1,dict(score=1.,y_hat=0,y_true=0,tw_with_max_loss=np.int64(-1)),{},CFG)
        self.assertIsNone(r['max_loss_window'])
        self.assertTrue(any(x['reason']=='missing_sentinel' for x in r['export_issues']))
    def test_invalid_required_values_fail(self):
        for values in [dict(score=np.nan),dict(score=np.inf),dict(y_hat=None),dict(tw_with_max_loss=1.5)]:
            r=dict(score=1.,y_hat=0,y_true=0);r.update(values)
            with self.assertRaises(ValueError): b.object_record(1,r,{},CFG)
    def test_export_failure_leaves_no_partial_result(self):
        def fail(cfg, stage):
            (stage/'objects.jsonl').write_text('partial')
            raise ValueError('bad record')
        with tempfile.TemporaryDirectory() as d, patch.object(b,'_export',fail):
            path=Path(d)
            with self.assertRaises(ValueError): b.export(CFG,path)
            self.assertEqual([p.name for p in path.iterdir()],['export_failed.json'])
    def test_complete_export_has_hashes(self):
        names=('uuid_mapping.json','objects.jsonl','missing_gt.json','metrics.json','checkpoint.json')
        def complete(cfg,stage):
            for name in names: (stage/name).write_text('{}')
        with tempfile.TemporaryDirectory() as d, patch.object(b,'_export',complete):
            path=Path(d);b.export(CFG,path)
            marker=json.loads((path/'export_complete.json').read_text())
            self.assertEqual(set(marker['files']),set(names))
            with self.assertRaises(FileExistsError): b.export(CFG,path)

if __name__=='__main__':unittest.main()
