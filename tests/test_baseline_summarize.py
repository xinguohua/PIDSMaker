"""Consumer-side integrity regressions for interrupted and altered exports."""
import json
from pathlib import Path
import tempfile
import unittest
from test_baseline_protocol import b

class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cell = Path(self.tmp.name)
        for name in b.EXPORT_FILES:
            (self.cell/name).write_text('{}')
        self.objects = self.cell/'objects.jsonl'
        self.objects.write_text(json.dumps(dict(node_id=1, gt=1, prediction=1))+'\n')
        self.marker = self.cell/'export_complete.json'
        self.marker.write_text(json.dumps({'files':{n:b.digest(self.cell/n) for n in b.EXPORT_FILES}}))
    def test_valid_publication_recomputes_metrics(self):
        self.assertEqual(b.summarize(self.objects)['TP'],1)
        self.assertEqual(b.summarize(self.objects)['f1'],1)
    def test_interruption_after_object_rename_without_marker(self):
        self.marker.unlink()
        with self.assertRaisesRegex(ValueError,'completion marker'): b.summarize(self.objects)
    def test_each_export_file_tampering_is_rejected(self):
        for name in b.EXPORT_FILES:
            with self.subTest(name=name):
                p=self.cell/name;original=p.read_bytes();p.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'hash mismatch'): b.summarize(self.objects)
                p.write_bytes(original)
    def test_incomplete_marker_cannot_bypass_missing_file(self):
        (self.cell/'checkpoint.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing or indirect'): b.summarize(self.objects)
        d=json.loads(self.marker.read_text());del d['files']['checkpoint.json'];self.marker.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'every expected'): b.summarize(self.objects)
    def test_failed_cell_and_failed_export_override_valid_marker(self):
        for name,value in [('status.json',{'state':'failed'}),('export_failed.json',{})]:
            with self.subTest(name=name):
                p=self.cell/name;p.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'Failed'): b.summarize(self.objects)
                p.unlink()

if __name__=='__main__': unittest.main()
