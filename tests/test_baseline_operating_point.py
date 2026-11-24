"""Exercise the project-level knee publication on synthetic native node exports."""
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline_knee import publish_knee
from baseline_matrix import EXPORT_FILES, digest, metrics, summarize


def write_json(path, obj):
    path.write_text(json.dumps(obj) + '\n')


def write_rows(path, scores, truths, kmeans_output=0):
    rows = [dict(node_id=i, gt=gt, score=score, prediction=kmeans_output,
                 score_direction='higher_is_anomalous', decision_rule='max_val_loss',
                 threshold=None, used_kmeans=True)
            for i, (score, gt) in enumerate(zip(scores, truths))]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return rows


class OperatingPointTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkpoint = 'a' * 64
        self.manifest = dict(model='orthrus', dataset='CADETS_E3', task_id='training-0', protocol=dict(
            train_dates=['2018-04-02'], val_dates=['2018-04-10'], test_dates=['2018-04-12']))
        self.scores = [9., 8., 2., 2., 1., 0.]

    def cell(self, name, truths):
        directory = self.root / name
        directory.mkdir()
        rows = write_rows(directory / 'objects.jsonl', self.scores, truths)
        write_json(directory / 'metrics.json', metrics(rows))
        write_json(directory / 'checkpoint.json', {'sha256': self.checkpoint})
        write_json(directory / 'uuid_mapping.json', {})
        write_json(directory / 'missing_gt.json', [])
        write_json(directory / 'export_complete.json', {
            'files': {f: digest(directory / f) for f in EXPORT_FILES}})
        return directory

    def calibration(self, dates):
        objects = self.root / 'dev.jsonl'
        write_rows(objects, [10., 8., 6., 4., 2., 0.], [1, 1, 1, 0, 0, 0])
        p = self.root / 'dev_manifest.json'
        write_json(p, dict(model='orthrus', dataset='CADETS_E3', checkpoint_sha256=self.checkpoint,
                           split='calibration', dates=dates, objects=objects.name,
                           objects_sha256=digest(objects)))
        return p

    def test_test_labels_do_not_fit_development_threshold(self):
        dev = self.calibration(['2018-04-09'])
        thresholds = []
        for name, labels in [('first', [1, 1, 0, 0, 0, 0]), ('second', [0, 0, 1, 1, 1, 1])]:
            cell = self.cell(name, labels)
            native = {p.name: p.read_bytes() for p in cell.iterdir()}
            report = publish_knee(cell, self.manifest, dev)
            self.assertEqual(report['scope'], 'independent_calibration')
            thresholds.append(report['policy']['threshold'])
            decisions = [json.loads(s) for s in (cell / 'knee/objects.jsonl').read_text().splitlines()]
            self.assertEqual([r['prediction'] for r in decisions], [int(s > thresholds[-1]) for s in self.scores])
            self.assertEqual(report['metrics']['TP'], metrics(decisions)['TP'])
            decision_bytes = ''.join(f"{r['node_id']}\t{r['prediction']}\n" for r in decisions).encode()
            self.assertEqual(report['decisions_sha256'], hashlib.sha256(decision_bytes).hexdigest())
            for f, payload in native.items():
                self.assertEqual((cell / f).read_bytes(), payload, f)
            self.assertEqual(summarize(cell / 'objects.jsonl')['TP'], 0)  # native k-means labels
        self.assertEqual(*thresholds)

    def test_missing_development_labels_are_not_an_independent_test(self):
        cell = self.cell('exploration', [1, 1, 0, 0, 0, 0])
        result = publish_knee(cell, self.manifest)
        self.assertEqual(result['scope'], 'retrospective_test')
        self.assertIsNone(result.get('metrics'))
        self.assertTrue(result['exploratory_metrics'])
        self.assertTrue((cell / 'knee/report.json').is_file())

    def test_overlap_with_training_or_test_is_rejected(self):
        cell = self.cell('overlap', [1, 1, 0, 0, 0, 0])
        for date in ['2018-04-02', '2018-04-12']:
            with self.subTest(date=date):
                dev = self.calibration([date])
                with self.assertRaises(ValueError):
                    publish_knee(cell, self.manifest, dev)
                self.assertFalse((cell / 'knee').exists())

    def test_replay_uses_frozen_threshold_without_reopening_calibration(self):
        dev = self.calibration(['2018-04-09'])
        truths = [0, 1, 1, 0, 0, 0]
        original = publish_knee(self.cell('original', truths), self.manifest, dev)
        dev.unlink()
        replay = publish_knee(self.cell('replay', truths), self.manifest, dev, frozen=original)
        for key in ('scope', 'policy', 'metrics', 'decisions_sha256'):
            self.assertEqual(replay[key], original[key])
        changed = dict(self.manifest, task_id='other-run')
        with self.assertRaises(ValueError):
            publish_knee(self.cell('wrong_model', truths), changed, frozen=original)

    def test_copied_test_scores_cannot_be_declared_development(self):
        cell = self.cell('leak', [1, 1, 0, 0, 0, 0])
        dev = self.calibration(['2018-04-09'])
        report = json.loads(dev.read_text())
        copied = dev.parent / report['objects']
        copied.write_bytes((cell / 'objects.jsonl').read_bytes())
        report['objects_sha256'] = digest(copied)
        write_json(dev, report)
        with self.assertRaises(ValueError):
            publish_knee(cell, self.manifest, dev)
        self.assertFalse((cell / 'knee').exists())

    def test_no_positive_label_has_no_actionable_threshold(self):
        cell = self.cell('benign', [0] * len(self.scores))
        result = publish_knee(cell, self.manifest)
        self.assertIsNone(result['policy'].get('threshold'))
        self.assertFalse((cell / 'knee/objects.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
