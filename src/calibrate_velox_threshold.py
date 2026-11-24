"""Apply a validation-node alert budget to a verified Velox replay."""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
import os
from pathlib import Path

import pandas as pd
import yaml


def metrics(counts):
    tp, fp, tn, fn = (counts[k] for k in ('TP', 'FP', 'TN', 'FN'))
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, recall=tp/(tp+fn),
                precision=tp/(tp+fp) if tp+fp else None, fpr=fp/(fp+tn),
                f1=2*tp/(2*tp+fp+fn))


def run(replay, rate, out):
    summary = json.loads((replay/'summary.json').read_text())
    assert summary['full_accepted'] and summary['checks']['scores_replayed']
    cell = Path(summary['source_cell'])
    manifest = json.loads((cell/'manifest.json').read_text())
    assert manifest['model'] == 'velox'
    cfg = yaml.safe_load((cell/'resolved_config.yml').read_text())
    settings = cfg['evaluation']['node_evaluation']
    assert settings['threshold_method'] == 'max_val_loss' and not settings['use_kmeans']
    dates = cfg['dataset']['val_dates']
    assert set(dates).isdisjoint(cfg['dataset']['test_dates'])
    assert all(interval[1][:10] not in dates and interval[2][:10] not in dates
               for interval in manifest['protocol']['attack_to_time_window'])
    epoch = 'model_epoch_' + str(cfg['training']['num_epochs']-1)
    files = sorted((replay/'replay_losses/val'/epoch).glob('*.csv'))
    reference = summary['coverage']['prediction_identity']['val']
    assert files and len(files) == reference['batches']
    endpoints = ['srcnode', 'dstnode'] if settings['use_dst_node_loss'] else ['srcnode']
    peaks = {}
    n_edges = 0
    for path in files:
        assert path.name[:10] in dates
        frame = pd.read_csv(path, usecols=endpoints+['loss'])
        for row in frame[endpoints+['loss']].itertuples(index=False,name=None):
            score = row[-1]
            if not math.isfinite(score):
                raise ValueError('Nonfinite validation loss')
            for node in row[:-1]:
                if node not in peaks or score > peaks[node]:
                    peaks[node] = score
        n_edges += len(frame)
    assert n_edges == reference['rows']
    scores = sorted(peaks.values())
    n = len(scores)
    if not 0 < rate < 1 or rate*n < 1:
        raise ValueError('Need at least one validation alert in the requested budget')
    allowed = int(rate*n)
    threshold = scores[n-allowed-1]
    val_alerts = sum(x > threshold for x in scores)
    assert val_alerts <= allowed
    selected = Counter()
    original = Counter()
    with (replay/'corrected_export/objects.jsonl').open() as records:
        for line in records:
            row = json.loads(line)
            score, label = row['score'], row['gt']
            assert label in (0,1) and math.isfinite(score)
            old = score > row['threshold']
            assert row['prediction'] == old and math.isclose(row['threshold'],scores[-1],rel_tol=1e-12)
            for counter, prediction in ((selected, score > threshold), (original, old)):
                counter[(('TN','FP'),('FN','TP'))[label][prediction]] += 1
    assert all(original[k]==summary['metrics_recomputed'][k] for k in ('TP','FP','TN','FN'))
    result = dict(method='validation_node_alert_budget', rate=str(rate), threshold=threshold,
                  old_threshold=scores[-1], comparison='strictly greater than', input_run=manifest['task_id'],
                  calibration_dates=dates, test_dates=cfg['dataset']['test_dates'], val_nodes=n,
                  val_edges=n_edges, val_max_alerts=allowed, val_alerts=val_alerts,
                  original_test=metrics(original), calibrated_test=metrics(selected))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay',type=Path)
    parser.add_argument('--rate',type=Fraction,default=Fraction(1,1000))
    args = parser.parse_args()
    run(args.replay,args.rate,Path(os.environ['RUN_OUTPUT_DIR']))
