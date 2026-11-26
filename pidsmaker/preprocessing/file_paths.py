"""First observed training paths for otherwise unnamed static file nodes."""
import hashlib
import json
from pathlib import Path


def read_evidence(cfg):
    path = getattr(cfg.construction, 'file_path_evidence', None)
    if not path:
        return {}
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != Path(path).stem:
        raise ValueError('File path evidence filename must be its SHA-256 digest')
    evidence = json.loads(raw)
    if (evidence['version'] != 1 or evidence['dataset'] != cfg.dataset.name
            or sorted(evidence['train_dates']) != sorted(cfg.dataset.train_dates)):
        raise ValueError('File path evidence belongs to another protocol')
    return evidence['paths']


def effective_file_path(uuid, original, evidence):
    if original is None or str(original).strip().lower() in ('', 'none', 'null'):
        return evidence.get(uuid, original)
    return original
