"""First observed training paths for otherwise unnamed static file nodes."""
import hashlib
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path


def evidence_location(dataset, dates):
    root = Path(os.environ.get('XDG_DATA_HOME') or Path.home()/'.local/share')
    split = hashlib.sha256('\n'.join(sorted(dates)).encode()).hexdigest()
    return root/'pidsmaker'/'first_file_paths'/dataset/split/'current.json'


def install_evidence(path, dataset, dates):
    link = evidence_location(dataset, dates)
    link.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=link.parent) as temp:
        staged = Path(temp)/'current.json'
        staged.symlink_to(path.resolve(strict=True))
        os.replace(staged, link)


def evidence_path(cfg):
    link = evidence_location(cfg.dataset.name, cfg.dataset.train_dates)
    if link.is_symlink() or link.exists():
        return link.resolve(strict=True)
    return None


def read_evidence(cfg):
    bound = getattr(getattr(cfg, "construction", None), "_first_file_paths", None)
    path = evidence_path(cfg) if bound is None else (Path(bound) if bound else None)
    if path is None:
        if cfg.dataset.name == 'CADETS_E3':
            raise FileNotFoundError('Missing first-use path evidence for this CADETS_E3 training split; prepare the dataset with first_file_paths.py')
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
