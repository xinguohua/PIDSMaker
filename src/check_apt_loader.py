"""Run a local APT detector's real preprocessing and read its train, val and test batches."""
import json
from pathlib import Path

import wandb

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.main import main
from pidsmaker.tasks.batching import get_preprocessed_graphs


if __name__ == "__main__":
    args, unknown = get_runtime_required_args(return_unknown_args=True)
    if unknown or args.dataset != "APT_AGENT_AB" or not args.test_mode:
        raise ValueError("Expected a local APT_AGENT_AB --test_mode compatibility check")
    args.database_password = None  # libpq reads PGPASSWORD / .pgpass
    wandb.init(mode="disabled")
    cfg = get_yml_cfg(args)
    main(cfg, stop_after="feat_inference")
    train, val, test, _ = get_preprocessed_graphs(cfg)
    report = {}
    for split, groups in (("train", train), ("val", val), ("test", test)):
        batches = [b for group in groups for b in group]
        edges = sum(b.src.numel() for b in batches)
        if not batches or not edges:
            raise ValueError(f"{args.model} produced no {split} input")
        report[split] = dict(batches=len(batches), events=edges)
    result = dict(model=args.model, manifest=cfg.dataset.manifest_sha256, splits=report,
                  scope="native test_mode: first active day/window of each split")
    (Path(args.artifact_dir) / f"{args.model}_loader.json").write_text(json.dumps(result, indent=2))
    print(result)
