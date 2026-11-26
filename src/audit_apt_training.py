"""Verify every Velox training edge before the optimizer sees the batches."""
from collections import Counter
import json
from pathlib import Path

from pidsmaker.config.schemas import SCHEMAS
from pidsmaker.utils.data_utils import to_original_node_ids
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import get_all_graphs_for_dates


def audit(cfg, train, output):
    rel = SCHEMAS["apt_agent"].rel2id
    expected = Counter()
    for path in get_all_graphs_for_dates(cfg.construction._graphs_dir, cfg.dataset.train_dates):
        g = load_graph_columns(path)
        assert not any(g.edge_attrs["y"]), path
        for a, b, t, op in zip(g.src, g.dst, g.edge_attrs["time"], g.edge_attrs["label"], strict=True):
            expected[int(g.node_ids[a]), int(g.node_ids[b]), int(t), rel[op]] += 1
    total = expected.total()
    counts = Counter()
    batches = 0
    for dataset in train:
        for batch in dataset:
            ids = batch.edge_type.argmax(dim=1) + 1
            for a, b, t, op in zip(to_original_node_ids(batch.src).tolist(),
                                   to_original_node_ids(batch.dst).tolist(), batch.t.tolist(),
                                   ids.tolist(), strict=True):
                key = (a, b, t, op)
                if expected[key] <= 0:
                    raise ValueError(f"Unexpected training edge {key}")
                expected[key] -= 1
                counts[rel[op]] += 1
            batches += 1
    assert total == counts.total() and not any(expected.values())
    Path(output, "train_edges.json").write_text(json.dumps(dict(
        manifest=cfg.dataset.manifest_sha256, edges=total, batches=batches, operations=counts
    ), indent=2))
