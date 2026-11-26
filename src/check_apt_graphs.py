"""Construct and verify the full local A/B graphs with PIDSMaker's Velox configuration."""
import json
import os
import subprocess
from pathlib import Path

from pidsmaker.config import get_runtime_required_args, get_yml_cfg, set_task_to_done
from pidsmaker.preprocessing.build_graph_methods import build_default_graphs as builder
from pidsmaker.utils.graph_store import load_graph_columns
from pidsmaker.utils.utils import datetime_to_ns_time_US, get_all_graphs_for_dates, init_database_connection


def main():
    # These compact graph pickles are produced by the local construction in this script.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    args = get_runtime_required_args()
    args.database_password = os.environ["PGPASSWORD"]
    cfg = get_yml_cfg(args)
    if cfg.dataset.name != "APT_AGENT_AB" or cfg.construction.fuse_edge or cfg._test_mode:
        raise ValueError("Verification requires the full A/B dataset and construction.fuse_edge=False")
    cur, con = init_database_connection(cfg)
    try:
        cur.execute("SELECT payload FROM apt_manifest")
        if cur.fetchone()[0] != json.loads(Path(os.environ["PIDSMAKER_APT_MANIFEST"]).read_text()):
            raise ValueError("Selected manifest does not match the local database")
    finally:
        con.close()
    if not (Path(cfg.construction._task_path) / "done.txt").is_file():
        builder.main(cfg)
    graph_dir = cfg.construction._graphs_dir
    report = dict(dataset=cfg.dataset.name, model_config=args.model,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  manifest_sha256=cfg.dataset.manifest_sha256, time_window_minutes=cfg.construction.time_window_size,
                  fuse_edge=cfg.construction.fuse_edge, benign_training=cfg.construction.benign_training,
                  graph_dir=graph_dir, dicts_dir=cfg.construction._dicts_dir, partitions={})
    cur, con = init_database_connection(cfg)
    try:
        for split in ("train", "val", "test"):
            count = windows = positives = 0
            for day in cfg.dataset[split + "_dates"]:
                bounds = builder.day_query_params(day, lambda s: datetime_to_ns_time_US(s, timezone="UTC"))
                cur.execute("SELECT event_uuid,label,timestamp_rec,operation,src_index_id,dst_index_id FROM event_table WHERE timestamp_rec >= %s "
                            "AND timestamp_rec < %s AND scene_id=%s" + (" AND label=0" if split != "test" else ""), bounds)
                expected = {row[0]: row[1:] for row in cur}
                for path in get_all_graphs_for_dates(graph_dir, [day]):
                    graph = load_graph_columns(path)
                    uuids, labels = graph.edge_attrs["event_uuid"], graph.edge_attrs["y"]
                    edges = zip(uuids, labels, graph.edge_attrs["time"], graph.edge_attrs["label"],
                                graph.src, graph.dst, strict=True)
                    for key, label, ts, op, src, dst in edges:
                        if expected.pop(key) != (label, ts, op, graph.node_ids[src], graph.node_ids[dst]):
                            raise ValueError(f"Edge mismatch in {path}: {key}")
                    count += len(uuids)
                    positives += sum(labels)
                    windows += 1
                if expected:
                    raise ValueError(f"{day}: {len(expected)} missing records")
            report["partitions"][split] = dict(windows=windows, edges=count, positives=positives)
    finally:
        con.close()
    dest = Path(args.artifact_dir) / "graph_audit.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    set_task_to_done(cfg.construction._task_path)
    print(dest, report["partitions"])


if __name__ == "__main__":
    main()
