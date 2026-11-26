"""Build scene-isolated graphs for local multi-source APT records."""
import json
import os
from pathlib import Path

from pidsmaker.preprocessing.build_graph_methods import build_default_graphs, build_magic_graphs
from pidsmaker.utils.utils import init_database_connection


def validate_input(cfg):
    manifest = json.loads(Path(os.environ["PIDSMAKER_APT_MANIFEST"]).read_text())
    cur, con = init_database_connection(cfg)
    try:
        cur.execute("SELECT payload FROM apt_manifest")
        if cur.fetchone()[0] != manifest:
            raise ValueError("APT manifest does not match the selected database")
    finally:
        con.close()

    # PIDSMaker consumes days. Require each qualified day to belong to its scene's
    # official board and split, even if another scene ran on the same calendar day.
    owners = {s["scene_id"]: (s["board"], s["split"]) for s in manifest["scenes"]}
    seen = set()
    for split in ("train", "val", "test"):
        for day in cfg.dataset[split + "_dates"]:
            scene = int(day.split("@")[1])
            if day in seen or owners[scene] != ("B" if split == "test" else "A", split):
                raise ValueError(f"Invalid APT scene partition: {split} {day}")
            seen.add(day)


def main(cfg):
    validate_input(cfg)
    if cfg.construction.used_method not in {"default", "magic"}:
        raise ValueError(f"Unsupported APT graph method: {cfg.construction.used_method}")
    build_default_graphs.main(cfg)
    if cfg.construction.used_method == "magic":
        build_magic_graphs.main(cfg)
