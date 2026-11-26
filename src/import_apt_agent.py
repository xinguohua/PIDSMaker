"""Import the local apt-agent A/B corpus. PG* variables select a fresh local database."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import psycopg2

from pidsmaker.config.schemas import SCHEMAS
from pidsmaker.preprocessing import apt_parser as reader
from pidsmaker.utils.dataset_utils import create_schema_tables, event_table_columns
from pidsmaker.utils.schema_loading import copy_rows

DAY = 86_400_000_000_000
EPOCH = date(1970, 1, 1)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(args):
    schema = SCHEMAS["apt_agent"]
    cfg = dict(raw_dir="", database=args.database, database_all_file=args.database, schema=schema.name,
               timezone="UTC", unused_dates=[], attack_to_time_window=[], ground_truth_relative_path=[])
    scenes = []
    for board, folder, expected in (("A", "train", 3), ("B", "b_official", 21)):
        group = sorted((args.root / "data" / folder).glob("*/Label_Index.json"))
        if len(group) != expected:
            raise ValueError(f"Expected {expected} {board} scenes, found {len(group)}")
        scenes.extend((board, p.parent) for p in group)
    if args.val_scene not in [s.name for b,s in scenes if b == "A"]:
        raise ValueError("Validation scene must be on A")
    report = dict(dataset=cfg, scenes=[], reader_sha256=digest(Path(reader.__file__)))
    dates = defaultdict(list)
    next_node = 0
    next_file = 0
    with psycopg2.connect(dbname=cfg["database"]) as con, con.cursor() as cur:
        cur.execute("SELECT to_regclass('event_table')")
        if cur.fetchone()[0]:
            raise ValueError("Use a fresh database; existing events would overlap")
        create_schema_tables(schema, cur)
        cur.execute("ALTER TABLE event_table ADD scene_id smallint NOT NULL, ADD file_id integer NOT NULL, ADD ordinal integer NOT NULL")
        cur.execute("CREATE TABLE apt_files(file_id integer PRIMARY KEY, scene_id smallint, board text, scene text, filepath text, sha256 text)")
        cols = event_table_columns(schema) + ["scene_id", "file_id", "ordinal"]
        for scene_id, (board, scene) in enumerate(scenes, 1):
            split = "test" if board == "B" else ("val" if scene.name == args.val_scene else "train")
            locations = reader.indexed_locations(scene / "Label_Index.json")
            files = {str(p.relative_to(scene)) for p in (scene/"data").rglob("*") if p.is_file() and not p.name.startswith(".") and p.stat().st_size}
            if files != locations.keys():
                raise ValueError(f"Index/file mismatch {scene}: {files ^ locations.keys()}")
            gold = set(json.loads((scene / "Label.json").read_text())["physical_id_set"])
            ids = {}
            nodes = defaultdict(list)
            counts = Counter()
            days = set()
            positives = 0
            for name, lookup in locations.items():
                path = (scene / name).resolve()
                path.relative_to((scene / "data").resolve())
                host = Path(name).parts[1].casefold()
                next_file += 1
                cur.execute("INSERT INTO apt_files VALUES (%s,%s,%s,%s,%s,%s)", (next_file, scene_id, board, scene.name, name, digest(path)))
                batch = []
                for ordinal, row in enumerate(reader.raw_records(path), 1):
                    pid = lookup.pop(ordinal)
                    label = int(pid in gold)
                    positives += label
                    ns = reader.timestamp_ns(row)
                    days.add(ns // DAY)
                    a, b, op = reader.project(row, path.name, host)
                    if op not in schema.rel2id:
                        raise ValueError(f"Unknown operation {op}")
                    ends = []
                    for kind, key, feature in (a, b):
                        node = (kind, key)
                        if node not in ids:
                            next_node += 1
                            ids[node] = str(next_node)
                            nodes[kind].append((str(next_node), key, next_node, feature.replace("\x00", "")))
                        ends.append(ids[node])
                    src, dst = ends
                    batch.append((src, src, op, dst, dst, pid, ns, label, scene_id, next_file, ordinal))
                    counts["bash_history" if path.name.endswith(".bash_history") else path.name] += 1
                    if len(batch) == 10000:
                        copy_rows(cur, schema.event_table, cols, batch)
                        batch.clear()
                copy_rows(cur, schema.event_table, cols, batch)
                if lookup:
                    raise ValueError(f"Missing {len(lookup)} indexed records in {name}")
            if positives != len(gold):
                raise ValueError(f"Unresolved labels in {scene}")
            for kind, rows in nodes.items():
                copy_rows(cur, schema.node_type(kind).table, ["node_uuid","hash_id","index_id","name"], rows)
            dates[split].extend(f"{EPOCH + timedelta(days=d)}@{scene_id:02d}" for d in sorted(days))
            item = dict(board=board, scene=scene.name, scene_id=scene_id, split=split, files=len(files), records=sum(counts.values()), positives=positives, sources=counts, nodes=len(ids), index_sha256=digest(scene/"Label_Index.json"), labels_sha256=digest(scene/"Label.json"))
            report["scenes"].append(item)
            print(board, scene_id, item["records"], flush=True)
        cur.execute("CREATE UNIQUE INDEX apt_event_id ON event_table(scene_id,event_uuid)")
        cur.execute("CREATE INDEX apt_event_time ON event_table(scene_id,timestamp_rec,event_uuid)")
        cur.execute("CREATE INDEX apt_event_gold ON event_table(scene_id,timestamp_rec) WHERE label=1")
        cur.execute("ANALYZE event_table")
        cur.execute("SELECT scene_id,count(*),sum(label) FROM event_table GROUP BY scene_id ORDER BY scene_id")
        assert cur.fetchall() == [(s["scene_id"],s["records"],s["positives"]) for s in report["scenes"]]
        for split in ("train","val","test"):
            cfg[f"{split}_dates"] = dates[split]
        days = sorted(d.split("@")[0] for group in dates.values() for d in group)
        cfg.update(start_date=days[0],end_date=str(date.fromisoformat(days[-1])+timedelta(days=1)))
        # Keep a recoverable manifest in the same transaction as its data.
        cur.execute("CREATE TABLE apt_manifest(payload jsonb NOT NULL)")
        cur.execute("INSERT INTO apt_manifest VALUES (%s)", (json.dumps(report),))
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--database", default="apt_agent_ab")
    parser.add_argument("--val-scene", default="uat4356")
    load(parser.parse_args())
