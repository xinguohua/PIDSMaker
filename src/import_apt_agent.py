"""Import the local apt-agent A/B corpus. PG* variables select a fresh local database."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import psycopg2

from pidsmaker.config.schemas import APT_OPERATIONS, SCHEMAS
from pidsmaker.utils.dataset_utils import create_schema_tables, event_table_columns
from pidsmaker.utils.schema_loading import copy_rows

DAY = 86_400_000_000_000
EPOCH = date(1970, 1, 1)


def entity(kind, key, label=None):
    return kind, str(key), str(label or key)


def project(r, filename, host):
    """Endpoints describe explicit entities; fallback is a host observation of an event class."""
    sensor = entity("host", host)
    def process(guid, image):
        return entity("process", (host, str(guid).strip("{}").lower()), image or "process")
    def observed(label):
        return entity("observation", (host, filename, label), label)
    def network(src, dst):
        return entity("endpoint", src), entity("endpoint", dst)

    if filename == "ecar.json":
        p = r.get("properties", {})
        kind, action = r["object"], r["action"]
        op = action if action in APT_OPERATIONS else "ECAR_OBS"
        actor, obj = r.get("actorID"), r.get("objectID")
        if kind == "FLOW" and p.get("src_ip") and p.get("dst_ip"):
            return (*network(p["src_ip"], p["dst_ip"]), op)
        create = kind == "PROCESS" and action == "CREATE"
        src = process(actor, p.get("parent_image_path" if create else "image_path")) if actor else sensor
        if kind == "PROCESS" and obj:
            dst = process(obj, p.get("target_image_path") or (p.get("image_path") if create or not actor else None))
        elif kind == "THREAD" and p.get("target_process_uuid"):
            dst = process(p["target_process_uuid"], p.get("target_image_path"))
        elif kind in {"MODULE", "FILE", "REGISTRY"} and obj:
            path = p.get("file_path") or p.get("registry_key")
            dst = entity("artifact", (host, kind, obj), path or kind)
        else:
            dst = observed(f"{kind} {action}")
        if action in {"READ", "LOAD"} and actor:
            return dst, src, op
        return src, dst, op

    if filename.endswith(".xml"):
        op = "SYSMON" if "sysmon" in filename else "SECURITY"
        if r.get("SourceIp") and r.get("DestinationIp"):
            return (*network(r["SourceIp"], r["DestinationIp"]), op)
        if r.get("ParentProcessGuid") and r.get("ProcessGuid"):
            return process(r["ParentProcessGuid"], r.get("ParentImage")), process(r["ProcessGuid"], r.get("Image")), op
        src = process(r["ProcessGuid"], r.get("Image")) if r.get("ProcessGuid") else sensor
        path = r.get("TargetFilename") or r.get("TargetObject")
        dst = entity("artifact", (host, path), path) if path else observed(f"{op} {r['EventID']}")
        return src, dst, op

    if filename.endswith(".json"):
        op = "ZEEK_" + filename.split(".")[0].upper()
        src = r.get("id.orig_h") or r.get("client_addr")
        dst = r.get("id.resp_h") or r.get("server_addr")
        if src and dst:
            return (*network(src, dst), op)
        fid = r.get("fuid") or r.get("id")
        target = entity("artifact", (host, fid), r.get("filename") or filename) if fid else observed(filename)
        return sensor, target, op

    if filename == "web_access.log":
        return entity("endpoint", r["message"].split()[0]), sensor, "WEB_ACCESS"
    if filename.endswith(".bash_history"):
        return sensor, observed(r["command_line"] or "empty command"), "SHELL_HISTORY"
    if filename == "syslog.log":
        fields = r["message"].split(maxsplit=6)
        return sensor, observed(fields[3] if len(fields) >= 4 else filename), "SYSLOG"
    raise ValueError(f"Unsupported source {filename}")


def timestamp_ns(reader, r):
    if "timestamp_ms" in r:
        return int(r["timestamp_ms"]) * 1_000_000
    if "ts" in r:
        return int(Decimal(str(r["ts"])) * 1_000_000_000)
    t = reader.timestamp(r)
    if not t:
        raise ValueError("Missing timestamp")
    value = int(Decimal(str(t)) * 1_000_000_000)
    # The XML clock may carry seven digits, beyond Python datetime precision.
    if r.get("@timestamp"):
        match = re.search(r":\d\d\.(\d+)", r["@timestamp"])
        if match:
            value = int(t) * 1_000_000_000 + int(match[1][:9].ljust(9, "0"))
    return value


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(args):
    spec = importlib.util.spec_from_file_location("apt_reader", args.root / "submission_agent/v7/data.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    schema = SCHEMAS["apt_agent"]
    cfg = dict(raw_dir="", database="apt_agent_ab", database_all_file="apt_agent_ab", schema=schema.name,
               timezone="UTC", unused_dates=[], attack_to_time_window=[], ground_truth_relative_path=[])
    scenes = []
    for board, folder, expected in (("A", "train", 3), ("B", "b_official", 21)):
        group = sorted((args.root / "data" / folder).glob("*/Label_Index.json"))
        if len(group) != expected:
            raise ValueError(f"Expected {expected} {board} scenes, found {len(group)}")
        scenes.extend((board, p.parent) for p in group)
    if args.val_scene not in [s.name for b,s in scenes if b == "A"]:
        raise ValueError("Validation scene must be on A")
    report = dict(dataset=cfg, scenes=[], reader_sha256=digest(Path(spec.origin)))
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
                path = reader.inside(scene, name)
                host = Path(name).parts[1].casefold()
                next_file += 1
                cur.execute("INSERT INTO apt_files VALUES (%s,%s,%s,%s,%s,%s)", (next_file, scene_id, board, scene.name, name, digest(path)))
                batch = []
                for ordinal, row in enumerate(reader.raw_records(path), 1):
                    pid = lookup.pop(ordinal)
                    label = int(pid in gold)
                    positives += label
                    ns = timestamp_ns(reader, row)
                    days.add(ns // DAY)
                    a, b, op = project(row, path.name, host)
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
    parser.add_argument("--val-scene", default="uat4356")
    load(parser.parse_args())
