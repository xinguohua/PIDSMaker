"""Shared dataset event indexes and run-specific node predictions for provenance analysis."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path


def component(value):
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Expected a single path component: {value!r}")
    return value


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


@contextmanager
def publish(target):
    """Serialize publication by key; a reader sees either the old state or a complete file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            yield None
            return
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            staged = Path(tmp) / target.name
            yield staged
            staged.chmod(0o444)
            os.replace(staged, target)


def index_path(root, dataset, version):
    return root / component(dataset) / component(version) / "events.sqlite"


def build_events(target, dataset, server_config):
    from baseline_matrix import protocol
    from server_credentials import database_options
    import psycopg2

    database = protocol()[dataset]["database"]
    tables = [
        ("subject", "subject_node_table", "path, cmd"),
        ("file", "file_node_table", "path, NULL"),
        (
            "netflow",
            "netflow_node_table",
            "concat(src_addr,':',src_port,' -> ',dst_addr,':',dst_port), NULL",
        ),
    ]
    with publish(target) as staged:
        if staged is None:
            return
        with (
            closing(
                psycopg2.connect(
                    dbname=database,
                    **database_options(json.loads(server_config.read_text())),
                )
            ) as source,
            closing(sqlite3.connect(staged)) as dest,
        ):
            source.set_session(readonly=True, isolation_level="REPEATABLE READ")
            dest.executescript(
                "create table nodes(id integer primary key,kind text,path text,cmd text,uuid text);"
                "create table events(rid integer primary key,src integer,dst integer,op text,ns integer,uuid text);"
            )
            for kind, table, fields in tables:
                with source.cursor(name=kind) as cursor:
                    cursor.itersize = 10000
                    cursor.execute(
                        f"select index_id,'{kind}',{fields},node_uuid from public.{table} where index_id is not null"
                    )
                    dest.executemany("insert into nodes values(?,?,?,?,?)", cursor)
            with source.cursor(name="events") as cursor:
                cursor.itersize = 10000
                cursor.execute(
                    "select _id,src_index_id,dst_index_id,substr(operation,7),timestamp_rec,event_uuid from public.event_table"
                )
                dest.executemany("insert into events values(?,?,?,?,?,?)", cursor)
            dest.execute("create index src_event on events(src,op)")
            dest.execute("create index dst_event on events(dst,op)")
            dest.execute(
                "create table dataset_identity(dataset text, database_name text)"
            )
            dest.execute(
                "insert into dataset_identity values(?,?)", (dataset, database)
            )
            dest.commit()


def adopt_frozen_index(target, source, manifest_path, dataset):
    """Publish an already completed, quiescent snapshot on the same filesystem."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"]["database"].lower() != dataset.lower():
        raise ValueError("Snapshot database and dataset disagree")
    with publish(target) as staged:
        if staged is None:
            if not target.samefile(source):
                raise ValueError(
                    "Dataset revision already published from another snapshot"
                )
            return
        wal = Path(str(source) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError(
                "Finish snapshot writes and checkpoint its WAL before adoption"
            )
        if source.stat().st_size != manifest["snapshot_bytes"]:
            raise ValueError("Snapshot size disagrees with completed manifest")
        with closing(readonly(source)) as db:
            indexes = {r[1] for r in db.execute("pragma index_list(events)")}
            if not {"src_event", "dst_event"} <= indexes:
                raise ValueError("Missing endpoint indexes")
            if (
                db.execute("select count(*) from events").fetchone()[0]
                != manifest["raw_rows"]
            ):
                raise ValueError("Incomplete event snapshot")
        # A hard link keeps the dataset available independently of the importing project.
        os.link(source, staged)


def publish_node_predictions(events, target, export):
    """Only publish records whose node IDs and UUIDs match this dataset version."""
    with publish(target) as staged:
        manifest = json.loads((export / "export_complete.json").read_text())
        expected = manifest["files"]["objects.jsonl"]
        if staged is None:
            with closing(readonly(target)) as db:
                actual = db.execute("select object_sha256 from provenance").fetchone()[
                    0
                ]
            if actual != expected:
                raise ValueError(
                    "This run key already identifies another prediction export"
                )
            return
        digest = hashlib.sha256()
        with closing(sqlite3.connect(staged.resolve().as_uri(), uri=True)) as db:
            db.execute(
                "create table pred(id integer primary key, uuid text not null, gt integer, prediction integer, score real, win integer)"
            )
            db.execute("create table provenance(object_sha256 text not null)")

            def rows():
                with (export / "objects.jsonl").open("rb") as f:
                    for line in f:
                        digest.update(line)
                        r = json.loads(line)
                        yield (
                            r["node_id"],
                            r["uuid"],
                            r["gt"],
                            r["prediction"],
                            r["score"],
                            r.get("max_loss_window"),
                        )

            db.executemany("insert into pred values(?,?,?,?,?,?)", rows())
            if digest.hexdigest() != expected:
                raise ValueError("Prediction file does not match its completed export")
            db.execute(
                "attach database ? as dataset",
                (events.resolve().as_uri() + "?immutable=1&mode=ro",),
            )
            bad = db.execute(
                "select p.id from pred p left join dataset.nodes n on n.id=p.id where n.uuid is not p.uuid limit 1"
            ).fetchone()
            if bad:
                raise ValueError(
                    f"Prediction node {bad[0]} does not match dataset UUID"
                )
            db.execute("insert into provenance values(?)", (expected,))
            db.commit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", required=True, help="Immutable source dataset revision")
    sub = p.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-events")
    build.add_argument("--server-config", type=Path, required=True)
    adopt = sub.add_parser("adopt-frozen-events")
    adopt.add_argument("--source", type=Path, required=True)
    adopt.add_argument("--manifest", type=Path, required=True)
    register = sub.add_parser("node-predictions")
    register.add_argument("--run", required=True)
    register.add_argument("--export", type=Path, required=True)
    query = sub.add_parser("incident")
    query.add_argument("node", type=int)
    query.add_argument("--op")
    a = p.parse_args()
    events = index_path(a.root, a.dataset, a.version)
    if a.action == "build-events":
        build_events(events, a.dataset, a.server_config)
    elif a.action == "adopt-frozen-events":
        adopt_frozen_index(events, a.source, a.manifest, a.dataset)
    elif a.action == "node-predictions":
        publish_node_predictions(
            events,
            events.parent / "runs" / component(a.run) / "predictions.sqlite",
            a.export,
        )
    else:
        with closing(readonly(events)) as db:
            db.row_factory = sqlite3.Row
            where = "(src=? or dst=?)"
            params = [a.node, a.node]
            if a.op:
                where += " and op=?"
                params.append(a.op)
            for row in db.execute(
                "select * from events where " + where + " order by ns,rid", params
            ):
                print(json.dumps(dict(row), ensure_ascii=False))


if __name__ == "__main__":
    main()
