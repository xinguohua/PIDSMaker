"""APT board record decoding and projection shared by import and compatibility checks."""
from datetime import datetime, timezone
from decimal import Decimal
import json
import re
from xml.etree import ElementTree

from pidsmaker.config.schemas import APT_OPERATIONS


def raw_records(path):
    """Yield exactly one event for each official 1-based file ordinal."""
    if path.suffix == ".xml":
        context = ElementTree.iterparse(path, events=("start", "end"))
        _, root = next(context)
        for event, element in context:
            if event != "end" or element.tag.rsplit("}", 1)[-1] != "Event":
                continue
            row = {}
            for system in element.findall("{*}System/*"):
                key = system.tag.rsplit("}", 1)[-1]
                if key == "TimeCreated":
                    row["@timestamp"] = system.get("SystemTime", "")
                elif key == "Computer":
                    row["Hostname"] = system.text or ""
                elif key == "Provider":
                    row["SourceName"] = system.get("Name", "")
                else:
                    row[key] = system.text or dict(system.attrib)
            for field in element.findall("{*}EventData/*"):
                row[field.get("Name") or field.tag.rsplit("}", 1)[-1]] = field.text or ""
            for section in element:
                if section.tag.rsplit("}", 1)[-1] not in {"System", "EventData"}:
                    for field in section.iter():
                        if field is not section and field.text and field.text.strip():
                            row[field.tag.rsplit("}", 1)[-1]] = field.text.strip()
            if str(row.get("EventID", "")).isdigit():
                row["EventID"] = int(row["EventID"])
            yield row
            root.clear()
        return
    with path.open(encoding="utf-8-sig") as source:
        if path.name.endswith(".bash_history"):
            ts, command = None, []
            for line in source:
                if re.fullmatch(r"#\d+\s*", line):
                    if ts is not None:
                        yield {"ts": ts, "command_line": "".join(command).rstrip("\n")}
                    ts, command = int(line[1:]), []
                elif ts is not None:
                    command.append(line)
                elif line.strip():
                    raise ValueError(f"Missing history timestamp in {path}")
            if ts is not None:
                yield {"ts": ts, "command_line": "".join(command).rstrip("\n")}
        elif path.suffix == ".json":
            for line in source:
                yield json.loads(line)
        elif path.suffix == ".log":
            for line in source:
                yield {"message": line.rstrip("\r\n")}
        else:
            raise ValueError(f"Unknown APT source format: {path}")


def indexed_locations(path):
    locations = {}
    with path.open() as source:
        for pid, entry in json.load(source).items():
            filename, ordinal = entry["filepath"], entry["linenumber"]
            if not filename.startswith("data/") or ".." in filename.split("/"):
                raise ValueError(f"Invalid source path: {filename}")
            rows = locations.setdefault(filename, {})
            if type(ordinal) is not int or ordinal < 1 or ordinal in rows:
                raise ValueError(f"Invalid record position: {filename} {ordinal}")
            rows[ordinal] = pid
    return locations


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


def timestamp_ns(r):
    if "timestamp_ms" in r:
        return int(r["timestamp_ms"]) * 1_000_000
    if "ts" in r:
        return int(Decimal(str(r["ts"])) * 1_000_000_000)
    text = r.get("@timestamp") or r.get("UtcTime") or r.get("timestamp")
    if not text:
        msg = r.get("message", "")
        iso = re.search(r"\d{4}-\d\d-\d\dT[\d:.]+(?:Z|[+-]\d\d:\d\d)", msg)
        if iso:
            text = iso.group()
        else:
            web = re.search(r"\[(\d\d/\w+/\d{4}:\d\d:\d\d:\d\d [+-]\d{4})\]", msg)
            if not web:
                raise ValueError("Missing record timestamp")
            return int(datetime.strptime(web[1], "%d/%b/%Y:%H:%M:%S %z").timestamp()) * 1_000_000_000
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    dt = dt.replace(tzinfo=dt.tzinfo or timezone.utc, microsecond=0)
    fraction = re.search(r":\d\d\.(\d+)", text)
    nanos = int(fraction[1][:9].ljust(9, "0")) if fraction else 0
    return int(dt.timestamp()) * 1_000_000_000 + nanos
