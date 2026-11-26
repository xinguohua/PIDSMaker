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
        pid_birth = {}
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
            if "security" in path.name:
                n = row.get("EventID")
                record = row.get("EventRecordID")
                for field in ("ProcessId", "ProcessID", "NewProcessId"):
                    if field not in row:
                        continue
                    value = str(row[field])
                    try:
                        pid = int(value, 16 if value.lower().startswith("0x") else 10)
                    except ValueError:
                        continue
                    if n == 4688 and field == "NewProcessId":
                        pid_birth[pid] = record
                    # A PID present before the first recorded creation is scoped
                    # to this log segment until its first exit or reuse.
                    row[field + "_instance"] = (pid, pid_birth.get(pid, "initial"))
                    if n == 4689:
                        pid_birth.pop(pid, None)
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
    def resource(category, *parts):
        values = tuple(str(x) for x in parts if x is not None and x != "")
        return entity("artifact", (host, category, *values), " ".join(values) or category)

    def socket(ip, port, protocol):
        return entity("endpoint", (ip, port, protocol), f"{ip}:{port or ''} {protocol or ''}".strip())

    def flow(src, sport, dst, dport, proto):
        return entity("endpoint", (src, sport, dst, dport, proto),
                      f"{src}:{sport} -> {dst}:{dport} {proto}")

    if filename == "ecar.json":
        p = r.get("properties", {})
        kind, action = r["object"], r["action"]
        op = f"ECAR_{kind}_{action}"
        actor, obj = r.get("actorID"), r.get("objectID")
        if kind == "FLOW" and p.get("src_ip") and p.get("dst_ip"):
            conn = flow(p["src_ip"], p.get("src_port"), p["dst_ip"], p.get("dst_port"), p.get("protocol"))
            if actor:
                return process(actor, p.get("image_path")), conn, op
            return socket(p["src_ip"], p.get("src_port"), p.get("protocol")), socket(p["dst_ip"], p.get("dst_port"), p.get("protocol")), op
        if kind == "USER_SESSION":
            return resource("account", r.get("principal")), resource("session", r.get("objectID"), p.get("logon_id"), p.get("session_id")), op
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
        n = int(r["EventID"])
        sysmon = "sysmon" in filename
        op = f"{'SYSMON' if sysmon else 'SECURITY'}_{n}"
        if op not in APT_OPERATIONS:
            raise ValueError(f"Unsupported Windows event {op}")
        if sysmon:
            src = process(r["ProcessGuid"], r.get("Image")) if r.get("ProcessGuid") else sensor
            if n in (8, 10):
                fields = {k.lower(): v for k, v in r.items()}
                return (process(fields["sourceprocessguid"], fields.get("sourceimage")),
                        process(fields["targetprocessguid"], fields.get("targetimage")), op)
            if n == 1:
                return process(r["ParentProcessGuid"], r.get("ParentImage")), src, op
            if n == 3:
                return src, flow(r["SourceIp"], r["SourcePort"], r["DestinationIp"], r["DestinationPort"], r["Protocol"]), op
            if n == 7:
                return resource("file", r["ImageLoaded"]), src, op
            if n == 5:
                return src, src, op
            key, field = {11: ("file", "TargetFilename"), 13: ("registry", "TargetObject"),
                          22: ("dns", "QueryName")}[n]
            return src, resource(key, r[field]), op

        def account(prefix):
            return resource("account", r.get(prefix + "UserSid"), r.get(prefix + "DomainName"), r.get(prefix + "UserName"))

        def process_ref(field, image):
            pid = r.get(field + "_instance", (r.get(field), r.get("EventRecordID")))
            return entity("process", (host, "security", pid), image or "process")

        if n == 5156:
            return process_ref("ProcessID", r["Application"]), flow(
                r["SourceAddress"], r["SourcePort"], r["DestAddress"], r["DestPort"], r["Protocol"]), op
        if n == 4688:
            return process_ref("ProcessId", r.get("ParentProcessName")), process_ref("NewProcessId", r["NewProcessName"]), op
        if n == 4689:
            return account("Subject"), process_ref("ProcessId", r["ProcessName"]), op
        if n in (4768, 4769, 4770, 4771):
            return account("Target"), resource("service", r.get("ServiceName")), op
        if n == 4698:
            return account("Subject"), resource("task", r.get("TaskName")), op
        if n == 4672:
            return account("Subject"), resource("session", r.get("SubjectLogonId")), op
        if n in (4634, 4800, 4801):
            return account("Target"), resource("session", r.get("TargetLogonId")), op
        if n == 4776:
            return entity("host", r["Workstation"].casefold()), account("Target"), op
        return account("Subject"), account("Target"), op

    if filename.endswith(".json"):
        op = "ZEEK_" + filename.split(".")[0].upper()
        if filename == "dhcp.json":
            return resource("client", r.get("mac"), r.get("client_addr")), resource("lease", r.get("assigned_addr"), r.get("server_addr")), op
        if r.get("id.orig_h") and r.get("id.resp_h"):
            src = socket(r["id.orig_h"], r.get("id.orig_p"), r.get("proto"))
            dst = socket(r["id.resp_h"], r.get("id.resp_p"), r.get("proto"))
            if filename == "dns.json" and r.get("query"):
                dst = resource("dns", dst[2], r["query"], r.get("qtype_name"))
            if filename == "http.json":
                dst = resource("http", dst[2], r.get("method"), r.get("host"), r.get("uri"))
            if filename == "ssl.json" and r.get("server_name"):
                dst = resource("tls", dst[2], r["server_name"])
            return src, dst, op
        fid = r.get("fuid") or r.get("id")
        target = entity("artifact", (host, "zeek_file", fid), r.get("filename") or r.get("certificate.subject") or r.get("machine") or filename) if fid else observed(filename)
        if r.get("tx_hosts"):
            return resource("transmitters", *r["tx_hosts"]), target, op
        return sensor, target, op

    if filename == "web_access.log":
        fields = r["message"].split('"')
        return entity("endpoint", fields[0].split()[0]), resource("web_request", fields[1]), "WEB_ACCESS"
    if filename.endswith(".bash_history"):
        return resource("account", filename.removesuffix(".bash_history")), observed(r["command_line"] or "empty command"), "SHELL_HISTORY"
    if filename == "syslog.log":
        fields = r["message"].split(maxsplit=7)
        return resource("syslog_app", fields[3]), observed(fields[-1]), "SYSLOG"
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
