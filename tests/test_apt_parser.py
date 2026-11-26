import tempfile
import unittest
from pathlib import Path

from pidsmaker.preprocessing.apt_parser import project, raw_records, timestamp_ns


class TestAPTParser(unittest.TestCase):
    def test_official_ordinals_and_xml_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "user.bash_history"
            history.write_text("#10\ncommand a\ncontinued\n#11\ncommand b\n")
            self.assertEqual(list(raw_records(history)), [
                {"ts": 10, "command_line": "command a\ncontinued"},
                {"ts": 11, "command_line": "command b"},
            ])
            xml = Path(tmp) / "windows_event_sysmon.xml"
            event = ('<Event xmlns="urn:events"><System><EventID>11</EventID>'
                     '<TimeCreated SystemTime="1970-01-01T00:00:01.1234567Z"/>'
                     '</System><EventData><Data Name="TargetFilename">x</Data></EventData></Event>')
            xml.write_text("<Events>" + 2 * event + "</Events>")
            a, b = raw_records(xml)
            self.assertEqual(a, b)
            self.assertEqual(timestamp_ns(a), 1123456700)
            self.assertEqual(project(a, xml.name, "host")[1][2], "x")

    def test_explicit_windows_relations(self):
        for number, suffix in ((8, "Guid"), (10, "GUID")):
            row = {"EventID": number, "SourceProcess" + suffix: "{a}",
                   "TargetProcess" + suffix: "{b}", "SourceImage": "one", "TargetImage": "two"}
            src, dst, op = project(row, "windows_event_sysmon.xml", "h")
            self.assertEqual((src[0], src[2], dst[0], dst[2], op),
                             ("process", "one", "process", "two", f"SYSMON_{number}"))
        row = dict(EventID=5156, ProcessID="4", Application="System", SourceAddress="a",
                   SourcePort="123", DestAddress="b", DestPort="443", Protocol="6")
        src, dst, op = project(row, "windows_event_security.xml", "h")
        self.assertEqual((src[2], dst[2], op), ("System", "a:123 -> b:443 6", "SECURITY_5156"))

    def test_directed_read_and_host_scoped_process(self):
        event = dict(object="FILE", action="READ", actorID="p", objectID="f",
                     properties={"file_path": "/report", "image_path": "/reader"})
        file_node, process_node, relation = project(event, "ecar.json", "one")
        self.assertEqual((file_node[0], process_node[0], relation), ("artifact", "process", "ECAR_FILE_READ"))
        self.assertNotEqual(process_node[1], project(event, "ecar.json", "two")[1][1])


if __name__ == "__main__":
    unittest.main()
