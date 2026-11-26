from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pidsmaker.config.schemas import NodeTypeSchema
from pidsmaker.preprocessing.build_graph_methods import build_default_graphs, build_magic_graphs
from pidsmaker.utils import dataset_utils

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize("builder", ["default", "magic"])
@pytest.mark.parametrize("features, expected", [
    (["type", "path", "cmd_line"], ["subject cat", "subject sh", "subject /tmp/None 0", "subject"]),
    (["path"], ["", "", "/tmp/None", ""]),
])
def test_missing_node_attributes(monkeypatch, builder, features, expected):
    module = build_default_graphs if builder == "default" else build_magic_graphs
    nt = SimpleNamespace(name="subject", table="subject_node_table",
                         columns=[("path", "path"), ("cmd_line", "cmd")])
    schema = SimpleNamespace(node_types=[nt])
    monkeypatch.setattr(dataset_utils, "get_schema", lambda cfg: schema)
    monkeypatch.setattr(module, "get_node_feats_from_cfg", lambda cfg: {"subject": features})
    db = MagicMock()
    db.cursor.return_value.fetchmany.side_effect = [[
        ("a", "A", 1, None, "cat"), ("b", "B", 2, "none", "sh"),
        ("c", "C", 3, "/tmp/None", 0), ("d", "D", 4, None, None),
    ], []]
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="test"), construction=SimpleNamespace(
        _first_file_paths="", use_hashed_label=False, null_label_tokens=False))
    if builder == "default":
        monkeypatch.setattr(module, "init_database_connection", lambda cfg: (None, db))
        monkeypatch.setattr(module, "get_schema", lambda cfg: schema)
        monkeypatch.setattr(module, "log", lambda message: None)
        cleaner = MagicMock()
        cleaner.clean_attributes.side_effect = lambda kind, attrs: attrs
        cleaner.normalize_label.side_effect = lambda label: label
        labels = module.compute_indexid2msg(cfg, cleaner)
        actual = [labels[str(i)][1] for i in range(1, 5)]
    else:
        _, _, labels, _ = module.get_node_list(SimpleNamespace(connection=db), cfg)
        actual = [labels[key] for key in "abcd"]
    assert actual == expected


def test_all_missing_auto_attributes_stay_empty():
    nt = NodeTypeSchema("subject", 1, "subjects", (("path", "path"), ("cmd", "cmd")),
                        auto_label=("cmd", "path"), null_token="[NO_CMD]")
    assert nt.auto_label_str({"type": "subject", "cmd": "", "path": ""}, True) == ""
