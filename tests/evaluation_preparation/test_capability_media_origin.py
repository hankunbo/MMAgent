"""Capability successors display original assets without granting execution."""

import copy
import json

import pytest

from packages.evaluation_preparation.evidence_media import acquisition_root, checked_file
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.evaluation_preparation.test_evidence_tree_v2 import PNG
from tests.evaluation_preparation.test_generic_definition_media import setup


def fixture(monkeypatch, tmp_path, depth=2):
    parent, _, report = setup(monkeypatch, tmp_path)
    history = report.pop("definition_revision")["original_history"]
    binding = json.loads((parent / "binding.json").read_bytes())
    nodes = []
    for index in range(depth + 1):
        key = "child" if index == depth else f"predecessor-{index}"
        folder = tmp_path / key
        folder.mkdir(exist_ok=True)
        value = dict(
            schema_version="generic-capability-successor/" + ("1.1.0" if index == 0 else "1.3.0"),
            key=key,
            parent_key="parent",
            original_history=copy.deepcopy(history),
        )
        if nodes:
            previous = nodes[-1]
            value["continuation"] = dict(
                predecessor_key=previous["key"],
                root_claim=copy.deepcopy(nodes[0]),
                inventory=dict(selection_checksum=digest(previous)),
            )
            (tmp_path / previous["key"] / "capability-continuation-successor.json").write_text(
                json.dumps(value)
            )
        else:
            (parent / "definition-review-successor.json").write_text(json.dumps(value))
        (folder / "input.json").write_text(json.dumps(value))
        (folder / "binding.json").write_text(json.dumps(binding))
        nodes.append(value)
    report["capability_revision"] = dict(
        schema_version="generic-capability-increment-report-scope/1.0.0",
        selection=copy.deepcopy(nodes[-1]),
    )
    return parent, report


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_capability_history_binds_native_masks_to_acquisition(monkeypatch, tmp_path, depth):
    parent, report = fixture(monkeypatch, tmp_path, depth)
    before = copy.deepcopy(report)
    root, _ = acquisition_root(report, "child")
    assert root == parent and report == before
    mask = parent / "native.png"
    mask.write_bytes(PNG)
    assert checked_file(root, mask.as_uri(), checksum(PNG)) == PNG
    other = tmp_path / "unrelated.png"
    other.write_bytes(PNG)
    with pytest.raises(ValueError, match="outside_run"):
        checked_file(root, other.as_uri(), checksum(PNG))


@pytest.mark.parametrize(
    "fault",
    [
        "report",
        "root_claim",
        "child_claim",
        "predecessor",
        "binding",
        "history",
        "native",
        "case",
        "original_report",
        "original_html",
        "input",
        "cycle",
        "version",
        "escape",
    ],
)
def test_capability_media_rejects_unbound_ancestry(monkeypatch, tmp_path, fault):
    parent, report = fixture(monkeypatch, tmp_path)
    if fault == "report":
        report["capability_revision"]["selection"]["key"] = "other"
    elif fault == "history":
        report["capability_revision"]["selection"]["original_history"]["root_key"] = "other"
    elif fault == "native":
        report["native_tools"] = {"other": True}
    elif fault == "case":
        report["result"]["case_code"] = "CASE:OTHER"
    elif fault == "original_html":
        (parent / "report.html").write_text("changed")
    else:
        path = {
            "root_claim": parent / "definition-review-successor.json",
            "child_claim": tmp_path / "predecessor-1/capability-continuation-successor.json",
            "predecessor": tmp_path / "predecessor-1/input.json",
            "binding": tmp_path / "child/binding.json",
            "original_report": parent / "report.json",
            "input": parent / "input.json",
            "cycle": tmp_path / "child/input.json",
            "version": tmp_path / "child/input.json",
            "escape": tmp_path / "child/input.json",
        }[fault]
        value = json.loads(path.read_bytes())
        if fault == "cycle":
            value["continuation"]["predecessor_key"] = "child"
        elif fault == "version":
            value["schema_version"] = "generic-capability-successor/99.0.0"
        elif fault == "escape":
            value["parent_key"] = "../other"
        else:
            value["changed"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="media_capability_|generic_run_key_invalid"):
        acquisition_root(report, "child")


def test_capability_media_depth_is_bounded(monkeypatch, tmp_path):
    _, report = fixture(monkeypatch, tmp_path, depth=4)
    with pytest.raises(ValueError, match="media_capability_"):
        acquisition_root(report, "child")
