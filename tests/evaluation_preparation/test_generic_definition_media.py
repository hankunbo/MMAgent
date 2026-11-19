"""Display ancestry binding, not permission to reuse historical observations."""

import copy
import json

import pytest

from packages.evaluation_preparation.evidence_media import acquisition_root, checked_file
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.evaluation_preparation.test_evidence_tree_v2 import PNG


def setup(monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_input, generic_run

    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    original_directory = generic_input.directory
    monkeypatch.setattr(generic_input, "directory", lambda k: original_directory(k, root=tmp_path))
    spec = dict(schema_version="fairvlmed-run-input/1.5.0", key="parent")
    original = dict(result=dict(case_code="CASE:TEST", complete=True), native_tools={"results": {}})
    html = "<html>immutable original</html>"
    history = dict(
        schema_version="generic-completed-definition-origin/1.0.0",
        root_key="parent",
        input_checksum=digest(spec),
        result_checksum=digest(original["result"]),
        report_checksum=digest([original, dict(html=html, report_checksum=digest(original))]),
    )
    selection = dict(
        schema_version="generic-definition-review-successor/1.0.0",
        key="child",
        parent_key="parent",
        original_history=history,
    )
    binding = dict(case_code="CASE:TEST", image={"checksum": digest("same input")})
    report = dict(
        result={"case_code": "CASE:TEST"},
        native_tools=copy.deepcopy(original["native_tools"]),
        definition_revision=dict(original_history=copy.deepcopy(history)),
    )
    for p, v in [
        (parent / "input.json", spec),
        (child / "input.json", selection),
        (parent / "definition-review-successor.json", selection),
        (parent / "binding.json", binding),
        (child / "binding.json", binding),
        (parent / "report.json", original),
    ]:
        p.write_text(json.dumps(v))
    (parent / "report.html").write_text(html)

    def export(key):
        assert key == "parent"
        return dict(json=str(parent / "report.json"), html=str(parent / "report.html"))

    monkeypatch.setattr(generic_run, "export", export)
    return parent, child, report


def test_new_interpretation_keeps_original_images_and_scores(monkeypatch, tmp_path):
    parent, child, report = setup(monkeypatch, tmp_path)
    before = copy.deepcopy(report)
    root, _ = acquisition_root(report, "child")
    assert root == parent and report == before
    mask = parent / "mask.png"
    mask.write_bytes(PNG)
    assert checked_file(root, mask.as_uri(), checksum(PNG)) == PNG
    with pytest.raises(ValueError, match="outside_run"):
        checked_file(child, mask.as_uri(), checksum(PNG))


@pytest.mark.parametrize(
    "fault",
    [
        "input",
        "claim",
        "binding",
        "history",
        "case",
        "tools",
        "result",
        "html",
        "version",
        "escape",
    ],
)
def test_mismatched_definition_ancestry_cannot_attach_images(monkeypatch, tmp_path, fault):
    parent, child, report = setup(monkeypatch, tmp_path)
    if fault == "html":
        (parent / "report.html").write_text("changed")
    elif fault == "history":
        report["definition_revision"]["original_history"]["result_checksum"] = digest("other")
    elif fault == "case":
        report["result"]["case_code"] = "CASE:OTHER"
    elif fault == "tools":
        report["native_tools"] = {"different": True}
    else:
        path = {
            "input": parent / "input.json",
            "claim": parent / "definition-review-successor.json",
            "binding": child / "binding.json",
            "result": parent / "report.json",
            "version": child / "input.json",
            "escape": child / "input.json",
        }[fault]
        value = json.loads(path.read_bytes())
        if fault == "version":
            value["schema_version"] = "generic-definition-review-successor/9.0.0"
        elif fault == "escape":
            value["parent_key"] = "../other"
        else:
            value["changed"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="media_definition_|generic_run_key_invalid"):
        acquisition_root(report, "child")
