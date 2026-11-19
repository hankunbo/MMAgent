"""A text-only successor displays exact prior masks without weakening path gates."""

import json

import pytest

from packages.evaluation_preparation.evidence_media import acquisition_root, checked_file
from packages.grounding.models import digest
from packages.object_store import checksum
from tests.evaluation_preparation.test_evidence_tree_v2 import PNG


def fixture(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_input, generic_run

    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    original_directory = generic_input.directory
    monkeypatch.setattr(
        generic_input, "directory", lambda key: original_directory(key, root=tmp_path)
    )
    spec = dict(schema_version="fairvlmed-run-input/1.5.0", key="parent")
    selection = dict(
        schema_version="generic-review-selection/1.1.0",
        parent_key="parent",
        review_key="child",
        key="child",
        input_checksum=digest(spec),
    )
    binding = dict(case_code="CASE:TEST", image=dict(checksum=digest("input")))
    report = dict(result=dict(case_code="CASE:TEST"), native_tools=dict(results={}))
    for path, value in (
        (parent / "input.json", spec),
        (child / "input.json", selection),
        (parent / "review-successor.json", {k: v for k, v in selection.items() if k != "key"}),
        (parent / "binding.json", binding),
        (child / "binding.json", binding),
        (parent / "report.json", report),
    ):
        path.write_text(json.dumps(value))
    calls = []

    def export(key):
        calls.append(key)
        assert key == "parent"
        return dict(json=str(parent / "report.json"))

    monkeypatch.setattr(generic_run, "export", export)
    return parent, child, report, calls


def test_successor_uses_owner_bound_parent_and_keeps_file_scope(tmp_path, monkeypatch):
    parent, child, report, calls = fixture(tmp_path, monkeypatch)
    mask = parent / "mask.png"
    mask.write_bytes(PNG)
    with pytest.raises(ValueError, match="media_file_outside_run"):
        checked_file(child, mask.as_uri(), checksum(PNG))
    root, binding = acquisition_root(report, "child")
    assert root == parent and binding["case_code"] == "CASE:TEST" and calls == ["parent"]
    assert checked_file(root, mask.as_uri(), checksum(PNG)) == PNG
    other = tmp_path / "other.png"
    other.write_bytes(PNG)
    with pytest.raises(ValueError, match="media_file_outside_run"):
        checked_file(root, other.as_uri(), checksum(PNG))
    assert acquisition_root(report, "parent")[0] == parent


@pytest.mark.parametrize("fault", ["claim", "input", "binding", "case", "native", "self", "escape"])
def test_parent_must_match_original_selection_and_report(tmp_path, monkeypatch, fault):
    parent, child, report, calls = fixture(tmp_path, monkeypatch)
    path = parent / "review-successor.json"
    if fault == "input":
        path = parent / "input.json"
    elif fault == "binding":
        path = child / "binding.json"
    elif fault in {"case", "native"}:
        path = parent / "report.json"
    elif fault in {"self", "escape"}:
        path = child / "input.json"
    raw = json.loads(path.read_bytes())
    if fault == "case":
        raw["result"]["case_code"] = "CASE:OTHER"
    elif fault == "native":
        raw["native_tools"] = {"other": True}
    elif fault in {"self", "escape"}:
        raw["parent_key"] = "child" if fault == "self" else "../other"
    else:
        raw["changed"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="media_review_|generic_run_key_invalid"):
        acquisition_root(report, "child")
