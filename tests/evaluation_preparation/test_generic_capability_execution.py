"""Recovery and entry guards; persistence integration is a separate gate."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_execution as execution
from packages.evaluation_preparation.generic_capability_successor import VERSION
from packages.evaluation_preparation.generic_definition_successor import CLAIM
from packages.grounding.models import digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_files(tmp_path, monkeypatch):
    value = dict(schema_version=VERSION, key="new", parent_key="old")
    binding = dict(case="CASE:TEST")
    save(tmp_path / "new/input.json", value)
    save(tmp_path / "new/binding.json", binding)
    save(tmp_path / "old/binding.json", binding)
    save(tmp_path / "old" / CLAIM, value)
    save(tmp_path / "new/started.json", dict(key="new", input_checksum=digest(value)))
    save(tmp_path / "new/evidence/protocol.json", dict(deadline="old_deadline"))
    monkeypatch.setattr(execution, "directory", lambda key: tmp_path / key)
    checked = SimpleNamespace(campaign=SimpleNamespace(config=object()))
    monkeypatch.setattr(execution, "origin", lambda *a, **k: checked)
    monkeypatch.setattr(execution, "selection_for", lambda *a, **k: value)
    calls = []
    monkeypatch.setattr(
        execution,
        "CapabilitySuccessor",
        lambda *a: SimpleNamespace(freeze=lambda: calls.append("original_freeze")),
    )
    return value, calls


@pytest.mark.parametrize(
    "missing", ["input.json", "binding.json", "started.json", "evidence/protocol.json"]
)
def test_cold_restore_never_creates_missing_window(tmp_path, monkeypatch, missing):
    _, calls = setup_files(tmp_path, monkeypatch)
    path = tmp_path / "new" / missing
    path.unlink()
    with pytest.raises(ValueError, match="saved_execution_required"):
        execution.load_selection("new")
    assert not path.exists()
    assert calls == []


@pytest.mark.parametrize("changed", ["parent_claim", "binding", "start", "version"])
def test_cold_restore_rejects_changed_identity_before_owner_freeze(tmp_path, monkeypatch, changed):
    _, calls = setup_files(tmp_path, monkeypatch)
    path = {
        "parent_claim": tmp_path / "old" / CLAIM,
        "binding": tmp_path / "new/binding.json",
        "start": tmp_path / "new/started.json",
        "version": tmp_path / "new/input.json",
    }[changed]
    save(path, {})
    with pytest.raises(ValueError, match="generic_capability_(selection|saved_start)"):
        execution.load_selection("new")
    assert calls == []


def test_cold_restore_uses_original_freeze_without_restarting_time(tmp_path, monkeypatch):
    _, calls = setup_files(tmp_path, monkeypatch)
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    execution.load_selection("new")
    assert calls == ["original_freeze"]
    assert files == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_started_or_unknown_execution_requires_explicit_local_recovery(tmp_path, monkeypatch):
    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.evaluation_preparation import generic_execution_lock
    from packages.multimodal_research import unknown_acquisition

    _, calls = setup_files(tmp_path, monkeypatch)

    @contextmanager
    def lease(*args):
        yield lambda: None

    monkeypatch.setattr(generic_execution_lock, "target", lambda: object())
    monkeypatch.setattr(CasePipelineRepository, "__init__", lambda *a: None)
    monkeypatch.setattr(CasePipelineRepository, "guarded_session", lease)
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda *a: calls.append("network"))
    with pytest.raises(ValueError, match="started_use_export_not_resend"):
        execution.run("old", "new")
    assert calls == []


@pytest.mark.parametrize(
    "changed", [None, "phase_input_checksum", "selection_checksum", "correspondence_checksum"]
)
def test_report_scope_requires_exact_saved_production_projection(changed):
    parent = SimpleNamespace(material=SimpleNamespace(scope=SimpleNamespace(run_id="RUN:TEST")))
    # Parent binding is represented by a real digest-compatible mapping at this
    # unit boundary; medical projection integration is covered separately.
    from unittest.mock import patch

    from packages.evaluation_preparation import generic_capability_execution as module

    projection = dict(
        phase_input_checksum="state",
        selection_checksum="selection",
        correspondence_checksum="mapping",
    )
    selection = dict(selection_checksum="selection", correspondence_checksum="mapping")
    if changed:
        projection[changed] = "changed"
    campaign = SimpleNamespace(
        selection=selection,
        documents=lambda run: SimpleNamespace(
            read=lambda key: (
                (projection,) if key == "generic-capability-demand-projection" else None
            )
        ),
    )
    with patch.object(module, "digest", return_value="state"):
        if changed:
            with pytest.raises(ValueError, match="report_saved_projection_binding"):
                execution.report_metadata(campaign, parent)
        else:
            value = execution.report_metadata(campaign, parent)
            assert value["whole_case_aggregation_required"]
            assert not value["historical_scores_modified"]
            assert value["projection"] == projection
