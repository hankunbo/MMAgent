"""Source failure fallback keeps original Runtime exclusion and immutable history."""

import json
from contextlib import contextmanager

import pytest

from packages.multimodal_research import execution_lease as lease
from tests.evaluation_preparation.test_multimodal_entry_lease import bound_entry, fake_repository


def source_entry(path, parent):
    spec = bound_entry(path)
    spec.update(
        schema_version="multimodal-live-entry/1.30.0",
        definition_failure_of=dict(
            parent=str(parent),
            entry="sha256:" + "a" * 64,
            protocol="sha256:" + "b" * 64,
            failure="sha256:" + "c" * 64,
            mode="routed_root_source_backlog_no_source_replay",
        ),
    )
    spec["entry_execution_lock"]["schema_version"] = lease.SOURCE_PROFILE
    (path / "entry.json").write_text(json.dumps(spec))
    return spec


def test_source_profile_is_explicit_and_context_resets(tmp_path, monkeypatch):
    fake_repository(monkeypatch)
    spec = source_entry(tmp_path / "child", tmp_path / "parent")
    assert lease.validate_entry(spec)
    with lease.entry_session(tmp_path / "child"):
        assert lease.check_current().spec == spec
    assert lease.check_current() is None
    spec["entry_execution_lock"]["schema_version"] = lease.ENTRY_PROFILE
    with pytest.raises(ValueError, match="entry_execution_lock_binding"):
        lease.validate_entry(spec)


@pytest.mark.parametrize("change", ["fresh", "parent_missing", "mode", "previous", "files"])
def test_source_lock_cannot_authorize_arbitrary_history(tmp_path, change):
    spec = source_entry(tmp_path / "child", tmp_path / "parent")
    if change == "fresh":
        spec["schema_version"] = "multimodal-live-entry/1.29.0"
    elif change == "parent_missing":
        spec.pop("definition_failure_of")
    elif change == "mode":
        spec["definition_failure_of"]["mode"] = "replay_unknown"
    elif change == "previous":
        spec["previous"] = "other"
    else:
        spec["entry_execution_lock"]["implementation"] = {}
    with pytest.raises(ValueError, match="entry_execution_lock_binding"):
        lease.validate_entry(spec)


def test_historical_probe_uses_original_runtime_not_missing_flock(tmp_path, monkeypatch):
    fake_repository(monkeypatch)
    spec = bound_entry(tmp_path / "parent")
    spec["implementation"] = dict(spec["entry_execution_lock"]["implementation"])
    (tmp_path / "parent/entry.json").write_text(json.dumps(spec))
    # Current implementation drift is expected after a reviewed repair, but
    # the old lock identity and its binding to the old implementation stay exact.
    monkeypatch.setattr(lease, "entry_implementation", lambda: {})
    with lease.historical_entry_probe(tmp_path / "parent", spec):
        assert not (tmp_path / "parent/lock").exists()
    assert lease.check_current() is None


def test_historical_probe_propagates_busy_and_rejects_tamper(tmp_path, monkeypatch):
    spec = bound_entry(tmp_path / "parent")
    spec["implementation"] = dict(spec["entry_execution_lock"]["implementation"])
    (tmp_path / "parent/entry.json").write_text(json.dumps(spec))
    fake_repository(monkeypatch)
    spec["implementation"]["apps/multimodal_case.py"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="historical_entry_lock_binding"):
        with lease.historical_entry_probe(tmp_path / "parent", spec):
            pytest.fail("changed historical lock accepted")

    class Busy:
        def __init__(self, target):
            pass

        @contextmanager
        def guarded_session(self, case, key):
            raise ValueError("synthetic_busy")
            yield

    spec = json.loads((tmp_path / "parent/entry.json").read_text())
    monkeypatch.setattr(lease, "CasePipelineRepository", Busy)
    with pytest.raises(ValueError, match="synthetic_busy"):
        with lease.historical_entry_probe(tmp_path / "parent", spec):
            pytest.fail("busy parent bypassed")
