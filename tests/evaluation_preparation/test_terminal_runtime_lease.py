"""Explicit supervisor version, no implicit fallback, and pre-dispatch checks."""

import errno
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as batch


def manifest():
    value = json.loads(Path("docs/acceptance/terminal-cohort-q-plan/manifest.json").read_bytes())
    value.update(
        schema_version="terminal-case-cohort/1.8.0",
        supervision_lease="original-runtime-case-pipeline-session/1.0.0",
    )
    return batch.read_manifest(value)


def unavailable(*args):
    raise OSError(errno.ENOLCK, "synthetic unavailable filesystem lock")


def test_explicit_manifest_keeps_original_commands_and_old_contract():
    new = manifest()
    raw = new.model_dump(mode="json")
    raw.pop("supervision_lease")
    raw["schema_version"] = "terminal-case-cohort/1.6.0"
    old = batch.read_manifest(raw)
    for case in new.cases:
        for action in ("prepare", "preflight", "run", "export"):
            assert batch.commands(new, case, Path("/synthetic"), action) == batch.commands(
                old, case, Path("/synthetic"), action
            )
    raw["supervision_lease"] = new.supervision_lease
    with pytest.raises(ValueError):
        batch.read_manifest(raw)


def test_invalid_or_missing_lease_profile_rejected():
    raw = manifest().model_dump(mode="json")
    raw["supervision_lease"] = "caller-says-unlocked"
    with pytest.raises(ValueError):
        batch.read_manifest(raw)
    raw.pop("supervision_lease")
    with pytest.raises(ValueError):
        batch.read_manifest(raw)


def test_runtime_selected_explicitly_and_finally_released(tmp_path, monkeypatch):
    entered = []

    @contextmanager
    def owned_session(output):
        entered.append(("acquired", output))
        try:
            yield
        finally:
            entered.append(("released", output))

    monkeypatch.setattr(batch, "runtime_supervision_session", owned_session)
    monkeypatch.setattr(batch.fcntl, "flock", lambda *a: pytest.fail("implicit filesystem lock"))
    with pytest.raises(RuntimeError, match="child"):
        with batch.lock(tmp_path, manifest()):
            raise RuntimeError("child")
    assert entered == [("acquired", tmp_path), ("released", tmp_path)]
    assert not (tmp_path / "cohort.lock").exists()


def test_legacy_enolck_is_not_automatic_runtime_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.fcntl, "flock", unavailable)
    monkeypatch.setattr(batch, "runtime_supervision_session", lambda *a: pytest.fail("fallback"))
    with pytest.raises(OSError) as error:
        with batch.lock(tmp_path):
            pytest.fail("unlocked")
    assert error.value.errno == errno.ENOLCK


def test_child_capability_preserves_error_and_closes_temp_file(tmp_path, monkeypatch):
    before = set(tmp_path.iterdir())
    monkeypatch.setattr(batch.fcntl, "flock", unavailable)
    result = batch.child_lock_capability(tmp_path)
    assert result == dict(available=False, errno=errno.ENOLCK, implicit_fallback=False)
    assert set(tmp_path.iterdir()) == before


def test_child_capability_other_error_is_not_misreported(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError(errno.EACCES, "synthetic permission denied")

    monkeypatch.setattr(batch.fcntl, "flock", denied)
    with pytest.raises(PermissionError):
        batch.child_lock_capability(tmp_path)


def test_run_refuses_child_lock_failure_before_command_intent(tmp_path, monkeypatch):
    selected = manifest()
    monkeypatch.setattr(batch, "load", lambda *a, **k: (tmp_path, selected, {}))

    @contextmanager
    def owned(output):
        yield

    monkeypatch.setattr(batch, "runtime_supervision_session", owned)
    monkeypatch.setattr(batch.fcntl, "flock", unavailable)
    monkeypatch.setattr(batch, "case_action", lambda *a: pytest.fail("child dispatched"))
    with pytest.raises(ValueError, match="cohort_child_filesystem_lock_unavailable"):
        batch.run(tmp_path, "execute", [selected.cases[0].id])
    assert not (tmp_path / "commands").exists()


def test_lock_preflight_is_readonly_and_does_not_claim_end_to_end(tmp_path, monkeypatch):
    selected = manifest()
    monkeypatch.setattr(batch, "load", lambda *a, **k: (tmp_path, selected, {}))

    @contextmanager
    def owned(output):
        yield

    monkeypatch.setattr(batch, "runtime_supervision_session", owned)
    monkeypatch.setattr(batch.fcntl, "flock", unavailable)
    value = batch.lease_preflight(tmp_path)
    assert value["supervision_lease_acquired"] is True
    assert value["execution_lock_ready"] is False
    assert value["api_requests"] == value["database_writes"] == 0
    assert value["owner_readiness_not_inferred"] is True
