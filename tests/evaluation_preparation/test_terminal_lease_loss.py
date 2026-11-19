"""Lease loss is a technical stop, never permission for a second dispatch."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from db.runtime.repository import case_pipeline as owner
from packages.evaluation_preparation import terminal_cohort as batch


def test_lease_loss_before_intent_performs_no_launch(tmp_path, monkeypatch):
    def lost():
        raise owner.SessionLeaseLostError("pipeline_session_lease_lost")

    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("dispatched"))
    with pytest.raises(owner.SessionLeaseLostError):
        batch.operation(tmp_path, SimpleNamespace(id="one"), "run", ["unused"], 5, lease_guard=lost)
    assert not (tmp_path / "commands/one/run/intent.json").exists()


def test_live_local_child_stopped_on_lease_loss_and_never_replayed(tmp_path, monkeypatch):
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count == 3:
            raise owner.SessionLeaseLostError("pipeline_session_lease_lost")

    case = SimpleNamespace(id="one")
    argv = [sys.executable, "-c", "import time;time.sleep(60)"]
    row = batch.operation(tmp_path, case, "run", argv, 10, lease_guard=guard)
    assert row["status"] == "lease_lost_remote_state_not_inferred"
    assert row["exit_code"] is None and type(row["child_exit_code"]) is int
    assert row["replay_authorized"] is False
    assert row["elapsed_seconds"] < 5
    assert batch.process_status(tmp_path / "commands/one/run") != "live_same_process"
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("replayed"))
    assert batch.operation(tmp_path, case, "run", argv, 10, lease_guard=guard) == row


def test_loss_after_intent_before_launch_records_no_child(tmp_path, monkeypatch):
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count == 2:
            raise owner.SessionLeaseLostError("pipeline_session_lease_lost")

    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: pytest.fail("dispatched"))
    row = batch.operation(
        tmp_path, SimpleNamespace(id="one"), "run", ["unused"], 5, lease_guard=guard
    )
    assert row["exit_code"] is None and row["child_exit_code"] is None
    assert row["status"] == "lease_lost_remote_state_not_inferred"
    assert (tmp_path / "commands/one/run/intent.json").exists()


def test_unconfirmed_intent_blocks_other_case_not_just_same_key(tmp_path):
    path = tmp_path / "commands/old-case/run/intent.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(
        ValueError, match="cohort_unconfirmed_command_requires_owner_reconciliation"
    ):
        batch.check_pending_commands(tmp_path)
    assert path.read_text() == "{}"


def test_terminal_failed_receipt_does_not_hide_uncertain_process(tmp_path):
    path = tmp_path / "commands/old-case/run"
    path.mkdir(parents=True)
    (path / "intent.json").write_text("{}")
    (path / "result.json").write_text(
        '{"status":"lease_lost_remote_state_not_inferred","exit_code":null}'
    )
    with pytest.raises(ValueError, match="cohort_uncertain_command_requires_owner_reconciliation"):
        batch.check_pending_commands(tmp_path)


def test_unrelated_completed_command_does_not_block(tmp_path):
    path = tmp_path / "commands/old-case/prepare"
    path.mkdir(parents=True)
    (path / "intent.json").write_text("{}")
    (path / "result.json").write_text('{"status":"command_exited","exit_code":0}')
    batch.check_pending_commands(tmp_path)


def test_public_supervisor_passes_guard_to_actual_child_and_stops_sequence(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_terminal_runtime_lease import manifest

    selected = manifest()
    monkeypatch.setattr(batch, "load", lambda *a, **k: (tmp_path, selected, {}))
    monkeypatch.setattr(batch, "child_lock_capability", lambda _: dict(available=True))
    count = 0

    def guard():
        nonlocal count
        count += 1
        if count == 4:
            raise owner.SessionLeaseLostError("pipeline_session_lease_lost")

    @contextmanager
    def session(_):
        yield guard

    monkeypatch.setattr(batch, "runtime_supervision_session", session)
    monkeypatch.setattr(
        batch,
        "commands",
        lambda *a: [[sys.executable, "-c", "import time; time.sleep(60)"]],
    )
    result = batch.run(tmp_path, "execute", [selected.cases[0].id])
    assert len(result) == 1 and count == 4
    assert result[0]["exit_code"] is None
    assert result[0]["status"] == "lease_lost_remote_state_not_inferred"
    assert not (tmp_path / "commands" / selected.cases[0].id / "preflight").exists()
    with pytest.raises(ValueError, match="cohort_uncertain_command_requires_owner_reconciliation"):
        batch.run(tmp_path, "prepare", [selected.cases[1].id])
    assert not (tmp_path / "commands" / selected.cases[1].id).exists()
