"""Outer I/O faults preserve the original pending/key and monotone cancellation."""

from threading import Event

import pytest

from packages.grounding.models import digest
from packages.planner_control.models import Command, Outcome
from tests.expert_agent.support import expert_reply
from tests.validation_recovery.support import fixture


@pytest.mark.parametrize("point", ["incident", "proposal", "report", "checkpoint"])
@pytest.mark.parametrize("committed", [False, True])
def test_cancel_during_confirmation_loss_recovers_without_new_requests(
    monkeypatch, point, committed
):
    h = fixture(blocked=("B",), replies=(expert_reply(action="safe_stop"),))
    signal = Event()
    before = h.planner.read(h.run_id)
    with monkeypatch.context() as m:
        if point == "checkpoint":
            original = h.runtime.save

            def broken(old, new):
                if new.phase == "stopped":
                    signal.set()
                    if committed:
                        original(old, new)
                    raise OSError("synthetic_confirmation_lost")
                return original(old, new)

            m.setattr(h.runtime, "save", broken)
        else:
            original = h.f.documents.write

            def broken(key, values):
                if key.endswith(":" + point):
                    signal.set()
                    if committed:
                        original(key, values)
                    raise OSError("synthetic_confirmation_lost")
                return original(key, values)

            m.setattr(h.f.documents, "write", broken)
        with pytest.raises(OSError, match="synthetic_confirmation_lost"):
            h.service.resolve(h.run_id, key="fault", cancel=signal)
    persisted = h.planner.read(h.run_id)
    requests = h.f.budget.used
    signal.clear()
    result = h.service.resolve(h.run_id, key="fault", cancel=signal)
    final = h.planner.read(h.run_id)
    assert final.phase == "stopped" and final.pending is None
    if point != "checkpoint" or not committed:
        assert persisted.cancelled and final.cancelled
    assert h.f.budget.used == requests and final.reserved == before.reserved
    assert final.started == before.started == 0 and final.graph == before.graph
    assert h.service.resolve(h.run_id, key="fault") == result


def test_report_failure_and_cancel_persistence_failure_preserve_primary_and_cause(monkeypatch):
    h = fixture(blocked=("B",), replies=(expert_reply(action="safe_stop"),))
    signal = Event()
    write, save = h.f.documents.write, h.runtime.save
    with monkeypatch.context() as m:

        def bad_write(key, values):
            if key.endswith(":report"):
                signal.set()
                raise OSError("synthetic_object_failure")
            return write(key, values)

        def bad_save(old, new):
            if new.cancelled:
                raise ConnectionError("synthetic_cancel_unconfirmed")
            return save(old, new)

        m.setattr(h.f.documents, "write", bad_write)
        m.setattr(h.runtime, "save", bad_save)
        with pytest.raises(OSError, match="synthetic_object_failure") as error:
            h.service.resolve(h.run_id, key="double", cancel=signal)
        assert isinstance(error.value.__cause__, ConnectionError)
    assert not h.planner.read(h.run_id).cancelled
    assert h.planner.read(h.run_id).pending is not None
    requests = h.f.budget.used
    h.service.resolve(h.run_id, key="double", cancel=signal)
    assert h.planner.read(h.run_id).cancelled and h.f.budget.used == requests


def test_same_key_policy_content_conflict_and_new_key_cannot_escape_unknown(monkeypatch):
    h = fixture(blocked=("B",))
    h.f.documents.fail = "object"
    with pytest.raises(OSError):
        h.service.resolve(h.run_id, key="pending")
    state = h.planner.read(h.run_id)
    with pytest.raises(ValueError, match="pending_step_requires_same_command"):
        h.service.resolve(h.run_id, key="new-key")
    with pytest.raises(ValueError, match="pending_step_requires_same_command"):
        h.planner.step(
            h.run_id,
            Command(
                key=state.pending.command.key,
                expected_revision=state.pending.command.expected_revision,
                action="cancel",
            ),
        )
    h.f.documents.fail = None
    h.service.resolve(h.run_id, key="pending")
    h.service.policy = h.policy.model_copy(
        update={"source": h.policy.source.model_copy(update={"checksum": digest("changed")})}
    )
    with pytest.raises(ValueError, match="recovery_key_content_conflict"):
        h.service.resolve(h.run_id, key="pending")


def test_unknown_execution_same_pending_no_automatic_relaunch(monkeypatch):
    h = fixture()
    h.service.resolve(h.run_id, key="branch")
    for action in ("select", "prepare"):
        h.planner.step(h.run_id, h.command(action))
    calls = []

    def unknown(*args, **kwargs):
        calls.append(1)
        return Outcome(status="unknown", reason="synthetic_cleanup_unknown")

    monkeypatch.setattr(h.owner, "invoke", unknown)
    monkeypatch.setattr(h.owner, "recover", unknown)
    cmd = h.command("execute")
    h.planner.step(h.run_id, cmd)
    pending = h.planner.read(h.run_id).pending
    for _ in range(6):
        h.service.resume_execution(h.run_id)
        state = h.planner.read(h.run_id)
        assert state.pending == pending and state.pending.launch_claimed
    assert state.retries == state.binding.profile.max_retries and state.started == 0
    assert h.owner.calls.count("execute") == 1  # Only initial synthetic branch A.
    with pytest.raises(ValueError, match="pending_step_requires_same_command"):
        h.service.resolve(h.run_id, key="unknown-escape")


@pytest.mark.parametrize("existing", [False, True])
def test_incident_read_failure_observes_cancel_without_replacing_pending(monkeypatch, existing):
    h = fixture(blocked=("B",))
    if existing:
        h.f.documents.fail = "object"
        with pytest.raises(OSError):
            h.service.resolve(h.run_id, key="read")
        h.f.documents.fail = None
    original = h.planner.read(h.run_id)
    signal = Event()
    read = h.f.documents.read
    with monkeypatch.context() as m:

        def failed(key):
            if key == "recovery:read:incident":
                signal.set()
                raise OSError("synthetic_incident_read_failed")
            return read(key)

        m.setattr(h.f.documents, "read", failed)
        with pytest.raises(OSError, match="synthetic_incident_read_failed"):
            h.service.resolve(h.run_id, key="read", cancel=signal)
    current = h.planner.read(h.run_id)
    assert current.cancelled and current.pending == original.pending
    signal.clear()
    h.service.resolve(h.run_id, key="read")
    assert h.planner.read(h.run_id).phase == "stopped" and h.f.budget.used == 0


@pytest.mark.parametrize("point", ["fallback-report", "fallback_checkpoint"])
def test_fallback_cancelled_during_outer_io_keeps_report_and_never_selects(monkeypatch, point):
    h = fixture(category="capability_gap")
    signal = Event()
    with monkeypatch.context() as m:
        if point == "fallback-report":
            write = h.f.documents.write

            def failed(key, values):
                write(key, values)
                if key.endswith(":fallback-report"):
                    signal.set()
                    raise OSError("synthetic_fallback_lost")

            m.setattr(h.f.documents, "write", failed)
        else:
            save = h.runtime.save

            def failed(old, new):
                value = save(old, new)
                if new.phase == "select" and new.events[-1].command.action == "fallback":
                    signal.set()
                    raise OSError("synthetic_fallback_lost")
                return value

            m.setattr(h.runtime, "save", failed)
        with pytest.raises(OSError, match="synthetic_fallback_lost"):
            h.service.resolve(h.run_id, key="fallback", cancel=signal)
    signal.clear()
    result = h.service.resolve(h.run_id, key="fallback")
    assert result["report"]["unobserved_nodes"] == ["GOAL:SYNTHETIC"]
    assert result["control"]["phase"] == "stopped" and h.f.budget.used == 0
    assert h.planner.read(h.run_id).cancelled and h.owner.calls.count("execute") == 1


def test_inner_08_result_io_and_control_failure_keep_original_error(monkeypatch):
    h = fixture(blocked=("B",), replies=(expert_reply(action="safe_stop"),))
    signal = Event()
    write, save = h.f.documents.write, h.runtime.save
    with monkeypatch.context() as m:

        def broken(key, values):
            if key.startswith("review:") and key.endswith(":result"):
                signal.set()
                raise OSError("synthetic_original_result_io")
            return write(key, values)

        def control_failure(old, new):
            if new.cancelled:
                raise ConnectionError("synthetic_control_io")
            return save(old, new)

        m.setattr(h.f.documents, "write", broken)
        m.setattr(h.runtime, "save", control_failure)
        with pytest.raises(OSError, match="synthetic_original_result_io") as caught:
            h.service.resolve(h.run_id, key="inner", cancel=signal)
        assert isinstance(caught.value.__cause__, ConnectionError)
    used = h.f.budget.used
    h.service.resolve(h.run_id, key="inner", cancel=signal)
    assert h.planner.read(h.run_id).cancelled and h.f.budget.used == used
