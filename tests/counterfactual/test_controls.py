from threading import Event

import pytest

from packages.counterfactual.service import CounterfactualSimulation
from packages.planner_control.models import Command, Phase
from tests.counterfactual.support import candidate, fixture, protection


@pytest.mark.parametrize(
    "kind", ["cancel", "deadline", "branches", "actions", "steps", "no_progress"]
)
def test_current_control_stops_without_new_requests(kind):
    x = fixture(profile_updates=dict(max_steps=10) if kind == "steps" else None)
    c = candidate()
    x.service.declare((c,))
    state = x.h.planner.read(x.f.binding.run_id)
    signal = Event()
    changes = {}
    if kind == "cancel":
        signal.set()
    elif kind == "deadline":
        x.h.planner.clock = lambda: state.binding.deadline
    elif kind == "steps":
        x.h.planner.step(x.f.binding.run_id, x.h.command("retrieve"))
    else:
        field, limit = {
            "branches": ("branches", "max_branches"),
            "actions": ("reserved", "max_actions"),
            "steps": ("steps", "max_steps"),
            "no_progress": ("no_progress", "max_no_progress"),
        }[kind]
        changes[field] = getattr(state.binding.profile, limit)
        x.h.planner.runtime.state = state.model_copy(update=changes)
        # Injected read-boundary Control only; not a persisted Planner transition.
    before = protection(x)
    result = x.service.run(c, cancel=signal)
    assert result["status"] == "stopped" and result["report"] is None
    assert protection(x) == before and x.f.budget.used == 1
    if kind == "cancel":
        signal.clear()
        assert x.service.run(c)["status"] == "stopped"


@pytest.mark.parametrize("point", ["policy", "plan", "result:one", "validation:one"])
@pytest.mark.parametrize("lost", [False, True])
def test_same_key_io_recovery_no_facts_or_requests(point, lost, monkeypatch):
    x = fixture()
    c = candidate()
    if point not in {"policy", "plan"}:
        x.service.declare((c,))
    before = protection(x)
    write = x.f.documents.write
    with monkeypatch.context() as m:

        def failure(key, values):
            if key == "simulation:" + point:
                if lost:
                    write(key, values)
                raise OSError("synthetic_reply_lost" if lost else "synthetic_io")
            return write(key, values)

        m.setattr(x.f.documents, "write", failure)
        with pytest.raises(OSError):
            if point in {"policy", "plan"}:
                x.service.declare((c,))
            else:
                x.service.run(c)
    x.service = CounterfactualSimulation(reader=x.reader, documents=x.f.documents, policy=x.policy)
    x.service.declare((c,))
    result = x.service.run(c)
    assert result["status"] == "candidate" and protection(x) == before
    assert x.service.run(c)["report"] == result["report"]


def test_unknown_original_pending_cannot_escape_via_simulation_key():
    x = fixture()
    c = candidate()
    x.service.declare((c,))
    planner = x.h.planner
    for name in ("retrieve", "instantiate", "identify", "select", "prepare"):
        planner.step(x.f.binding.run_id, x.h.command(name))
    state = planner.read(x.f.binding.run_id)
    command = Command(key="unknown-execute", action="execute", expected_revision=state.revision)
    from packages.planner_control.models import Pending
    from packages.planner_control.transitions import request_checksum

    pending = Pending(
        command=command,
        request_checksum=request_checksum(state.binding, command),
        execution_key="original-unknown",
        launch_claimed=True,
    )
    planner._save(state, pending=pending, phase=Phase.EXECUTE)
    before = planner.read(x.f.binding.run_id)
    with pytest.raises(ValueError, match="original_pending_requires_recovery"):
        x.service.run(c)
    assert planner.read(x.f.binding.run_id) == before and x.f.budget.used == 1


@pytest.mark.parametrize("change", ["withdraw", "deadline", "cancel"])
def test_report_saved_then_current_stop_preserves_historical_report(change, monkeypatch):
    x = fixture()
    c = candidate()
    x.service.declare((c,))
    signal = Event()
    write = x.f.documents.write

    def changed(key, values):
        write(key, values)
        if key == "simulation:result:one":
            if change == "withdraw":
                x.f.reader.withdrawn = True
            elif change == "deadline":
                x.h.planner.clock = lambda: x.f.binding.deadline
            else:
                signal.set()

    monkeypatch.setattr(x.f.documents, "write", changed)
    result = x.service.run(c, cancel=signal)
    assert result["status"] == "stopped" and result["report"] is not None
    assert x.service.read(candidate_id=c.id, historical=True) == result["report"]
    assert x.service.run(c, cancel=signal)["status"] == "stopped" and x.f.budget.used == 1


def test_lost_report_then_withdrawal_zero_requests(monkeypatch):
    x = fixture()
    c = candidate()
    x.service.declare((c,))
    write = x.f.documents.write
    with monkeypatch.context() as m:

        def lost(key, values):
            write(key, values)
            if key == "simulation:result:one":
                raise OSError("confirmation_lost")

        m.setattr(x.f.documents, "write", lost)
        with pytest.raises(OSError):
            x.service.run(c)
    x.f.reader.withdrawn = True
    result = x.service.run(c)
    assert result["status"] == "stopped" and result["report"]
    assert x.f.budget.used == 1


def test_original_error_and_cancel_persistence_error_keep_cause(monkeypatch):
    x = fixture()
    c = candidate()
    x.service.declare((c,))
    signal = Event()
    read = x.f.documents.read
    observe = x.h.planner.observe_recovery_cancel

    def fail_read(key):
        if key == "simulation:plan":
            signal.set()
            raise OSError("original_object_error")
        return read(key)

    def fail_cancel(*args, **kwargs):
        if signal.is_set():
            raise RuntimeError("cancel_confirmation_unknown")
        return observe(*args, **kwargs)

    monkeypatch.setattr(x.f.documents, "read", fail_read)
    monkeypatch.setattr(x.h.planner, "observe_recovery_cancel", fail_cancel)
    with pytest.raises(OSError, match="original_object_error") as error:
        x.service.run(c, cancel=signal)
    assert "cancel_confirmation_unknown" in str(error.value.__cause__)


@pytest.mark.parametrize("change", ["scope", "policy", "key", "budget", "combination"])
def test_explicit_policy_conflicts(change):
    x = fixture()
    c = candidate()
    x.service.declare((c,))
    if change == "scope":
        x.service.policy = x.policy.model_copy(update=dict(run_id="RUN:OTHER"))
    elif change == "policy":
        x.service.policy = x.policy.model_copy(update=dict(tolerance=0.9))
    elif change == "key":
        c = candidate("remove_resource", tuple(r.code for r in x.parent.resources))
    elif change == "budget":
        with pytest.raises(ValueError, match="budget_or_duplicate"):
            x.service.declare(tuple(candidate(id=str(n)) for n in range(65)))
        return
    else:
        x.service.policy = x.policy.model_copy(
            update=dict(forbidden_combinations=(("remove_evidence",),))
        )
    with pytest.raises(ValueError):
        x.service.run(c)
    assert x.f.budget.used == 1


@pytest.mark.parametrize("point", ["policy", "plan", "validation:one"])
def test_cancel_during_outer_write_is_persisted(point, monkeypatch):
    x = fixture()
    c = candidate()
    signal = Event()
    if point == "validation:one":
        x.service.declare((c,))
    write = x.f.documents.write

    def cancel_write(key, values):
        write(key, values)
        if key == "simulation:" + point:
            signal.set()

    monkeypatch.setattr(x.f.documents, "write", cancel_write)
    if point == "validation:one":
        assert x.service.run(c, cancel=signal)["status"] == "stopped"
    elif point == "policy":
        with pytest.raises(ValueError, match="cancelled"):
            x.service.declare((c,), cancel=signal)
    else:
        x.service.declare((c,), cancel=signal)
    assert x.h.planner.read(x.f.binding.run_id).cancelled
    assert x.f.budget.used == 1
