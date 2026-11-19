from threading import Event

import pytest

from packages.grounding.models import digest
from packages.planner_control.models import Phase
from packages.task_runtime.transitions import change
from tests.planner.support import advance
from tests.task_runtime.support import configured


def prepare(owner, runtime, receipt, key="t"):
    now = runtime.state.binding.started_at
    receipt = change(receipt, "claim", dict(worker="w", token=key), runtime.state, now)
    return change(
        receipt,
        "prepare",
        dict(worker="w", token=key),
        runtime.state,
        now,
        command=owner.next_command(receipt),
    )


def test_original_planner_one_step_not_whole_run():
    owner, _, runtime, domain, receipt, _ = configured()
    receipt = prepare(owner, runtime, receipt)
    assert receipt.command.action == "load"
    state = owner.advance(receipt, cancel=Event())
    assert state.phase == Phase.RETRIEVE and domain.calls == ["load"]
    # Lost dispatch confirm: replay original command, no duplicate original invocation.
    assert owner.advance(receipt, cancel=Event()) == state
    assert domain.calls == ["load"]
    receipt = change(
        receipt, "confirm", dict(worker="w", token="t"), state, runtime.state.binding.started_at
    )
    assert receipt.command is None and receipt.state == "queued"


def test_cancel_before_execution_and_inherited_deadline():
    owner, planner, runtime, domain, receipt, _ = configured()
    advance(planner)
    receipt = prepare(owner, runtime, receipt)
    original_deadline = runtime.state.binding.deadline
    cancel = Event()
    cancel.set()
    state = owner.advance(receipt, cancel=cancel)
    assert state.phase == Phase.STOPPED and state.cancelled
    assert "execute" not in domain.calls and state.binding.deadline == original_deadline


def test_unknown_uses_exact_pending_recovery_only():
    owner, planner, runtime, domain, receipt, _ = configured()
    advance(planner)
    receipt = prepare(owner, runtime, receipt)
    domain.failure = "execute"
    owner.advance(receipt, cancel=Event())
    assert runtime.state.pending.launch_claimed
    pending = runtime.state.pending.command
    assert owner.next_command(receipt) == pending
    domain.failure = None
    owner.advance(receipt, cancel=Event())
    assert domain.calls.count("execute") == 1 and domain.recoveries == 1
    assert runtime.state.pending is None


def test_original_source_owner_and_scope_fail_closed():
    owner, _, _, domain, receipt, context = configured()
    context.synthetic = False
    with pytest.raises(ValueError, match="dispatch_synthetic_only"):
        owner.current(receipt.request)
    context.synthetic = True
    with pytest.raises(ValueError, match="dispatch_scope"):
        owner.current(receipt.request.model_copy(update={"context_checksum": digest("wrong")}))

    def withdrawn(_):
        raise ValueError("source_withdrawn")

    domain.current = withdrawn
    with pytest.raises(ValueError, match="source_withdrawn"):
        owner.current(receipt.request, sources=True)
    # Reading confirmed facts/monotone cancellation must remain possible.
    assert owner.current(receipt.request).binding.run_id == receipt.request.run_id


def test_uninstalled_repository_owner_is_never_trusted():
    from db.runtime.repository.dispatch import DispatchRepository

    _, _, _, _, receipt, _ = configured()
    repository = object.__new__(DispatchRepository)
    repository.owner = None
    with pytest.raises(ValueError, match="dispatch_owner_required"):
        repository.submit(receipt.request)


def test_cancel_revision_and_same_key_conflicts_through_repository():
    from db.runtime.repository.dispatch import DispatchRepository

    owner, _, runtime, _, receipt, _ = configured()
    repository = object.__new__(DispatchRepository)
    repository.owner = owner
    values, commands = [receipt], {}
    repository.read = lambda _: values[0]
    repository.now = lambda: runtime.state.binding.started_at
    repository.history = lambda run, key: commands.get(key)

    def save(old, new, state, key, operation):
        values[0] = new
        commands[key] = dict(command=operation, receipt=new.model_dump(mode="json"))
        return new

    repository._save = save
    run_id = receipt.request.run_id
    with pytest.raises(ValueError, match="dispatch_control_changed"):
        repository.command(run_id, "cancel", "cancel", expected_revision=1)
    changed = repository.command(run_id, "cancel", "cancel", expected_revision=0)
    assert changed.cancel_requested and changed.revision == 1
    assert repository.command(run_id, "cancel", "cancel", expected_revision=0) == changed
    with pytest.raises(ValueError, match="dispatch_key_conflict"):
        repository.command(run_id, "cancel", "cancel", expected_revision=1)
    assert len(commands) == 1 and values[0] == changed


def test_worker_shutdown_does_not_claim_or_cancel_case():
    from types import SimpleNamespace

    from apps.worker.service import Worker

    owner, _, runtime, domain, receipt, _ = configured()
    worker = Worker(SimpleNamespace(owner=owner, read=lambda _: receipt), worker_id="w")
    shutdown = Event()
    shutdown.set()
    assert (
        worker.run_bounded(receipt.request.run_id, batch_key="batch", shutdown=shutdown) == receipt
    )
    assert domain.calls == [] and not runtime.state.cancelled


def test_expired_original_deadline_only_closes_no_domain_call():
    owner, planner, runtime, domain, receipt, _ = configured()
    receipt = prepare(owner, runtime, receipt)
    planner.clock = lambda: runtime.state.binding.deadline
    result = owner.advance(receipt, cancel=Event())
    assert result.phase == Phase.STOPPED
    assert domain.calls == [] and result.binding.deadline == receipt.deadline
