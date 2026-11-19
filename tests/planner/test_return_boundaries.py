"""Independent 08A checks; anonymous fixtures, no real model/GPU/patient/production DB."""

from threading import Event as Cancellation
from types import SimpleNamespace

import pytest

from agents.planner.service import Planner
from packages.grounding.models import digest
from packages.planner_control.models import Event, Phase
from packages.research_runtime.planner_steps import ResearchPlannerSteps
from packages.tool_execution.models import ExecutionError
from packages.tool_execution.trusted_models import TrustedTechnicalOutcome
from packages.tool_workflow import ResearchContext
from tests.planner.support import MemoryRuntime, advance, binding, command
from tests.planner.support import setup as control_setup
from tests.tool_execution.test_trusted_local import setup as setup
from tests.tool_execution.trusted_support import prepared
from tests.tool_workflow.support import choose


def wired(setup):
    f = prepared(setup)
    b = binding(case_code=f.request.scope.case_code, run_id=f.request.lock.run_id)
    b = b.model_copy(
        update={
            "locks": tuple(
                r.model_copy(
                    update=dict(
                        code=f.plan.document.catalog_ref.code,
                        version=f.plan.document.catalog_ref.version,
                        checksum=f.plan.document.catalog_ref.checksum,
                    )
                )
                if r.role == "catalog"
                else r.model_copy(update=dict(checksum=f.plan.document.config.checksum))
                if r.role == "execution"
                else r
                for r in b.locks
            )
        }
    )

    class Research:
        def read(self, run_id):
            assert run_id == b.run_id
            return SimpleNamespace(checksum=b.run_context_checksum)

        def compile(self, run_id, choice):
            self.read(run_id)
            admission = f.compiler.admission.admit_research(f.request)
            return f.compiler.compile(
                ResearchContext(request=f.request, admission=admission), choice
            )

    snapshots = SimpleNamespace(current=lambda _: b.graph)
    from packages.planner_control.models import Outcome

    snapshots.step = lambda *_: Outcome(reason="synthetic_snapshot")
    owner = ResearchPlannerSteps(
        snapshots=snapshots,
        research=Research(),
        choice=choose(f.compiler.admission.admit_research(f.request)),
        executor_for=lambda _: f.service,
        operation="tool",
    )
    runtime = MemoryRuntime()
    p = Planner(runtime=runtime, owner=owner)
    p.begin(b)
    advance(p)
    return p, runtime, f


def test_trusted_success_is_confirmed_without_spurious_unknown(setup):
    p, r, f = wired(setup)
    c = command(p, "execute")
    result = p.step(r.state.binding.run_id, c)
    records = tuple(f.repo.attempts.values())
    assert len(records) == 1 and records[0]["result"]["status"] == "SUCCEEDED"
    print(
        "trusted_success",
        dict(
            return_type=type(result).__name__,
            phase=r.state.phase,
            reason=r.state.reason,
            durable_status=records[0]["status"],
        ),
    )
    assert isinstance(result, Event) and result.after == Phase.VERIFY
    assert (r.state.reserved, r.state.started, r.state.confirmed) == (1, 1, 1)
    revision = r.state.revision
    assert p.step(r.state.binding.run_id, c) == result and r.state.revision == revision
    assert sum(status == "RUNNING" for _, status in f.repo.transitions) == 1


def test_trusted_known_prelaunch_rejection_is_not_unknown(setup, monkeypatch):
    p, r, f = wired(setup)
    original = f.service.execute

    def revoke_before_execute(*args, **kwargs):
        f.catalogs.disabled = (f.request.candidate.implementation_ref,)
        return original(*args, **kwargs)

    monkeypatch.setattr(f.service, "execute", revoke_before_execute)
    c = command(p, "execute")
    result = p.step(r.state.binding.run_id, c)
    assert not f.repo.attempts
    print(
        "trusted_prelaunch_rejection",
        dict(
            return_type=type(result).__name__,
            phase=r.state.phase,
            reason=r.state.reason,
            attempts=len(f.repo.attempts),
        ),
    )
    assert isinstance(result, Event) and result.outcome.status == "rejected"
    assert r.state.pending is None
    assert (r.state.reserved, r.state.started, r.state.confirmed) == (1, 0, 0)
    revision = r.state.revision
    assert p.step(r.state.binding.run_id, c) == result and r.state.revision == revision
    p.step(r.state.binding.run_id, command(p, "fallback"))
    p.step(r.state.binding.run_id, command(p, "select"))
    assert r.state.phase == Phase.STOPPED and not f.repo.attempts


def test_cancelled_confirmation_then_source_failure_still_closes(monkeypatch):
    p, r, o = control_setup()
    advance(p)
    c = command(p, "execute")
    o.failure = "execute"
    p.step(r.state.binding.run_id, c)
    o.failure = None
    cancel = Cancellation()
    cancel.set()
    p.step(r.state.binding.run_id, c, cancel=cancel)
    assert r.state.phase == Phase.VERIFY and r.state.cancelled and r.state.confirmed == 1

    def source_failure(_):
        raise ValueError("graph_inventory_changed_reread_reassembly_required")

    monkeypatch.setattr(o, "current", source_failure)
    result = p.step(r.state.binding.run_id, command(p, "verify"), cancel=cancel)
    assert result.after == Phase.STOPPED and r.state.cancelled and r.state.confirmed == 1


@pytest.mark.parametrize("fault", ["no_attempt", "before_confirm", "after_confirm", "cleanup"])
def test_uncertain_never_relaunches_and_confirmed_recovery_counts_once(setup, monkeypatch, fault):
    p, r, f = wired(setup)
    c = command(p, "execute")
    run = r.state.binding.run_id
    original = f.repo.confirm
    if fault == "no_attempt":
        monkeypatch.setattr(
            f.service,
            "execute",
            lambda *a, **k: TrustedTechnicalOutcome(
                policy_checksum=digest(f.service._policy),
                capabilities=f.service._policy.capabilities,
                state="uncertain",
                reason="attempt_outcome_uncertain_offline_recovery",
            ),
        )
    elif fault == "cleanup":
        original_cleanup = f.service._runner._cleanup

        def lost_cleanup(proc, observed):
            assert original_cleanup(proc, observed)  # Actually clean the harmless child.
            return False

        monkeypatch.setattr(f.service._runner, "_cleanup", lost_cleanup)
    else:

        def lost_confirm(workflow):
            if fault == "after_confirm":
                original(workflow)
            raise ConnectionError("lost_confirmation")

        monkeypatch.setattr(f.repo, "confirm", lost_confirm)
    first = p.step(run, c)
    assert first.pending.launch_claimed and first.confirmed == 0
    assert first.reserved == 1
    monkeypatch.setattr(f.repo, "confirm", original)
    cancel = Cancellation()
    cancel.set()
    second = p.step(run, c, cancel=cancel)
    launches = sum(s == "RUNNING" for _, s in f.repo.transitions)
    if fault in {"before_confirm", "after_confirm"}:
        assert isinstance(second, Event) and second.after == Phase.VERIFY
        assert r.state.cancelled and r.state.confirmed == r.state.started == 1
        count = len(r.audit)
        assert p.step(run, c) == second and len(r.audit) == count
        assert launches == 1
    else:
        assert second.pending.launch_claimed and second.cancelled and second.confirmed == 0
        for _ in range(4):
            last = p.step(run, c)
        assert last.pending.execution_key == first.pending.execution_key
        assert last.reserved == 1 and last.confirmed == 0
        assert launches == (0 if fault == "no_attempt" else 1)
    assert sum(s == "RUNNING" for _, s in f.repo.transitions) == launches


def test_return_contract_rejects_wrong_type_version_policy_plan_and_source(setup):
    f = prepared(setup)
    key = "strict-return"
    result = f.service.execute(f.plan, key)
    assert result.state == "recorded"
    assert f.service.execution_result(f.plan, key, result) == result.workflow
    wrong_values = [
        SimpleNamespace(workflow=result.workflow),
        result.model_dump(mode="json"),
        result.workflow,
        result.model_copy(update={"schema_version": "trusted-local-outcome/9.0.0"}),
        result.model_copy(update={"policy_checksum": digest("foreign-policy")}),
        result.model_copy(
            update={"capabilities": result.capabilities.model_copy(update={"network": "deny"})}
        ),
        result.model_copy(update={"state": "technical_rejection"}),
        result.model_copy(
            update={
                "workflow": result.workflow.model_copy(
                    update={"plan_checksum": digest("foreign-plan")}
                )
            }
        ),
    ]
    for value in wrong_values:
        with pytest.raises((ValueError, TypeError)):
            f.service.execution_result(f.plan, key, value)
    with pytest.raises(ExecutionError, match="unconfirmed"):
        f.service.execution_result(f.plan, "other-key", result)
    record = f.repo.recover(f.plan.document.run_id, key)
    original = dict(record)
    for field, value in [
        ("plan_checksum", digest("other")),
        ("request_hash", digest("other")),
        ("run_id", "RUN:OTHER"),
        ("status", "ACTIVE"),
        ("result", None),
    ]:
        record[field] = value
        with pytest.raises(ValueError):
            f.service.execution_result(f.plan, key, result)
        record.update(original)
    # A fully valid result from another owned attempt is still the wrong source.
    other = f.service.execute(f.plan, "other-attempt")
    with pytest.raises(ValueError):
        f.service.execution_result(f.plan, key, other)
    assert sum(s == "RUNNING" for _, s in f.repo.transitions) == 2


@pytest.mark.parametrize("active", [False, True])
def test_rejection_with_existing_attempt_is_unknown(setup, monkeypatch, active):
    p, r, f = wired(setup)
    original_execute, original_confirm = f.service.execute, f.repo.confirm
    if active:

        def unconfirmed(workflow):
            raise ConnectionError("confirmation_not_received")

        monkeypatch.setattr(f.repo, "confirm", unconfirmed)

    def execute_then_revoke(plan, key, **kwargs):
        original_execute(plan, key, **kwargs)
        f.catalogs.disabled = (f.request.candidate.implementation_ref,)
        rejected = original_execute(plan, key, **kwargs)
        assert rejected.state == "technical_rejection"
        return rejected

    monkeypatch.setattr(f.service, "execute", execute_then_revoke)
    c = command(p, "execute")
    result = p.step(r.state.binding.run_id, c)
    assert result.pending.launch_claimed and result.confirmed == 0
    assert len(f.repo.attempts) == 1
    assert next(iter(f.repo.attempts.values()))["status"] == ("ACTIVE" if active else "CONFIRMED")
    monkeypatch.setattr(f.repo, "confirm", original_confirm)
    f.catalogs.disabled = ()
    event = p.step(r.state.binding.run_id, c)
    assert event.after == Phase.VERIFY and r.state.confirmed == 1
    assert sum(s == "RUNNING" for _, s in f.repo.transitions) == 1


def test_missing_attempt_alone_or_unsubstantiated_rejection_is_not_authority(setup):
    f = prepared(setup)
    for state, reason in [
        ("uncertain", "unknown"),
        ("technical_rejection", "controlled_validation_rejected"),
        ("technical_rejection", "controlled_execution_rejected"),
    ]:
        value = TrustedTechnicalOutcome(
            policy_checksum=digest(f.service._policy),
            capabilities=f.service._policy.capabilities,
            state=state,
            reason=reason,
        )
        with pytest.raises(ExecutionError, match="unconfirmed"):
            f.service.execution_result(f.plan, "no-execute", value)
    assert not f.repo.attempts


@pytest.mark.parametrize("boundary", ["source", "graph"])
@pytest.mark.parametrize("saved_cancel", [False, True])
def test_cancelled_verify_failure_closes_with_same_event_and_counts(
    monkeypatch, boundary, saved_cancel
):
    p, r, o = control_setup()
    advance(p, until="verify")
    cancel = Cancellation()
    if saved_cancel:
        # A cancellation checkpoint, with unchanged facts, is already durable.
        p._save(r.state, cancelled=True)
    before = r.state

    def changed(_):
        cancel.set()  # Also exercise a signal arriving during the source read.
        if boundary == "source":
            raise ValueError("source_rejected")
        return before.graph.model_copy(update={"checksum": digest("external-change")})

    monkeypatch.setattr(o, "current", changed)
    c = command(p, "verify")
    event = p.step(before.binding.run_id, c, cancel=cancel)
    assert event.after == Phase.STOPPED and event.outcome.status == "rejected"
    assert r.state.cancelled and r.state.pending is None
    assert r.state.confirmed == before.confirmed == 1
    assert r.state.result_refs == before.result_refs and r.state.graph == before.graph
    checkpoint, audit = r.state, list(r.audit)
    assert p.step(before.binding.run_id, c) == event
    assert (r.state, r.audit) == (checkpoint, audit)
    assert o.calls.count("execute") == 1


@pytest.mark.parametrize("recovering", [False, True])
def test_cancel_arriving_during_unknown_owner_call_is_saved(recovering, monkeypatch):
    p, r, o = control_setup()
    advance(p)
    cancel = Cancellation()
    c = command(p, "execute")
    o.failure = "execute"
    if recovering:
        p.step(r.state.binding.run_id, c)
        original = o.recover

        def recover(*args):
            cancel.set()
            return original(*args)

        monkeypatch.setattr(o, "recover", recover)
    else:
        o.on_call = lambda *_: cancel.set()
    first = p.step(r.state.binding.run_id, c, cancel=cancel)
    assert first.cancelled and first.pending.launch_claimed
    assert p.step(r.state.binding.run_id, c).cancelled
    assert o.calls.count("execute") == 1
