"""Parent-control contract probes, not a qualified component execution claim."""

from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.planner.service import Planner
from packages.grounding.models import digest
from packages.planner_control.models import (
    Candidate,
    Command,
    Control,
    Outcome,
    Pending,
    Phase,
    request_checksum,
)
from packages.research_runtime.component_steps import ComponentPlannerSteps, ParentCancellation
from tests.planner.support import FakeOwner, MemoryRuntime, advance, binding, command


def state():
    b = binding()
    cmd = Command(key="execute", expected_revision=0, action="execute")
    s = Control(
        binding=b,
        graph=b.graph,
        phase=Phase.EXECUTE,
        reserved=1,
        selected=Candidate(
            id="installed_tool", branch="installed", operation="tool", and_ready=True
        ),
        pending=Pending(
            command=cmd,
            request_checksum=request_checksum(b, cmd),
            execution_key="planner:one",
            launch_claimed=True,
        ),
    )
    return s, SimpleNamespace(read=lambda rid: s)


def test_parent_cancellation_inherits_absolute_cutoff_and_external_event():
    s, runtime = state()
    caller = Event()
    signal = ParentCancellation(runtime, s, s.binding.deadline, caller)
    assert not signal.is_set()
    caller.set()
    assert signal.is_set()
    assert ParentCancellation(runtime, s, datetime.now(UTC) - timedelta(seconds=1), None).is_set()
    cancelled = s.model_copy(update={"cancelled": True})
    runtime.read = lambda rid: cancelled
    assert ParentCancellation(runtime, s, s.binding.deadline, None).is_set()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pending", None),
        ("selected", None),
        ("reserved", 2),
    ],
)
def test_parent_signal_refuses_changed_control_not_new_budget(field, value):
    s, runtime = state()
    runtime.read = lambda rid: s.model_copy(update={field: value})
    signal = ParentCancellation(runtime, s, s.binding.deadline, None)
    assert signal.is_set()
    assert signal.reason == "component_parent_control_changed"


def adapter(s, runtime):
    # This unit fixture isolates control boundaries. The PG test uses the real
    # constructor/registration/admission and records its current qualification gap.
    owner = object.__new__(ComponentPlannerSteps)
    owner.sessions = runtime
    owner.work_deadline = s.binding.deadline
    owner._candidate = lambda: s.selected
    owner._current_plan = Mock(side_effect=AssertionError("must not compile"))
    owner.executor_for = Mock(side_effect=AssertionError("must not launch"))
    return owner


def test_cancelled_or_expired_before_compile_never_dispatches():
    s, runtime = state()
    owner = adapter(s, runtime)
    event = Event()
    event.set()
    assert owner.invoke(s, s.pending, cancel=event).reason == "deadline_or_cancel_before_launch"
    owner.work_deadline = datetime.now(UTC) - timedelta(seconds=1)
    assert owner.invoke(s, s.pending, cancel=None).reason == "deadline_or_cancel_before_launch"
    owner._current_plan.assert_not_called()
    owner.executor_for.assert_not_called()


def test_expiry_during_compile_never_dispatches():
    s, runtime = state()
    owner = adapter(s, runtime)
    event = Event()

    owner._saved = Mock(return_value=(None, None))

    def compile(_, **kwargs):
        event.set()
        return object(), object()

    owner._current_plan = compile
    assert owner.invoke(s, s.pending, cancel=event).reason == "deadline_or_cancel_before_launch"
    owner.executor_for.assert_not_called()


def test_public_invoke_cannot_use_stale_or_unreserved_control():
    s, runtime = state()
    owner = adapter(s, runtime)
    with pytest.raises(ValueError, match="current_control_required"):
        owner.invoke(s.model_copy(update={"revision": s.revision + 1}), s.pending, cancel=None)
    empty = s.model_copy(update={"reserved": 0})
    runtime.read = lambda rid: empty
    with pytest.raises(ValueError, match="parent_reservation_required"):
        owner.invoke(empty, empty.pending, cancel=None)


def test_unknown_recovery_uses_child_attempt_never_execute_or_readmit():
    s, runtime = state()
    owner = adapter(s, runtime)
    owner._saved = Mock(
        return_value=(SimpleNamespace(document=SimpleNamespace(run_id="RUN:CHILD")), None)
    )
    executor = SimpleNamespace(recover=Mock(return_value=None), execute=Mock())
    owner.executor_for = Mock(return_value=executor)
    assert owner.recover(s, s.pending).status == "unknown"
    executor.recover.assert_called_once_with("RUN:CHILD", s.pending.execution_key)
    executor.execute.assert_not_called()
    owner._current_plan.assert_not_called()


def test_original_planner_claim_loss_cannot_relaunch_or_reset_reservation():
    b, runtime = binding(), MemoryRuntime()
    fake = FakeOwner(b)
    fake.choices = (
        Candidate(id="installed_tool", branch="installed", operation="tool", and_ready=True),
    )
    planner = Planner(runtime=runtime, owner=fake)
    planner.begin(b)
    advance(planner, until="execute")
    assert planner.read(b.run_id).reserved == 1
    execute = command(planner, "execute")
    fake.failure = "execute"
    planner.step(b.run_id, execute)
    uncertain = planner.read(b.run_id)
    assert uncertain.pending.launch_claimed and uncertain.reserved == 1
    owner = adapter(uncertain, runtime)
    owner._saved = Mock(
        return_value=(SimpleNamespace(document=SimpleNamespace(run_id="RUN:CHILD")), None)
    )
    executor = SimpleNamespace(recover=Mock(return_value=None), execute=Mock())
    owner.executor_for = Mock(return_value=executor)
    cold = Planner(runtime=runtime, owner=owner)
    cold.step(b.run_id, execute)
    assert cold.read(b.run_id).reserved == 1
    assert cold.read(b.run_id).binding.deadline == b.deadline
    executor.execute.assert_not_called()


def test_original_owners_required():
    with pytest.raises(ValueError, match="original_owners_required"):
        ComponentPlannerSteps(
            snapshots=None,
            requests=object(),
            research=object(),
            choice=None,
            executor_for=None,
            sessions=None,
            work_deadline=datetime.now(UTC),
        )


def test_local_limits_do_not_require_database_and_read_failure_is_technical():
    s, runtime = state()
    runtime.read = Mock(side_effect=OSError("synthetic unavailable"))
    caller = Event()
    caller.set()
    assert ParentCancellation(runtime, s, s.binding.deadline, caller).is_set()
    runtime.read.assert_not_called()
    assert ParentCancellation(runtime, s, datetime.now(UTC) - timedelta(seconds=1), None).is_set()
    runtime.read.assert_not_called()
    signal = ParentCancellation(runtime, s, s.binding.deadline, None)
    assert signal.is_set() and signal.reason == "component_parent_control_unavailable"
    assert ComponentPlannerSteps._stopped(signal).gaps[0].kind == "technical"


def test_candidate_identity_binds_component_method_not_generic_installed_tool():
    owner = object.__new__(ComponentPlannerSteps)
    owner.requests = SimpleNamespace(registration_binding={"component": "first", "method": "one"})
    first = owner._candidate()
    owner.requests.registration_binding = {"component": "second", "method": "one"}
    second = owner._candidate()
    assert first.id != second.id and first.branch != second.branch
    owner.requests.registration_binding = {"component": "first", "method": "two"}
    assert owner._candidate() != first


def test_technical_stop_reason_has_exact_readonly_recovery():
    s, runtime = state()
    owner = adapter(s, runtime)
    workflow = {"synthetic_confirmed_workflow": True}
    original = Outcome(status="rejected", reason="tool_cancelled")
    owner._result = Mock(return_value=original)
    saved = dict(
        schema_version="component-parent-stop/1.0.0",
        parent_binding=digest(s.binding),
        execution_key=s.pending.execution_key,
        child_run_id="RUN:CHILD",
        workflow_checksum=digest(workflow),
        reason="component_parent_control_unavailable",
    )
    owner.documents = SimpleNamespace(read=Mock(return_value=(saved,)))
    outcome = owner._outcome(s, s.pending, workflow, "RUN:CHILD")
    assert outcome.reason == saved["reason"] and outcome.gaps[0].kind == "technical"
    assert owner._outcome(s, s.pending, workflow, "RUN:CHILD") == outcome
    with pytest.raises(ValueError, match="component_stop_binding"):
        owner._outcome(s, s.pending, workflow, "RUN:OTHER")
    owner.documents.read = Mock(return_value=None)
    assert owner._outcome(s, s.pending, workflow, "RUN:CHILD") == original


@pytest.mark.parametrize("create", [False, True])
def test_one_parent_reservation_cannot_start_multistep_plan(create):
    s, runtime = state()
    owner = adapter(s, runtime)
    request = SimpleNamespace(
        code="REQ:ONE",
        version="1.0.0",
        checksum=digest("request"),
        lock=SimpleNamespace(run_id="RUN:CHILD"),
    )
    child = SimpleNamespace(run_id="RUN:CHILD")
    owner.requests = SimpleNamespace(request_key="one", for_key=Mock(return_value=request))
    owner.research = SimpleNamespace(
        create=Mock(return_value=child),
        read=Mock(return_value=child),
        compile=Mock(return_value=SimpleNamespace(document=SimpleNamespace(steps=(1, 2)))),
    )
    owner.current = Mock(return_value=s.graph)
    owner.choice = "installed choice"
    with pytest.raises(ValueError, match="component_one_step_budget_required"):
        ComponentPlannerSteps._current_plan(owner, s, create=create)
    assert owner.research.create.call_count == int(create)
    assert owner.research.read.call_count == int(not create)
