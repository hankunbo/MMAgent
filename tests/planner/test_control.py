"""08A deterministic control, exact retry and failure boundaries."""

from datetime import timedelta
from pathlib import Path
from threading import Event as Cancellation

import pytest
from pydantic import ValidationError

from packages.grounding.models import digest, validate
from packages.planner_control.models import (
    Candidate,
    Command,
    Control,
    Gap,
    Phase,
    Profile,
)
from tests.planner.support import advance, command, setup


def test_normal_trace_and_empty_memory():
    p, r, o = setup()
    advance(p, until="done")
    s = r.state
    assert s.phase == Phase.STOPPED and len(p.trace(s.binding.run_id)) == 9
    assert (s.reserved, s.started, s.confirmed) == (1, 1, 1)
    assert s.memory == () and p.memory.retrieve(s.binding) == ()
    assert "review" not in o.calls and s.reason == "expert_not_required"


@pytest.mark.parametrize(
    "action",
    [
        "retrieve",
        "instantiate",
        "select",
        "prepare",
        "execute",
        "recover",
        "verify",
        "review",
        "fallback",
    ],
)
def test_illegal_initial_actions(action):
    p, r, _ = setup()
    with pytest.raises(ValueError, match="illegal_state_action"):
        p.step(r.state.binding.run_id, command(p, action))
    assert r.state.revision == 0


def test_strict_input_and_construct_revalidation():
    p, r, _ = setup()
    with pytest.raises(ValidationError):
        r.state.binding.profile.seconds = 60
    with pytest.raises(ValidationError):
        validate(Control, r.state.model_copy(update={"phase": "WAITING_FOR_HUMAN"}))
    with pytest.raises(ValueError):
        p.begin(r.state.binding.model_copy(update={"input_checksum": digest("changed")}))
    with pytest.raises(ValueError):
        p.step(
            r.state.binding.run_id,
            Command(key="bad", expected_revision=0, action="load").model_copy(
                update={"GraphDelta": {}}
            ),
        )
    with pytest.raises(ValueError):
        p.begin(
            r.state.binding.model_copy(
                update={"deadline": r.state.binding.deadline + timedelta(seconds=1)}
            )
        )


def test_exact_retry_and_key_conflict():
    p, r, o = setup()
    c = command(p, "load")
    e = p.step(r.state.binding.run_id, c)
    p.step(r.state.binding.run_id, command(p, "retrieve"))
    assert p.step(r.state.binding.run_id, c) == e
    assert o.calls == ["load", "retrieve"]
    with pytest.raises(ValueError, match="content_conflict"):
        p.step(r.state.binding.run_id, c.model_copy(update={"action": "retrieve"}))


def test_external_graph_change_rereads_and_invalidates():
    p, r, o = setup()
    advance(p)
    o.graph = o.graph.model_copy(
        update={"version": "case-evidence-graph/1.20.0", "checksum": digest("external")}
    )
    e = p.step(r.state.binding.run_id, command(p, "execute"))
    assert e.after == Phase.RETRIEVE and r.state.selected is None
    assert "execute" not in o.calls and not r.state.result_refs
    assert r.state.binding.graph != r.state.graph


@pytest.mark.parametrize("boundary", ["deadline", "cancel", "actions", "steps", "expert"])
def test_boundaries(boundary):
    profile = (
        Profile(max_actions=0)
        if boundary == "actions"
        else Profile(max_steps=1)
        if boundary == "steps"
        else Profile(expert_required=boundary == "expert")
    )
    p, r, o = setup(profile=profile)
    if boundary == "steps":
        advance(p, until="retrieve")
        c = command(p, "retrieve")
    elif boundary == "expert":
        advance(p, until="review")
        c = command(p, "review")
    elif boundary == "actions":
        advance(p, until="prepare")
        c = command(p, "prepare")
    else:
        advance(p)
        c = command(p, "execute")
    cancel = Cancellation()
    if boundary == "cancel":
        cancel.set()
    if boundary == "deadline":
        p.clock = lambda: r.state.binding.deadline
    p.step(r.state.binding.run_id, c, cancel=cancel)
    assert r.state.phase == Phase.STOPPED
    assert "execute" not in o.calls or boundary == "expert"
    assert r.state.reason == "expert_required_unavailable" if boundary == "expert" else True


def test_confirm_after_cancel_and_deadline():
    p, r, o = setup()
    advance(p)
    c = command(p, "execute")
    o.failure = "execute"
    s = p.step(r.state.binding.run_id, c)
    assert s.pending.launch_claimed and s.reason == "unknown_job_no_relaunch"
    o.failure = None
    p.clock = lambda: r.state.binding.deadline
    cancel = Cancellation()
    cancel.set()
    e = p.step(r.state.binding.run_id, c, cancel=cancel)
    assert e.after == Phase.VERIFY and r.state.confirmed == 1
    p.step(r.state.binding.run_id, command(p, "verify"), cancel=cancel)
    p.step(r.state.binding.run_id, command(p, "review"), cancel=cancel)
    assert r.state.phase == Phase.STOPPED
    assert o.calls.count("execute") == 1 and o.recoveries == 1


def test_unknown_job_recovery_bounded_never_relaunch():
    p, r, o = setup()
    advance(p)
    o.failure = "execute"
    c = command(p, "execute")
    for _ in range(8):
        p.step(r.state.binding.run_id, c)
    assert r.state.retries == r.state.binding.profile.max_retries
    assert o.calls.count("execute") == 1 and r.state.reserved == 1
    with pytest.raises(ValueError, match="pending_step"):
        p.step(r.state.binding.run_id, command(p, "cancel"))


@pytest.mark.parametrize("failure", ["rollback", "lost_reply"])
def test_transaction_failure_or_confirm_reply_lost(failure):
    p, r, o = setup()
    advance(p)
    c = command(p, "execute")
    o.on_call = lambda *args: setattr(r, "fail", failure)
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    e = p.step(r.state.binding.run_id, c)
    assert e.after == Phase.VERIFY and o.calls.count("execute") == 1
    assert r.state.confirmed == 1 and r.state.reserved == 1


def test_reservation_survives_prepare_confirmation_loss():
    p, r, o = setup()
    advance(p, until="prepare")
    c = command(p, "prepare")
    o.on_call = lambda *args: setattr(r, "fail", "rollback")
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    o.on_call = None
    p.step(r.state.binding.run_id, c)
    assert r.state.reserved == 1


def test_or_fallback_and_and_gate():
    p, r, o = setup()
    o.choices = (
        Candidate(id="A", branch="A", operation="new_review", and_ready=True),
        Candidate(id="B", branch="B", operation="tool", and_ready=False),
        Candidate(id="C", branch="C", operation="observe", and_ready=True),
    )
    advance(p, until="prepare")
    p.step(r.state.binding.run_id, command(p, "prepare"))
    assert r.state.reason == "08C_domain_acceptance_not_ready"
    p.step(r.state.binding.run_id, command(p, "fallback"))
    p.step(r.state.binding.run_id, command(p, "select"))
    assert r.state.selected.id == "C" and r.state.attempted == ("A",)


def test_illegal_provider_cannot_invent_candidate():
    p, r, _ = setup()
    advance(p, until="select")

    class Bad:
        def candidates(self, state):
            return (Candidate(id="invented", branch="x", operation="tool", and_ready=True),)

    p.actions = Bad()
    e = p.step(r.state.binding.run_id, command(p, "select"))
    assert e.outcome.reason == "illegal_provider" and r.state.selected is None


@pytest.mark.parametrize(
    "kind",
    [
        "missing_data",
        "product_demand",
        "mapping",
        "unknown_producer",
        "safety",
        "technical",
        "unavailable_port",
    ],
)
def test_gap_layers_do_not_invent_capability(kind):
    gap = Gap(kind=kind, target="node", reason="missing")
    assert gap.capability_code is None and gap.dispatch == "recorded_for_11_not_dispatched"
    with pytest.raises(ValidationError):
        Gap(kind=kind, target="node", reason="missing", capability_code="CAPABILITY:FAKE")


def test_resolved_gap_requires_owned_requirement():
    with pytest.raises(ValidationError):
        Gap(kind="capability", target="node", reason="missing")


def test_no_agent_db_graph_or_process_authority():
    import ast

    for path in Path("agents/planner").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("db", "apps", "subprocess", "psycopg"))
        assert not any(
            token in path.read_text() for token in ("apply_graph_delta(", "Popen(", "cursor(")
        )


@pytest.mark.parametrize("action", ["select", "fallback"])
def test_internal_transition_confirmation_is_atomic(action):
    p, r, o = setup()
    if action == "fallback":
        o.choices = (Candidate(id="A", branch="A", operation="new_branch", and_ready=True),)
        advance(p, until="execute")
    else:
        advance(p, until="select")
    c = command(p, action)
    original = r.save

    def fail_finish(old, new):
        if new.pending is None:
            r.fail = "lost_reply"
        return original(old, new)

    r.save = fail_finish
    with pytest.raises(RuntimeError):
        p.step(r.state.binding.run_id, c)
    r.save = original
    p.step(r.state.binding.run_id, c)
    assert r.state.branches == (1 if action == "fallback" else 0)


def test_fallback_budget_and_no_information_progress():
    p, r, o = setup(profile=Profile(max_no_progress=1))
    o.choices = (Candidate(id="A", branch="A", operation="new_branch", and_ready=True),)
    advance(p, until="execute")
    p.step(r.state.binding.run_id, command(p, "fallback"))
    assert r.state.phase == Phase.STOPPED and r.state.reason == "fallback_budget"


def test_schema_snapshot(monkeypatch):
    from agents.planner.schema_tools import main

    monkeypatch.setattr("sys.argv", ["schema_tools"])
    main()


def test_report_and_replay_never_call_owner():
    p, r, o = setup()
    advance(p, until="done")
    before = o.calls[:]
    assert p.replay(r.state.binding.run_id) == r.state
    report = p.report(r.state.binding.run_id)
    assert report.clinical_effect is None and report.api_requests == 0
    assert o.calls == before and report.control_checksum == digest(r.state)


def test_repository_rejects_unaccompanied_phase_jump():
    p, r, _ = setup()
    with pytest.raises(ValueError, match="transition_requires_event"):
        r.save(r.state, r.state.model_copy(update=dict(phase=Phase.EXECUTE, revision=1)))


def test_selection_and_fallback_do_not_reset_no_progress():
    p, r, o = setup(profile=Profile(max_no_progress=2))
    o.choices = tuple(
        Candidate(id=x, branch=x, operation="new_branch", and_ready=True) for x in "ABC"
    )
    advance(p, until="execute")
    assert r.state.no_progress == 1
    for action in ("fallback", "select", "prepare"):
        p.step(r.state.binding.run_id, command(p, action))
    assert r.state.no_progress == 2
    p.step(r.state.binding.run_id, command(p, "fallback"))
    assert r.state.phase == Phase.STOPPED and r.state.reason == "fallback_budget"


def test_unavailable_action_provider_uses_existing_deterministic_baseline():
    p, r, _ = setup()
    advance(p, until="select")

    class Unavailable:
        def candidates(self, state):
            raise TimeoutError("synthetic_provider_unavailable")

    p.actions = Unavailable()
    e = p.step(r.state.binding.run_id, command(p, "select"))
    assert e.outcome.reason == "provider_unavailable_deterministic_baseline"
    assert r.state.selected == r.state.candidates[0]


def test_resolved_capability_keeps_owned_requirement_and_no_dispatch():
    from packages.planner_control.models import Reference

    requirement = Reference(
        role="requirement",
        code="REQUIREMENT:SYNTHETIC",
        version="1.0.0",
        checksum=digest("requirement"),
    )
    gap = Gap(
        kind="capability",
        target="node",
        reason="resolved_no_tool",
        capability_code="CAPABILITY:SYNTHETIC",
        requirement_ref=requirement,
    )
    assert gap.requirement_ref == requirement
    assert gap.dispatch == "recorded_for_11_not_dispatched"
