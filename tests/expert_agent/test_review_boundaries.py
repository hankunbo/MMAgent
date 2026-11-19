"""Independent 08D probes: all synthetic, no external tools or model APIs."""

from threading import Event
from types import SimpleNamespace

import pytest

from agents.expert_agent.actions import ResearchActionSource
from agents.expert_agent.models import ReviewFacts, ReviewPolicy
from agents.planner.service import Planner
from apps.expert_agent_assembly import assemble_research_review
from packages.grounding.models import digest
from packages.planner_control.models import Command, Outcome, RankingPolicy
from tests.agent_provider.support import ref, review_from_wire
from tests.expert_agent.support import expert_reply, protocol, run, setup
from tests.planner.support import FakeOwner, MemoryRuntime


def integrated(f, runtime=None):
    class Owner(FakeOwner):
        def current(self, binding):
            return f.reader.current().material.scope.graph

        def invoke(self, state, pending, *, cancel):
            if pending.command.action == "execute":
                return Outcome(reason="synthetic_no_tool", refs=(ref("material", "ASSET:READ"),))
            return super().invoke(state, pending, cancel=cancel)

    runtime = runtime or MemoryRuntime()
    owner = Owner(f.binding)
    owner.choices = f.source.choices
    planner = Planner(runtime=runtime, owner=owner)
    planner.begin(f.binding)

    def command(action):
        state = planner.read(f.binding.run_id)
        return Command(
            key=f"step-{state.steps}-{action}", expected_revision=state.revision, action=action
        )

    for name in (
        "load",
        "retrieve",
        "instantiate",
        "identify",
        "select",
        "prepare",
        "execute",
        "verify",
    ):
        planner.step(f.binding.run_id, command(name))
    event = planner.begin_research_review(f.binding.run_id, command("review"))
    service = assemble_research_review(
        planner=planner,
        run_id=f.binding.run_id,
        domain_owner=f.reader,
        documents=f.documents,
        ports=f.ports,
        action_source=ResearchActionSource(f.source, f.reader),
        ranking_policy=RankingPolicy(),
        review_policy=ReviewPolicy(comparison_protocol=digest(protocol())),
        facts=lambda state: ReviewFacts(),
        install=f.context.install,
        protocol=protocol(),
    )
    frame = service.context.read(event)
    action = next(a for a in frame.actions.actions if a.branch == "research_review")
    return SimpleNamespace(
        planner=planner,
        service=service,
        args=dict(event=event, proposed_action_id=action.id, role="review"),
        command=command,
    )


def test_baseline_acceptance_control():
    f = setup([review_from_wire])
    h = integrated(f)
    result = h.service.run(**h.args)
    assert result["status"] == "accepted"
    assert h.planner.read(f.binding.run_id).phase == "retrieve"
    assert h.service.run(**h.args) == result
    assert f.budget.used == 1


def test_step_budget_blocks_new_review_request():
    base = setup()
    b = base.binding.model_copy(
        update={"profile": base.binding.profile.model_copy(update={"max_steps": 8})}
    )
    f = setup([review_from_wire], binding=b)
    h = integrated(f)
    before = h.planner.read(f.binding.run_id)
    assert before.steps == before.binding.profile.max_steps == 8
    result = h.service.run(**h.args)
    after = h.planner.read(f.binding.run_id)
    print(
        {
            "probe": "step_budget",
            "limit": 8,
            "steps": after.steps,
            "status": result["status"],
            "requests": f.budget.used,
            "phase": after.phase,
        }
    )
    assert f.budget.used == 0, "new Provider request started after locked max_steps exhausted"
    assert after.phase == "stopped"


def test_locked_expert_required_cannot_be_downgraded_by_default_facts():
    base = setup()
    b = base.binding.model_copy(
        update={"profile": base.binding.profile.model_copy(update={"expert_required": True})}
    )
    f = setup([review_from_wire], binding=b)
    h = integrated(f)
    result = h.service.run(**h.args)
    print(
        {
            "probe": "expert_required",
            "locked_required": True,
            "status": result["status"],
            "roles": [r.role for r in f.budget.records()],
        }
    )
    assert result["status"] != "accepted", (
        "mandatory Expert was skipped and domain candidate accepted"
    )
    assert "expert" in [r.role for r in f.budget.records()]


def test_cancellation_after_acceptance_is_durable_and_stops_next_cycle(monkeypatch):
    f = setup([review_from_wire])
    h = integrated(f)
    signal = Event()
    accept = h.service.acceptance.accept

    def accepted_then_cancel(*args, **kwargs):
        receipt = accept(*args, **kwargs)
        signal.set()
        return receipt

    monkeypatch.setattr(h.service.acceptance, "accept", accepted_then_cancel)
    result = h.service.run(**h.args, cancel=signal)
    state = h.planner.read(f.binding.run_id)
    print(
        {
            "probe": "cancel_after_accept",
            "status": result["status"],
            "phase": state.phase,
            "cancelled": state.cancelled,
            "signal": signal.is_set(),
            "requests": f.budget.used,
        }
    )
    assert state.cancelled, "cancel signal lost after confirmed domain acceptance"
    assert state.phase == "stopped"
    assert (
        h.service.acceptance.read(key=h.args["event"].code + ":domain")[-1]["status"] == "accepted"
    )


def test_cancellation_during_candidate_saved_to_original_checkpoint():
    signal = Event()

    def reply(body):
        signal.set()
        return review_from_wire(body)

    f = setup([reply])
    h = integrated(f)
    result = h.service.run(**h.args, cancel=signal)
    state = h.planner.read(f.binding.run_id)
    print(
        {
            "probe": "cancel_during_candidate",
            "status": result["status"],
            "phase": state.phase,
            "cancelled": state.cancelled,
        }
    )
    assert result["status"] == "safe_stop"
    assert state.cancelled, "cancel signal did not reach durable original Control"


def test_withdrawn_source_before_expert_recovery_does_not_dispatch(monkeypatch):
    f = setup([review_from_wire, expert_reply()], facts={"required": True})
    write = f.documents.write

    def lost(key, values):
        write(key, values)
        if key.endswith(":selected"):
            raise OSError("selected_confirmation_lost")

    monkeypatch.setattr(f.documents, "write", lost)
    with pytest.raises(OSError, match="selected_confirmation_lost"):
        run(f, role="review")
    monkeypatch.setattr(f.documents, "write", write)
    assert f.budget.used == 1
    f.reader.withdrawn = True
    result = run(f, role="review")
    print(
        {
            "probe": "withdraw_before_expert",
            "status": result["status"],
            "requests": f.budget.used,
            "roles": [r.role for r in f.budget.records()],
        }
    )
    assert f.budget.used == 1, "Expert request dispatched before checking already withdrawn owner"
    assert result["status"] == "safe_stop"


def test_withdrawn_source_after_acceptance_has_bounded_terminal_checkpoint(monkeypatch):
    f = setup([review_from_wire])
    h = integrated(f)
    accept = h.service.acceptance.accept

    def accepted_then_withdraw(*args, **kwargs):
        receipt = accept(*args, **kwargs)
        f.reader.withdrawn = True
        return receipt

    monkeypatch.setattr(h.service.acceptance, "accept", accepted_then_withdraw)
    errors = []
    for _ in range(2):
        try:
            h.service.run(**h.args)
        except ValueError as error:
            errors.append(str(error))
    state = h.planner.read(f.binding.run_id)
    saved = h.service.acceptance.read(key=h.args["event"].code + ":domain")
    print(
        {
            "probe": "withdraw_after_accept",
            "errors": errors,
            "phase": state.phase,
            "pending": state.pending is not None,
            "domain_status": saved[-1]["status"],
            "requests": f.budget.used,
        }
    )
    assert not errors, (
        "confirmed acceptance with revoked source never reaches bounded terminal report"
    )
    assert state.phase == "stopped" and state.pending is None
    assert saved[-1]["status"] == "accepted" and f.budget.used == 1
