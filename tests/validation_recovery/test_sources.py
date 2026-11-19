from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.planner_control.actions import RankedActions
from packages.planner_control.models import Candidate, Outcome
from packages.validation_recovery import (
    BranchRoute,
    Failure,
    NearestLegalActions,
    RecoveryPolicy,
    ValidationRecovery,
)
from tests.agent_provider.support import ref, review_from_wire
from tests.expert_agent.support import expert_reply, setup
from tests.expert_agent.test_review_boundaries import integrated
from tests.validation_recovery.support import fixture


def test_nearest_legal_fork_precedes_farther_and_rederives_after_withdrawal():
    h = fixture()
    c = Candidate(id="C", branch="C", operation="observe", and_ready=True)
    h.f.source.choices += (c,)
    # Pure ActionSource unit probe with an explicit synthetic snapshot, not a persisted
    # Planner transition. The service integration tests use actual identify events.
    state = h.planner.read(h.run_id).model_copy(update={"candidates": h.f.source.choices})
    policy = h.policy.model_copy(
        update={"routes": h.policy.routes + (BranchRoute(branch="C", forks=("root", "far")),)}
    )
    source = NearestLegalActions(h.f.source, policy)
    first = source.read(state, h.planner.ranking.policy)
    assert [a.branch for a in first.actions if a.candidate] == ["B"]
    h.f.source.blocked.add("B")
    second = source.read(state, h.planner.ranking.policy)
    assert [a.branch for a in second.actions if a.candidate] == ["C"]
    h.f.source.blocked.add("C")
    assert not any(a.candidate for a in source.read(state, h.planner.ranking.policy).actions)


@pytest.mark.parametrize("error", [OSError("catalog_io"), ValueError("object_checksum_invalid")])
def test_catalog_read_failure_is_not_fake_missing_tool_or_business_stop(monkeypatch, error):
    h = fixture()
    original = h.planner.read(h.run_id)

    def broken(*args):
        raise error

    monkeypatch.setattr(h.f.source, "read", broken)
    with pytest.raises(type(error), match=str(error)):
        h.service.resolve(h.run_id, key="catalog-error")
    assert h.planner.read(h.run_id) == original
    assert not any(k.startswith("recovery:") for k in h.f.documents.saved)


def test_accepted_summary_scores_and_old_result_remain_exact_after_stop(monkeypatch):
    from packages.path_graph_validation import PathGraphValidation, ResearchPathReader

    f = setup([review_from_wire, expert_reply(action="safe_stop")])
    parent = f.reader.state
    h = integrated(f)
    accepted = h.service.run(**h.args)
    historical = digest(f.acceptance.read(key=accepted["acceptance_key"]))
    path = PathGraphValidation(
        reader=ResearchPathReader(
            acceptance=f.acceptance, parent=parent, key=accepted["acceptance_key"]
        ),
        documents=f.documents,
    )
    summary = path.inspect(key="accepted")
    for name in ("retrieve", "instantiate", "identify", "select", "prepare", "execute", "verify"):
        h.planner.step(f.binding.run_id, h.command(name))
    policy = RecoveryPolicy(
        routes=(BranchRoute(branch="A", forks=("root",)), BranchRoute(branch="B", forks=("root",))),
        source=next(r for r in f.binding.locks if r.role == "rule"),
    )
    nearest = NearestLegalActions(f.source, policy)
    h.planner.ranking = RankedActions(nearest)
    h.service.context.source = nearest
    service = ValidationRecovery(
        planner=h.planner,
        review=h.service,
        documents=f.documents,
        policy=policy,
        failure_reader=lambda _: Failure(
            category="valid_refutation",
            target="GOAL:SYNTHETIC",
            reason="bounded_research_direction",
            sources=(ref("graph", "ASSET:SOURCE"),),
        ),
        summary_reader=lambda: path.read(key="accepted"),
    )
    before = h.planner.read(f.binding.run_id)
    result = service.resolve(f.binding.run_id, key="accepted-stop")
    assert result["report"]["incident"]["summary"] == summary
    assert result["control"]["phase"] == "stopped"
    assert h.planner.read(f.binding.run_id).graph == before.graph
    assert digest(f.acceptance.read(key=accepted["acceptance_key"])) == historical
    assert path.read(key="accepted", historical=True) == summary
    # Source withdrawal cannot turn a saved advisory result into fresh authorization.
    f.reader.withdrawn = True
    used = f.budget.used
    assert service.resolve(f.binding.run_id, key="accepted-stop")["control"]["phase"] == "stopped"
    assert f.budget.used == used and path.read(key="accepted", historical=True) == summary


def test_demand_reregistration_across_two_failures_is_one_intent(monkeypatch):
    h = fixture(category="capability_gap", replies=(expert_reply(action="safe_stop"),))
    first = h.service.resolve(h.run_id, key="first")
    for action in ("select", "prepare"):
        h.planner.step(h.run_id, h.command(action))
    monkeypatch.setattr(
        h.owner,
        "invoke",
        lambda *args, **kwargs: Outcome(
            status="rejected", reason="synthetic_second_failure", progress=False
        ),
    )
    h.planner.step(h.run_id, h.command("execute"))
    second = h.service.resolve(h.run_id, key="second")
    assert second["report"]["incident"]["requirements"] == first["incident"]["requirements"]
    assert len([k for k in h.f.documents.saved if k.startswith("recovery:requirement:")]) == 1
    assert second["control"]["phase"] == "stopped"


def test_selected_tools_use_original_research_steps_and_choice_binding(monkeypatch):
    from packages.research_runtime.planner_steps import ResearchPlannerSteps
    from packages.tool_workflow import WorkflowChoice
    from packages.validation_recovery.tool_steps import SelectedResearchSteps

    h = fixture()
    choice = WorkflowChoice(roots=(), nodes=())
    selected = Candidate(
        id="tool",
        branch="tool",
        operation="tool",
        and_ready=True,
        input_refs=(ref("workflow_choice", "CHOICE:SYNTHETIC", checksum=digest(choice)),),
    )
    state = h.planner.read(h.run_id).model_copy(update={"selected": selected})
    adapter = SelectedResearchSteps(
        snapshots=SimpleNamespace(current=lambda _: state.graph),
        research=None,
        candidates=lambda _: (selected,),
        choice_for=lambda *_: choice,
        executor_for=lambda _: None,
    )
    calls = []

    def invoked(owner, control, pending, *, cancel):
        calls.append((owner.choice, pending))
        return Outcome(reason="delegated_to_original_06")

    monkeypatch.setattr(ResearchPlannerSteps, "invoke", invoked)
    pending = SimpleNamespace(command=SimpleNamespace(action="prepare"))
    adapter.invoke(state, pending, cancel=None)
    assert calls == [(choice, pending)]
    adapter.choice_for = lambda *_: choice.model_copy(update={"roots": (digest("wrong"),)})
    with pytest.raises(ValueError, match="recovery_workflow_choice_binding"):
        adapter.invoke(state, pending, cancel=None)
    assert len(calls) == 1
