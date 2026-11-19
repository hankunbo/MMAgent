"""Synthetic counterexamples: owned reservation is not a new budget request."""

from datetime import timedelta

import pytest

from agents.planner.service import Planner
from packages.grounding.models import digest, seal
from packages.planner_control.actions import RankedActions
from packages.planner_control.models import Command, Phase, Profile, RankingPolicy
from tests.grounding.test_database import database as database
from tests.planner.support import FakeOwner, command
from tests.planner.test_actions import Source, Suggest, ranked_binding, select, setup_ranked


@pytest.fixture(autouse=True)
def no_tool_execution(monkeypatch):
    from packages.tool_execution.service import ToolExecutionService
    from packages.tool_workflow.compiler import ToolWorkflowCompiler

    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("forbidden")
        raise AssertionError("prepare recovery test must not compile or execute tools")

    monkeypatch.setattr(ToolWorkflowCompiler, "compile", forbidden)
    monkeypatch.setattr(ToolExecutionService, "execute", forbidden)
    yield
    assert attempts == []


def idempotent_preparation(owner):
    """Fake original-owner idempotency, not a new Planner cache or production port."""
    original, records, invocations = owner.invoke, {}, []

    def invoke(state, pending, *, cancel):
        if pending.command.action != "prepare":
            return original(state, pending, cancel=cancel)
        invocations.append(pending.execution_key)
        if pending.execution_key not in records:
            records[pending.execution_key] = original(state, pending, cancel=cancel)
        return records[pending.execution_key]

    owner.invoke = invoke
    return records, invocations


def inject_save_failure(runtime, stage):
    original = runtime.save

    def save(old, new):
        if (
            stage == "after_reservation"
            and new.pending
            and new.pending.reserved
            and not old.pending.reserved
        ):
            original(old, new)
            raise ConnectionError("reservation_reply_lost")
        if stage == "before_confirmation" and len(new.events) > len(old.events):
            raise ConnectionError("prepare_confirmation_failed")
        return original(old, new)

    runtime.save = save
    return original


@pytest.mark.parametrize("max_actions", [1, 2])
@pytest.mark.parametrize("stage", ["after_reservation", "before_confirmation"])
def test_same_prepare_retry_reuses_owned_reservation(max_actions, stage):
    p, r, owner, _ = setup_ranked(agent=Suggest(), profile=Profile(max_actions=max_actions))
    records, invocations = idempotent_preparation(owner)
    select(p)
    cmd = command(p, "prepare")
    original = inject_save_failure(r, stage)
    with pytest.raises(ConnectionError):
        p.step(r.state.binding.run_id, cmd)
    r.save = original
    assert r.state.phase == Phase.PREPARE and r.state.pending.reserved and r.state.reserved == 1
    assert owner.calls.count("execute") == 0
    p = Planner(runtime=r, owner=owner, ranking=p.ranking)
    event = p.step(r.state.binding.run_id, cmd)
    print(
        dict(
            stage=stage,
            max_actions=max_actions,
            phase=r.state.phase.value,
            reason=event.outcome.reason,
            reserved=r.state.reserved,
            prepares=owner.calls.count("prepare"),
            executes=owner.calls.count("execute"),
        )
    )
    assert event.after == Phase.EXECUTE, event.outcome.reason
    assert r.state.reserved == 1 and r.state.pending is None
    assert len(records) == 1 and len(invocations) == (2 if stage == "before_confirmation" else 1)
    before = (r.state, tuple(r.audit), len(invocations))
    assert p.step(r.state.binding.run_id, cmd) == event
    assert before == (r.state, tuple(r.audit), len(invocations))


def test_ranked_pg_roundtrip_and_last_reservation_recovery(database, tmp_path):
    # Original case/research Runtime entrypoints; no tool compile or execute needed.
    from db import manage
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.contracts import CaseBundle
    from packages.research_runtime.models import ResearchRunContext
    from tests.database.knowledge_support_04a import sql
    from tests.disease_packs.support import anonymous_case, code
    from tests.tool_admission.support import fixture
    from tests.tool_admission.test_research import research

    f = fixture(tmp_path)
    req = research(f, tmp_path)
    repo = ResearchRunRepository(manage.test_target("runtime"))
    case = anonymous_case("GOAL:SYNTHETIC:REVIEW")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=req.scope.case_code,
                scope=case.scope.model_copy(update=dict(case_code=req.scope.case_code)),
                encounter_code=req.scope.encounter_code,
                patient_hash=req.scope.patient_hash,
                laterality=req.scope.laterality,
                artifacts=tuple(r.artifact for r in f.owners.records.values()),
            )
        ).model_dump_json()
    )
    repo.create_case_bundle(case, code("TASK"))
    context = seal(
        ResearchRunContext, request=req, synthetic=True, scope_checksum=digest(req.scope)
    )
    repo.create_research_run(context, "review-context")
    runtime = PlannerRepository(manage.test_target("runtime"))
    start, sha = runtime.origin(context.run_id)
    policy = RankingPolicy()
    b = ranked_binding(
        policy,
        case_code=req.scope.case_code,
        run_id=context.run_id,
        run_context_checksum=sha,
        started_at=start,
        deadline=start + timedelta(seconds=300),
        profile=Profile(max_actions=1),
    )
    source, owner, agent = Source(b), FakeOwner(b), Suggest()
    owner.choices = source.choices
    records, invocations = idempotent_preparation(owner)
    ranking = RankedActions(source, policy=policy, agent=agent)
    p = Planner(runtime=runtime, owner=owner, ranking=ranking)
    p.begin(b)

    def cmd(action):
        s = p.read(b.run_id)
        return Command(
            key=f"review-{s.steps}-{action}", expected_revision=s.revision, action=action
        )

    for action_ in ("load", "retrieve", "instantiate", "identify"):
        p.step(b.run_id, cmd(action_))
    chosen = cmd("select")
    event = p.step(b.run_id, chosen)
    assert runtime.read(b.run_id).selected.id == "B"
    assert p.step(b.run_id, chosen) == event and agent.calls == 1
    rows = sql(
        database.admin,
        "SELECT payload FROM audit.event WHERE run_id=%s "
        "AND correlation_code='SOURCE:PLANNER_CONTROL' ORDER BY event_sequence",
        (b.run_id,),
    )
    assert rows[-1][0]["state_checksum"] == digest(runtime.read(b.run_id))
    prepare = cmd("prepare")
    original = inject_save_failure(runtime, "after_reservation")
    with pytest.raises(ConnectionError):
        p.step(b.run_id, prepare)
    runtime.save = original
    saved = runtime.read(b.run_id)
    assert saved.pending.reserved and saved.reserved == 1 and saved.phase == Phase.PREPARE
    p = Planner(runtime=runtime, owner=owner, ranking=ranking)
    # A second fault window: owner preparation completes, confirmation fails.
    original = inject_save_failure(runtime, "before_confirmation")
    with pytest.raises(ConnectionError):
        p.step(b.run_id, prepare)
    runtime.save = original
    assert runtime.read(b.run_id) == saved and len(records) == 1
    p = Planner(
        runtime=PlannerRepository(manage.test_target("runtime")), owner=owner, ranking=ranking
    )
    resumed = p.step(b.run_id, prepare)
    assert len(records) == 1 and len(invocations) == 2
    state = runtime.read(b.run_id)
    rows = sql(
        database.admin,
        "SELECT payload FROM audit.event WHERE run_id=%s "
        "AND correlation_code='SOURCE:PLANNER_CONTROL' ORDER BY event_sequence",
        (b.run_id,),
    )
    assert rows[-1][0]["state_checksum"] == digest(state)
    assert sql(database.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0] == 0
    assert owner.calls.count("execute") == 0 and agent.calls == 1
    print(
        dict(
            pg_roundtrip="B_selected_and_Audit_matched",
            phase=state.phase.value,
            reason=resumed.outcome.reason,
            reserved=state.reserved,
            prepares=owner.calls.count("prepare"),
            executes=owner.calls.count("execute"),
        )
    )
    assert resumed.after == Phase.EXECUTE, resumed.outcome.reason

    assert state.selected.id == "B" and state.reserved == 1 and not state.pending
    assert rows[-1][0]["event"] == resumed.model_dump(mode="json")
    assert rows[-1][0]["binding_checksum"] == digest(b)
    assert len(rows) == state.revision + 1
    assert sum(1 for row in rows if row[0]["pending"] and row[0]["pending"]["reserved"]) == 1
    # A lost final response is answered from the confirmed event, even on another instance.
    p = Planner(
        runtime=PlannerRepository(manage.test_target("runtime")), owner=owner, ranking=ranking
    )
    assert p.step(b.run_id, prepare) == resumed
    assert runtime.read(b.run_id) == state and len(invocations) == 2 and agent.calls == 1
    assert sql(
        database.admin,
        "SELECT count(*) FROM audit.event WHERE run_id=%s "
        "AND correlation_code='SOURCE:PLANNER_CONTROL'",
        (b.run_id,),
    ) == [(len(rows),)]
