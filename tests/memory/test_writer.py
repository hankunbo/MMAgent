"""10B only: real owned PG, original synthetic Runtime/Planner/acceptance ports."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.planner.service import Planner
from apps.memory_execution_sources import ExecutionSources
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.grounding.models import digest, seal
from packages.memory.writer_models import Segment, WriterRecord
from packages.memory_writer import MemoryWriter, WriterUnavailable
from packages.planner_control.models import Command, Outcome, RankingPolicy
from packages.research_runtime.models import ResearchRunContext
from packages.tool_admission.models import ResearchRequest
from tests.database.knowledge_support_04a import sql
from tests.memory.test_relationships import counts
from tests.planner.support import FakeOwner


def cmd(planner, rid, action):
    state = planner.read(rid)
    return Command(key=f"b-{state.steps}-{action}", action=action, expected_revision=state.revision)


@pytest.fixture(scope="module")
def installed(memory):
    m = memory
    runtime = ResearchLifecycleRepository(m.pg.target)
    from tests.counterfactual.support import candidate, run

    run(m.x, candidate(targets=(m.x.parent.policy.items[0].id,), id="writer-existing-cf"))
    planner, rid = m.x.h.planner, m.pg.binding.run_id
    if planner.read(rid).phase != "stopped":
        planner.step(rid, cmd(planner, rid, "stop"))
    sources = ExecutionSources(
        runtime=runtime,
        run_id=rid,
        applicability=("USE:SYNTHETIC",),
        planner=planner,
        research=m.owners,
    )
    m.registry.owners[rid] = sources
    writer = MemoryWriter(repository=m.writer, sources=sources)
    yield SimpleNamespace(m=m, runtime=runtime, sources=sources, writer=writer, rid=rid)
    evidence = os.environ.get("MMAGENT_10B_EVIDENCE_DIR")
    if evidence:
        out = Path(evidence) / "examples"
        out.mkdir(exist_ok=True)
        for name, key in (("normal", "normal"), ("failure", "no-graph")):
            rid = installed_run_ids[name] if name == "failure" else rid
            key = "writer:" + digest([rid, key])[7:] + ":episode"
            result = m.reader.read(key, historical=True)
            (out / (name + ".json")).write_text(result.model_dump_json(indent=2) + "\n")
        (out / "replay.json").write_text(
            json.dumps(
                writer.replay(
                    "writer:" + digest([sources.run_id, "normal"])[7:] + ":episode", historical=True
                ),
                indent=2,
            )
            + "\n"
        )


installed_run_ids = {}


def clone(x, tag, *, planner=False, owner_type=FakeOwner, ranking=None, profile=None):
    m = x.m
    old = x.runtime.read_research_run(x.rid)
    raw = old.request.model_dump(mode="json", exclude={"checksum"})
    raw["lock"]["run_id"] = "RUN:OID_WRITER_" + tag.upper()
    request = seal(ResearchRequest, **raw)
    context = seal(
        ResearchRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    x.runtime.create_research_run(context, "10b-" + tag)
    rid = context.run_id
    p = None
    if planner:
        start, sha = m.pg.documents.sessions.origin(rid)
        from datetime import timedelta

        run_profile = m.pg.binding.profile.model_copy(update=profile or {})
        b = m.pg.binding.model_copy(
            update=dict(
                run_id=rid,
                run_context_checksum=sha,
                started_at=start,
                deadline=start + timedelta(seconds=run_profile.seconds),
                profile=run_profile,
            )
        )
        owner = owner_type(b)
        p = Planner(runtime=m.pg.documents.sessions, owner=owner, ranking=ranking)
        p.begin(b)
    sources = ExecutionSources(
        runtime=x.runtime, run_id=rid, applicability=("USE:SYNTHETIC",), planner=p
    )
    m.registry.owners[rid] = sources
    return SimpleNamespace(
        context=context,
        rid=rid,
        sources=sources,
        planner=p,
        writer=MemoryWriter(repository=m.writer, sources=sources),
    )


def test_normal_atomic_episode_and_replay(installed):
    x, m = installed, installed.m
    before_run = x.sources.snapshot()
    before = counts(m)
    result = x.writer.write(run_id=x.rid, request_key="normal")
    after = counts(m)
    assert isinstance(result.record, WriterRecord)
    assert result.state == "candidate" and not result.eligible_for_planning
    d = result.record.detail
    assert d.initial_graph != d.final_graph
    assert d.initial_graph.missing_reason is None
    assert d.scores.missing_reason is None
    assert d.terminal.scope == "planner_control" and d.terminal.runtime_status == "CREATED"
    assert after["memory.record"] - before["memory.record"] == len(result.record.parents) + 1
    assert after["memory.dependency"] - before["memory.dependency"] == len(result.record.parents)
    assert after["memory.request"] - before["memory.request"] == 1
    assert after["audit.event"] - before["audit.event"] == 1
    assert x.writer.write(run_id=x.rid, request_key="normal") == result
    assert counts(m) == after and x.sources.snapshot() == before_run
    replay = x.writer.replay(result.record.key)
    assert len(replay["steps"]) == len(before_run["control"]["events"])
    assert [s["index"] for s in replay["steps"]] == list(range(d.trajectory_count))
    assert m.reader.read(result.record.key) == result
    assert x.writer.replay(result.record.key, historical=True)["current"] is False
    with m.writer._connect() as c:
        assert (
            c.execute("SELECT rolsuper FROM pg_roles WHERE rolname=session_user").fetchone()[
                "rolsuper"
            ]
            is False
        )


def test_pre_graph_failure_no_fact_reader(installed):
    x = installed
    f = clone(x, "pregraph")
    report = x.runtime.reject_preparation(f.context, "compile-failed", digest("synthetic-config"))
    before = x.runtime.read_memory_execution(f.rid)
    result = f.writer.write(run_id=f.rid, request_key="no-graph")
    installed_run_ids["failure"] = f.rid
    assert report["graph"] is None and report["score_checksum"] is None
    d = result.record.detail
    assert d.terminal.scope == "preparation_rejection"
    assert d.terminal.runtime_status == "FAILED_TECHNICAL"
    assert d.initial_graph.missing_reason == "no_graph_before_failure"
    assert d.scores.missing_reason == "no_accepted_research_scores"
    assert d.trajectory_count == 0 and not result.record.parents
    assert x.runtime.read_memory_execution(f.rid) == before
    assert f.writer.replay(result.record.key)["steps"] == []


@pytest.mark.parametrize("mode", ["cancel", "missing_tool", "unknown", "running"])
def test_original_control_termination_matrix(installed, mode):
    x = installed

    class Owner(FakeOwner):
        def invoke(self, state, pending, *, cancel):
            if mode == "missing_tool" and pending.command.action == "execute":
                return Outcome(status="unavailable", reason="tool_port_unavailable", progress=False)
            return super().invoke(state, pending, cancel=cancel)

    f = clone(x, mode, planner=True, owner_type=Owner)
    p = f.planner
    if mode == "cancel":
        p.step(f.rid, cmd(p, f.rid, "cancel"))
    elif mode != "running":
        if mode == "unknown":
            p.owner.failure = "execute"
        for action in (
            "load",
            "retrieve",
            "instantiate",
            "identify",
            "select",
            "prepare",
            "execute",
        ):
            p.step(f.rid, cmd(p, f.rid, action))
        if mode == "missing_tool":
            p.step(f.rid, cmd(p, f.rid, "stop"))
    before = counts(x.m)
    original = f.sources.snapshot()
    result = f.writer.write(run_id=f.rid, request_key=mode)
    if mode in {"unknown", "running"}:
        assert result["status"] == "not_ready" and result["saved"] is False
        assert counts(x.m) == before
    else:
        replay = f.writer.replay(result.record.key)
        assert replay["terminal"]["cancelled"] == (mode == "cancel")
        if mode == "missing_tool":
            assert any(s["reason"] == "tool_port_unavailable" for s in replay["steps"])
    assert f.sources.snapshot() == original


@pytest.mark.parametrize("failure", ["member", "audit"])
def test_atomic_interruption_and_same_key_recovery(installed, monkeypatch, failure):
    x, m = installed, installed.m
    before, original = counts(m), x.sources.snapshot()
    with monkeypatch.context() as patch:
        if failure == "audit":

            def fail(*args):
                raise RuntimeError("injected_audit_failure")

            patch.setattr(m.writer, "_audit", fail)
        else:
            insert, counter = m.writer._insert_record, [0]

            def fail(*args):
                counter[0] += 1
                if counter[0] == 2:
                    raise RuntimeError("injected_member_failure")
                return insert(*args)

            patch.setattr(m.writer, "_insert_record", fail)
        with pytest.raises(RuntimeError, match="injected"):
            x.writer.write(run_id=x.rid, request_key="retry-" + failure)
    assert counts(m) == before and x.sources.snapshot() == original
    result = x.writer.write(run_id=x.rid, request_key="retry-" + failure)
    after = counts(m)
    assert x.writer.write(run_id=x.rid, request_key="retry-" + failure) == result
    assert counts(m) == after


def test_object_error_is_not_withdrawal(installed, monkeypatch):
    x, m = installed, installed.m
    before = counts(m)

    def fail(*args):
        raise OSError("synthetic object unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(m.pg.store, "read_bounded", fail)
        with pytest.raises(WriterUnavailable, match="memory_writer_source_unavailable"):
            x.writer.write(run_id=x.rid, request_key="object-failure")
    assert counts(m) == before
    assert sql(m.db.admin, "SELECT count(*) FROM memory.withdrawal")[0][0] == 0


def test_same_key_changed_content_stable_conflict(installed):
    x = installed
    before = counts(x.m)
    for _ in range(2):
        with pytest.raises(ValueError, match="memory_key_content_conflict"):
            x.writer.write(
                run_id=x.rid,
                request_key="normal",
                supplemental=(("simulation", "writer-existing-cf", "counterfactual_report"),),
            )
    assert counts(x.m) == before
    result = x.writer.write(
        run_id=x.rid,
        request_key="with-cf",
        supplemental=(("simulation", "writer-existing-cf", "counterfactual_report"),),
    )
    assert len(result.record.detail.supplemental_members) == 1


def test_tampered_projection_scope_and_version_no_write(installed):
    x = installed
    values = x.writer.prepare(run_id=x.rid, request_key="tampered")
    before = counts(x.m)
    for field, value in (
        ("schema_version", "research-memory/9.0.0"),
        ("run_id", "RUN:OID_FOREIGN"),
    ):
        changed = values[-1].model_copy(update={field: value})
        with pytest.raises(ValueError):
            x.m.writer.create_episode("tampered", (*values[:-1], changed))
    first = values[0]
    assert isinstance(first.detail, Segment)
    forged = first.detail.steps[0].model_copy(update={"reason": "invented_success"})
    bad = first.model_copy(
        update={
            "detail": first.detail.model_copy(update={"steps": (forged, *first.detail.steps[1:])})
        }
    )
    with pytest.raises(ValueError):
        x.m.writer.create_episode("tampered", (bad, *values[1:]))
    assert counts(x.m) == before


def test_source_invalidation_and_historical_replay(installed):
    x = installed
    f = clone(x, "withdrawn")
    x.runtime.reject_preparation(f.context, "fail", digest("cfg"))
    result = f.writer.write(run_id=f.rid, request_key="withdrawn")
    source = result.record.sources[0]
    x.m.writer.withdraw("writer-withdraw", source, reason="source_withdrawn")
    with pytest.raises(ValueError, match="memory_parent_invalid|memory_source_withdrawn"):
        f.writer.replay(result.record.key)
    assert f.writer.replay(result.record.key, historical=True)["current"] is False


def test_agent_advice_not_adopted_saved_legal_set(installed):
    from packages.planner_control.actions import RankedActions
    from tests.planner.test_actions import Source, Suggest

    x = installed
    f = clone(x, "advice", planner=True)
    p = f.planner
    b = p.read(f.rid).binding
    source = Source(b)
    p.owner.choices = source.choices
    agent = Suggest(lambda s, r: r.model_copy(update={"ranked_ids": ("not-a-legal-action",)}))
    p.ranking = RankedActions(source, policy=RankingPolicy(), agent=agent)
    for action in ("load", "retrieve", "instantiate", "identify", "select", "stop"):
        p.step(f.rid, cmd(p, f.rid, action))
    selected = p.read(f.rid).events[4].outcome.decision
    assert selected.rejected_advice == ("invalid_advice",)
    before_calls = agent.calls
    result = f.writer.write(run_id=f.rid, request_key="advice")
    replay = f.writer.replay(result.record.key)
    assert replay["steps"][4]["selection"]["missing_reason"] is None
    assert replay["steps"][4]["legal_actions"]["missing_reason"] is None
    assert replay["steps"][4]["expert_advice"]["missing_reason"] == "expert_advice_not_recorded"
    assert agent.calls == before_calls


def test_real_audit_conflict_rolls_back_entire_aggregate(installed):
    from datetime import UTC, datetime

    from db import manage
    from db.audit.repository import AuditEvent, PostgresAuditWriter
    from db.memory.repository import MemoryUnavailable
    from db.runtime.repository.case_pipeline import identity

    x, m = installed, installed.m
    manage.bootstrap_audit_principals(
        m.db.admin, manage.test_target("audit_writer").user, manage.test_target("audit_reader").user
    )
    audit = PostgresAuditWriter(manage.test_target("audit_writer"))
    audit.append_event(
        AuditEvent(
            event_id=identity("MESSAGE", "memory:episode:audit-conflict"),
            event_type="OFFLINE_REVIEW_RECORDED",
            actor="MEMORY",
            correlation_code="SOURCE:MEMORY",
            idempotency_key=identity("TASK", "memory:episode:audit-conflict"),
            occurred_at=datetime.now(UTC),
            payload={"synthetic": "intentional_conflict"},
        )
    )
    before, original = counts(m), x.sources.snapshot()
    with pytest.raises(MemoryUnavailable, match="memory_database_unavailable"):
        x.writer.write(run_id=x.rid, request_key="audit-conflict")
    assert counts(m) == before and x.sources.snapshot() == original


def test_writer_permission_and_no_partial_public_create(installed):
    from db import manage
    from db.memory.repository import MemoryRepository

    x = installed
    values = x.writer.prepare(run_id=x.rid, request_key="permission")
    before = counts(x.m)
    for cap in ("reader", "reviewer"):
        repo = MemoryRepository(
            manage.test_target("memory_" + cap), capability=cap, sources=x.m.registry
        )
        with pytest.raises(ValueError, match="memory_writer_required"):
            repo.create_episode("permission", values)
    with pytest.raises(ValueError, match="memory_writer_batch_required"):
        x.m.writer.create(values[-1])
    assert counts(x.m) == before


def test_stopped_simulation_read_only_gate_and_drift(installed, monkeypatch):
    from packages.evidence_details.current_rejection import CurrentResearchRejected

    x = installed
    service = x.m.x.service
    before = counts(x.m)
    calls = len(x.m.x.h.planner.owner.calls)
    with pytest.raises(ValueError, match="simulation_control:stopped"):
        service.read(candidate_id="writer-existing-cf", historical=False)
    report = service.read_evidence(candidate_id="writer-existing-cf")
    assert report["schema_version"] == "counterfactual-report/1.1.0"
    assert counts(x.m) == before and len(x.m.x.h.planner.owner.calls) == calls

    def withdrawn():
        raise CurrentResearchRejected("source_withdrawn")

    with monkeypatch.context() as patch:
        patch.setattr(service, "reader", withdrawn)
        with pytest.raises(CurrentResearchRejected, match="source_withdrawn"):
            service.read_evidence(candidate_id="writer-existing-cf")
    assert service.read(candidate_id="writer-existing-cf", historical=True) == report


def test_missing_fact_port_cannot_admit_accepted_graph(installed, monkeypatch):
    x = installed
    before = counts(x.m)
    monkeypatch.setattr(x.sources, "research", None)
    result = x.writer.write(run_id=x.rid, request_key="no-fact-port")
    assert result["status"] == "not_ready"
    assert result["reason"] == "memory_fact_reader_required"
    assert counts(x.m) == before


def test_completion_request_required_even_for_historical_read(installed):
    from tests.database.knowledge_support_04a import connect

    x = installed
    result = x.writer.write(run_id=x.rid, request_key="completion-guard")
    # Administrator is used only for corruption injection, never positive business writes.
    with connect(x.m.db.admin) as c:
        c.execute("DELETE FROM memory.request WHERE key='episode:completion-guard'")
    for historical in (True, False):
        with pytest.raises(ValueError, match="memory_aggregate_incomplete"):
            x.m.reader.read(result.record.key, historical=historical)


def test_unknown_after_commit_same_key_recovers_complete_aggregate(installed, monkeypatch):
    from db.memory.repository import MemoryUnavailable

    x, m = installed, installed.m
    original = x.sources.snapshot()
    write = m.writer._write
    with monkeypatch.context() as patch:

        def unavailable():
            raise OSError("synthetic post-commit unavailable")

        def committed_then_unavailable(*args):
            result = write(*args)
            patch.setattr(x.sources, "snapshot", unavailable)
            return result

        patch.setattr(m.writer, "_write", committed_then_unavailable)
        with pytest.raises(MemoryUnavailable, match="memory_source_unavailable"):
            x.writer.write(run_id=x.rid, request_key="lost-ack")
    before = counts(m)
    result = x.writer.write(run_id=x.rid, request_key="lost-ack")
    assert counts(m) == before and x.sources.snapshot() == original
    assert x.writer.replay(result.record.key)["current"] is True


def test_long_pg_original_events_are_segmented_without_source_key_json(installed):
    from packages.planner_control.models import Candidate

    x = installed

    class GapOwner(FakeOwner):
        def __init__(self, binding):
            super().__init__(binding)
            self.choices = tuple(
                Candidate(
                    id=f"branch-{n:02d}",
                    branch=f"branch-{n:02d}",
                    operation="observe",
                    and_ready=True,
                )
                for n in range(16)
            )

        def invoke(self, state, pending, *, cancel):
            if pending.command.action == "execute":
                return Outcome(status="gap", reason="missing_input", progress=False)
            return super().invoke(state, pending, cancel=cancel)

    f = clone(
        x,
        "long",
        planner=True,
        owner_type=GapOwner,
        profile=dict(max_steps=256, max_actions=32, max_branches=32, max_no_progress=32),
    )
    p = f.planner
    for action in ("load", "retrieve", "instantiate", "identify"):
        p.step(f.rid, cmd(p, f.rid, action))
    for _ in range(16):
        for action in ("select", "prepare", "execute", "fallback"):
            p.step(f.rid, cmd(p, f.rid, action))
    p.step(f.rid, cmd(p, f.rid, "stop"))
    original = p.replay(f.rid)
    assert len(original.events) == 69
    result = f.writer.write(run_id=f.rid, request_key="long")
    assert len(result.record.detail.segments) == 5
    assert len(result.record.parents) == 7
    replay = f.writer.replay(result.record.key)
    assert [s["command_key"] for s in replay["steps"]] == [e.command.key for e in original.events]
    assert all(s.key == f.rid for s in result.record.sources)
    assert p.replay(f.rid) == original
