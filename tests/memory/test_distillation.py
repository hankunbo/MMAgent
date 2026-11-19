"""10C only: original 09/10B ports, one owned PostgreSQL, candidate content and failures."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.memory_execution_sources import ExecutionSources
from apps.memory_sources import ResearchSources
from db.memory.repository import MemoryUnavailable
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.counterfactual.service import CounterfactualSimulation
from packages.grounding.models import canonical, digest
from packages.memory.distillation_models import AbstractData, PairData, RecoveryData
from packages.memory.models import Dependency, read_formal
from packages.memory.writer_models import parse_record
from packages.memory_distillation import MemoryDistiller
from packages.memory_writer import MemoryWriter
from tests.counterfactual.support import candidate, fixture
from tests.database.knowledge_support_04a import sql
from tests.memory.support import second_case
from tests.memory.test_memory import record
from tests.memory.test_relationships import counts
from tests.memory.test_writer import clone


def install(m, pg, x):
    research = ResearchSources(
        fact_reader=x.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=x.service,
        documents=pg.documents,
    )
    runtime = ResearchLifecycleRepository(m.pg.target)
    sources = ExecutionSources(
        runtime=runtime,
        run_id=pg.binding.run_id,
        applicability=("USE:SYNTHETIC",),
        planner=x.h.planner,
        research=research,
    )
    m.registry.owners[pg.binding.run_id] = sources
    return sources, MemoryWriter(repository=m.writer, sources=sources)


@pytest.fixture(scope="module")
def distilled(memory):
    m = memory
    alt_pg = second_case(m.pg, same_case=True)
    alt = fixture(pg=alt_pg)
    alt_sources, alt_writer = install(m, alt_pg, alt)
    x = m.x
    targets = tuple(i.id for i in x.parent.policy.items)
    policy = x.policy.model_copy(
        update=dict(
            allowed={
                "remove_evidence": targets,
                "replace_tool": targets,
                "select_check": targets,
            },
            alternatives={"alt": alt.fact.checksum},
        )
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=m.pg.documents,
        policy=policy,
        alternative_readers={"alt": alt.reader},
    )
    plans = (
        candidate(targets=targets, id="remove"),
        candidate("replace_tool", targets=targets[:1], id="missing-tool"),
        candidate("select_check", targets=targets[:1], id="check", alternative="alt"),
    )
    x.service.declare(plans)
    reports = {c.id: x.service.run(c)["report"] for c in plans}
    # Original recovery produces a bounded missing-input report; no invented success.
    from packages.planner_control.actions import RankedActions
    from packages.validation_recovery import (
        BranchRoute,
        Failure,
        NearestLegalActions,
        RecoveryPolicy,
        ValidationRecovery,
    )
    from tests.agent_provider.support import ref

    recovery_policy = RecoveryPolicy(
        routes=(BranchRoute(branch="A", forks=("root",)), BranchRoute(branch="B", forks=("root",))),
        source=next(r for r in m.pg.binding.locks if r.role == "rule"),
    )
    x.h.planner.ranking = RankedActions(NearestLegalActions(x.f.source, recovery_policy))
    for action in ("retrieve", "instantiate", "identify"):
        x.h.planner.step(m.pg.binding.run_id, x.h.command(action))
    failure = Failure(
        category="missing_input",
        target="GOAL:SYNTHETIC",
        reason="synthetic_missing",
        sources=(ref("observation", "OBS:SYNTHETIC"),),
    )
    recovery = ValidationRecovery(
        planner=x.h.planner,
        review=x.h.service,
        documents=m.pg.documents,
        policy=recovery_policy,
        failure_reader=lambda state: failure,
        summary_reader=x.reader.path_reader,
    )
    recovery.resolve(m.pg.binding.run_id, key="10c-recovery")
    sources, writer = install(m, m.pg, x)
    sources.research.recovery = recovery
    if x.h.planner.read(m.pg.binding.run_id).phase != "stopped":
        x.h.planner.step(m.pg.binding.run_id, x.h.command("stop"))
    alt.h.planner.step(alt_pg.binding.run_id, alt.h.command("stop"))
    episode = writer.write(
        run_id=m.pg.binding.run_id,
        request_key="10c-original",
        supplemental=(
            *(("simulation", c.id, "counterfactual_report") for c in plans),
            ("recovery", "10c-recovery", "recovery-stop-report"),
        ),
    )
    alternative_episode = alt_writer.write(
        run_id=alt_pg.binding.run_id, request_key="10c-alternative"
    )
    host = SimpleNamespace(m=m, runtime=sources.runtime, rid=m.pg.binding.run_id)
    failed = clone(host, "10c-failed")
    sources.runtime.reject_preparation(failed.context, "10c-compile-failed", digest("synthetic"))
    failed_episode = failed.writer.write(run_id=failed.rid, request_key="10c-failure")
    distiller = MemoryDistiller(repository=m.writer)
    before = (
        sources.snapshot(),
        alt_sources.snapshot(),
        counts(m),
        list(x.h.planner.owner.calls),
        list(alt.h.planner.owner.calls),
        x.f.budget.records(),
        alt.f.budget.records(),
    )
    prepared = distiller.prepare(request_key="10c-content", episode_keys=(episode.record.key,))
    saved = m.writer.create_distillation("10c-content", prepared)
    yield SimpleNamespace(
        m=m,
        x=x,
        alt=alt,
        sources=sources,
        writer=writer,
        episode=episode,
        alternate=alternative_episode,
        failed=failed_episode,
        distiller=distiller,
        prepared=prepared,
        saved=saved,
        reports=reports,
        before=before,
        alt_sources=alt_sources,
        recovery=recovery,
    )
    evidence = os.environ.get("MMAGENT_10C_EVIDENCE_DIR")
    if evidence:
        out = Path(evidence) / "examples"
        out.mkdir(parents=True, exist_ok=True)
        for value in prepared:
            (out / (value.kind + "-" + value.key[-8:] + ".json")).write_text(
                m.reader.read(value.key, historical=True).model_dump_json(indent=2) + "\n"
            )
        (Path(evidence) / "owned-lease.json").write_text(
            json.dumps(
                {"root": str(m.db.root) if hasattr(m.db, "root") else "see targeted lease log"}
            )
        )


def test_original_reports_content_null_removal_and_recovery(distilled):
    x = distilled
    pairs = [r.detail for r in x.prepared if isinstance(r.detail, PairData)]
    assert len(pairs) == 3
    for pair in pairs:
        original = x.reports[pair.report_source.key]
        assert pair.classification == original["classification"]
        for field in ("baseline", "simulated", "comparisons", "alternatives"):
            assert getattr(pair, field) == original[field]
        assert pair.semantics_version == "counterfactual-semantics/1.1.0"
    removal = next(p for p in pairs if p.report_source.key == "remove")
    assert removal.classification == "necessary"
    assert removal.comparisons["score"]["simulated"] is None
    assert removal.comparisons["score"]["delta"] is None
    assert all(
        c.operation == "remove" and c.removed and c.simulated_value is None for c in removal.changes
    )
    missing = next(p for p in pairs if p.report_source.key == "missing-tool")
    assert missing.classification == "indeterminate" and missing.simulated is None
    assert missing.changes[0].missing_reason == "missing_alternative_result"
    recovery = next(r.detail for r in x.prepared if isinstance(r.detail, RecoveryData))
    assert recovery.failure["category"] == "missing_input"
    assert recovery.recovery_success == "not_inferred_from_report"
    a = x.saved.record.detail
    assert isinstance(a, AbstractData) and a.case_count == a.run_count == 1
    assert a.counterexamples and a.independent_patient_count is None
    assert a.generalization == "unestablished_candidate_only" and not a.advice.executable
    assert x.saved.state == "candidate" and not x.saved.eligible_for_planning
    assert x.before[0] == x.sources.snapshot() and x.before[1] == x.alt_sources.snapshot()
    assert x.before[3:5] == (x.x.h.planner.owner.calls, x.alt.h.planner.owner.calls)
    assert x.before[5:] == (x.x.f.budget.records(), x.alt.f.budget.records())


def test_paired_binding_precise_alternative_sources(distilled):
    x = distilled
    pair = next(
        r
        for r in x.prepared
        if isinstance(r.detail, PairData) and r.detail.report_source.key == "check"
    )
    comparison = pair.detail.alternatives["alt"]
    assert comparison["before"]["case_code"] == comparison["after"]["case_code"]
    assert comparison["before"]["input_checksum"] == comparison["after"]["input_checksum"]
    assert comparison["before"]["run_id"] != comparison["after"]["run_id"]
    assert comparison["attribution"] == "no_single_tool_or_causal_attribution"
    alt_sources = [s for s in pair.sources if s.run_id == x.alternate.record.run_id]
    assert {s.kind for s in alt_sources} == {"records", "graph", "scores", "receipt"}
    assert pair.parents == (Dependency(key=x.episode.record.key, checksum=x.episode.checksum),)


def test_duplicate_case_and_missing_failed_episode_retained(distilled):
    x = distilled
    # Same Episode repeated and same case in two runs must not inflate case support.
    a = x.distiller.prepare(
        request_key="dedup",
        episode_keys=(x.episode.record.key, x.episode.record.key, x.alternate.record.key),
    )[-1]
    assert a.detail.case_count == 1 and a.detail.run_count == 2
    assert len(a.detail.supporting_episodes) == 2
    f = x.distiller.write(request_key="failed-retained", episode_keys=(x.failed.record.key,))
    assert f.record.detail.case_count == 1 and not f.record.detail.experiences
    assert {v["field"] for v in f.record.detail.missing} == {"simulation", "recovery"}
    assert f.record.detail.contexts[0].scores.missing_reason == "no_accepted_research_scores"
    assert f.record.detail.counterexamples[0]["runtime_status"] == "FAILED_TECHNICAL"


def test_direct_repository_rejects_omission_forged_content_and_scope(distilled):
    x = distilled
    before = counts(x.m)
    values = x.prepared
    for change in (
        {"sources": values[0].sources[:-1]},
        {"parents": ()},
        {"run_id": x.alternate.record.run_id},
        {"detail": values[0].detail.model_copy(update={"classification": "context_redundant"})},
    ):
        bad = values[0].model_copy(update=change)
        with pytest.raises(ValueError, match="memory_"):
            x.m.writer.create_distillation("10c-content", (bad, *values[1:]))
        assert counts(x.m) == before
    with pytest.raises(ValueError, match="memory_distillation_binding"):
        x.m.writer.create_distillation("10c-content", values[1:])
    with pytest.raises(ValueError, match="memory_distillation_batch_required"):
        x.m.writer.create(values[0])
    assert counts(x.m) == before


def test_same_key_recovery_conflict_and_legacy_bytes(distilled, monkeypatch):
    x = distilled
    before = counts(x.m)
    assert x.m.writer.create_distillation("10c-content", x.prepared) == x.saved
    assert counts(x.m) == before
    conflicting = x.distiller.prepare(
        request_key="10c-content", episode_keys=(x.failed.record.key,)
    )
    with pytest.raises(ValueError, match="memory_key_content_conflict"):
        x.m.writer.create_distillation("10c-content", conflicting)
    assert counts(x.m) == before
    assert canonical(x.m.reader.read(x.saved.record.key, historical=True).record) == canonical(
        x.saved.record
    )
    assert canonical(parse_record(x.episode.record.model_dump(mode="json"))) == canonical(
        x.episode.record
    )
    # Original 1.0 adapter has no aggregate port; R02 synchronization remains usable.
    with monkeypatch.context() as patch:
        patch.setitem(x.m.registry.owners, x.sources.run_id, x.sources.research)
        legacy = x.m.writer.create(record(x.m, "10c-legacy"))
        assert canonical(x.m.writer.synchronize(legacy.record.key).record) == canonical(
            legacy.record
        )
    for name in ("Episode", "TrajectoryStep", "CounterfactualPair"):
        raw = Path("packages/contracts/examples/v1/" + name + ".valid.json").read_bytes()
        assert read_formal(raw)[1] == raw


@pytest.mark.parametrize("fault", ["transaction", "audit"])
def test_transaction_and_audit_failure_zero_partial_write(distilled, monkeypatch, fault):
    x = distilled
    key = "rollback-" + fault
    prepared = x.distiller.prepare(request_key=key, episode_keys=(x.failed.record.key,))
    before = counts(x.m)
    with monkeypatch.context() as patch:
        original = x.m.writer._insert_record

        def fail(*args):
            if fault == "transaction":
                original(*args)
            raise RuntimeError("10c_injected_failure")

        patch.setattr(x.m.writer, "_audit" if fault == "audit" else "_insert_record", fail)
        with pytest.raises(RuntimeError, match="10c_injected_failure"):
            x.m.writer.create_distillation(key, prepared)
    assert counts(x.m) == before
    assert x.m.writer.create_distillation(key, prepared).state == "candidate"


def test_unknown_io_no_withdrawal_and_current_rebuild(distilled, monkeypatch):
    x = distilled
    before = counts(x.m)
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise OSError("synthetic object io")

        patch.setattr(x.m.pg.documents, "read", fail)
        with pytest.raises(MemoryUnavailable, match="memory_source_unavailable"):
            x.m.writer.synchronize(x.saved.record.key)
        assert x.m.reader.read(x.saved.record.key, historical=True).state == "candidate"
    assert counts(x.m) == before
    with monkeypatch.context() as patch:
        original = x.x.service.read_evidence

        def changed(*args, **kwargs):
            report = original(*args, **kwargs)
            return {**report, "classification": "forged"}

        patch.setattr(x.x.service, "read_evidence", changed)
        with pytest.raises(ValueError, match="memory_source_changed|memory_source_binding"):
            x.m.reader.read(x.saved.record.key)
    assert counts(x.m) == before


def test_r01_scope_and_parent_invalid_remain_enforced(distilled):
    x = distilled
    from packages.memory.models import validate_relationships

    p = x.episode.record
    child = x.alternate.record.model_copy(
        update={"parents": (Dependency(key=p.key, checksum=digest(p), relation="episode_member"),)}
    )
    with pytest.raises(ValueError, match="memory_relation_kind|memory_relation_scope"):
        validate_relationships(child, {p.key: p})
    child = p.model_copy(
        update={
            "run_id": x.alternate.record.run_id,
            "parents": (Dependency(key=p.key, checksum=digest(p), relation="revision_of"),),
        }
    )
    with pytest.raises(ValueError, match="memory_relation_scope"):
        validate_relationships(child, {p.key: p})
    # Separate failure branch: invalidation never contaminates the main fixture.
    saved = x.distiller.write(request_key="parent-invalid", episode_keys=(x.failed.record.key,))
    x.m.writer.withdraw("10c-parent-withdraw", x.failed.record.sources[0])
    assert x.m.reader.read(saved.record.key, historical=True).state == "deprecated"
    with pytest.raises(ValueError, match="memory_parent_invalid|memory_source_withdrawn"):
        x.m.reader.read(saved.record.key)


def test_z_alternative_withdrawal_propagates_and_history_survives(distilled, monkeypatch):
    x = distilled
    original_bytes = sql(
        x.m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key"
    )
    monkeypatch.setattr(x.alt.f.reader, "withdrawn", True)
    with pytest.raises(ValueError, match="source_withdrawn"):
        x.m.reader.read(x.saved.record.key)
    result = x.m.writer.synchronize(x.saved.record.key)
    assert result.state == "deprecated" and not result.current
    after = counts(x.m)
    assert x.m.writer.synchronize(x.saved.record.key) == result
    assert counts(x.m) == after
    assert (
        sql(x.m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
        == original_bytes
    )
