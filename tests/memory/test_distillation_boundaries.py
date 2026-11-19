"""Additional small PG cases: explicit cross-case abstraction and current corruption rejection."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.memory_execution_sources import ExecutionSources
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.grounding.models import digest
from packages.memory_distillation import MemoryDistiller
from packages.memory_writer import MemoryWriter
from tests.database.knowledge_support_04a import sql
from tests.memory.support import second_case
from tests.memory.test_relationships import counts
from tests.memory.test_writer import clone


@pytest.fixture(scope="module")
def boundary(memory):
    m = memory
    runtime = ResearchLifecycleRepository(m.pg.target)
    host = SimpleNamespace(m=m, runtime=runtime, rid=m.pg.binding.run_id)
    first = clone(host, "10c-boundary")
    runtime.reject_preparation(first.context, "10c-boundary-failed", digest("synthetic"))
    a = first.writer.write(run_id=first.rid, request_key="bound-first")
    pg = second_case(m.pg)
    rid = pg.binding.run_id
    runtime.reject_preparation(
        runtime.read_research_run(rid), "10c-second-failed", digest("synthetic")
    )
    owner = ExecutionSources(runtime=runtime, run_id=rid, applicability=("USE:SYNTHETIC",))
    m.registry.owners[rid] = owner
    b = MemoryWriter(repository=m.writer, sources=owner).write(
        run_id=rid, request_key="bound-second"
    )
    distiller = MemoryDistiller(repository=m.writer)
    before = sql(
        m.db.admin,
        "SELECT (SELECT count(*) FROM runtime.run),"
        "(SELECT count(*) FROM runtime.tool_attempt),"
        "(SELECT count(*) FROM runtime.tool_step_execution)",
    )
    saved = distiller.write(
        request_key="cross-case", episode_keys=(a.record.key, b.record.key, a.record.key)
    )
    after = sql(
        m.db.admin,
        "SELECT (SELECT count(*) FROM runtime.run),"
        "(SELECT count(*) FROM runtime.tool_attempt),"
        "(SELECT count(*) FROM runtime.tool_step_execution)",
    )
    assert before == after
    evidence = os.environ.get("MMAGENT_10C_EVIDENCE_DIR")
    if evidence:
        out = Path(evidence)
        (out / "start-counts.json").write_text(
            json.dumps(
                dict(
                    columns=["runs", "tool_attempts", "tool_steps"],
                    before=before,
                    after=after,
                    delta=[0, 0, 0],
                    provider_requests_added=0,
                    provider_basis="failed Episodes have no Provider port",
                ),
                indent=2,
            )
            + "\n"
        )
        (out / "examples").mkdir(exist_ok=True)
        (out / "examples/cross-case-failures.json").write_text(
            saved.model_dump_json(indent=2) + "\n"
        )
    return SimpleNamespace(m=m, saved=saved, distiller=distiller)


def test_explicit_cross_case_intersection_and_no_patient_independence(boundary):
    x = boundary
    r = x.saved.record
    assert r.detail.case_count == r.detail.run_count == 2
    assert len(r.detail.contexts) == 2 and len(r.detail.supporting_episodes) == 2
    assert r.detail.evidence_count == 2 and len(r.detail.counterexamples) == 2
    assert all(
        c.population is None and c.input_checksum is None and c.input_missing_reason
        for c in r.detail.contexts
    )
    assert r.detail.independent_patient_count is None
    assert r.applicability == ("USE:SYNTHETIC",)
    assert len({s.case_code for s in r.sources}) == 2
    assert x.saved.state == "candidate" and not x.saved.eligible_for_retrieval
    with pytest.raises(ValueError, match="memory_publication_policy_unfrozen"):
        x.m.reviewer.change_state("not-published", r.key, "published")


def test_current_rebuild_rejects_self_consistent_forged_history(boundary, monkeypatch):
    x = boundary
    key = "current-forgery"
    values = x.distiller.prepare(
        request_key=key, episode_keys=x.saved.record.detail.supporting_episodes
    )
    r = values[-1]
    bad = r.model_copy(
        update=dict(
            detail=r.detail.model_copy(
                update=dict(
                    advice=r.detail.advice.model_copy(update=dict(codes=("forged_advice",)))
                )
            )
        )
    )
    # Test-only admission fault, never a legitimate save path. Real SQL, immutable
    # creator, source/relations, request and Audit remain enabled under the writer.
    with monkeypatch.context() as patch:
        patch.setattr(x.m.writer, "prepare_distillation", lambda *args: (*values[:-1], bad))
        with pytest.raises(ValueError, match="memory_distillation_binding"):
            x.m.writer.create_distillation(key, (*values[:-1], bad))
    assert x.m.reader.read(r.key, historical=True).record == bad
    before = counts(x.m)
    with pytest.raises(ValueError, match="memory_distillation_binding"):
        x.m.reader.read(r.key)
    assert counts(x.m) == before


def test_10b_required_acceptance_set_still_enforced(boundary):
    m = boundary.m
    planner, rid = m.x.h.planner, m.pg.binding.run_id
    planner.step(rid, m.x.h.command("stop"))
    owner = ExecutionSources(
        runtime=ResearchLifecycleRepository(m.pg.target),
        run_id=rid,
        applicability=("USE:SYNTHETIC",),
        planner=planner,
        research=m.owners,
    )
    m.registry.owners[rid] = owner
    writer = MemoryWriter(repository=m.writer, sources=owner)
    values = writer.prepare(run_id=rid, request_key="r01-still-required")
    episode = values[-1]
    without_scores = episode.model_copy(
        update=dict(sources=tuple(s for s in episode.sources if s.kind != "scores"))
    )
    before = counts(m)
    with pytest.raises(ValueError, match="memory_accepted_sources_incomplete"):
        m.writer.create_episode("r01-still-required", (*values[:-1], without_scores))
    assert counts(m) == before
