"""10B-R01: public boundaries on the original unified owned PG and real facts."""

import pytest

from packages.memory_writer.projection import project
from tests.memory import test_writer
from tests.memory.test_relationships import counts

installed = test_writer.installed


@pytest.mark.parametrize("missing_port", [False, True])
def test_repository_cannot_omit_accepted_facts(installed, monkeypatch, missing_port):
    x = installed
    snapshot = x.sources.snapshot()
    assert snapshot["control"]["graph"]["version"] == "case-evidence-graph/2.0.0"
    key = "r01-omit-all-" + str(missing_port)
    records = project(snapshot, x.sources.snapshot_reference(snapshot), request_key=key)
    if missing_port:
        monkeypatch.setattr(x.sources, "research", None)
        result = x.writer.write(run_id=x.rid, request_key=key)
        assert result["reason"] == "memory_fact_reader_required" and not result["saved"]
    before = counts(x.m)
    code = "memory_fact_reader_required" if missing_port else "memory_accepted_sources_incomplete"
    with pytest.raises(ValueError, match="^" + code + "$"):
        x.m.writer.create_episode(key, records)
    assert counts(x.m) == before


@pytest.mark.parametrize("omitted", ["records", "graph", "scores", "receipt"])
def test_partial_set_rejected_at_repository_and_revalidation(installed, omitted):
    x = installed
    key = "r01-partial-" + omitted
    snapshot = x.sources.snapshot()
    complete = x.writer.prepare(run_id=x.rid, request_key=key)
    facts = tuple(
        s for s in complete[-1].sources if s.owner == "research_fact" and s.kind != omitted
    )
    values = project(
        snapshot, x.sources.snapshot_reference(snapshot), request_key=key, fact_sources=facts
    )
    before = counts(x.m)
    with pytest.raises(ValueError, match="^memory_accepted_sources_incomplete$"):
        x.m.writer.create_episode(key, values)
    # model_copy bypasses construction; original owner must still reject each envelope.
    for original in complete:
        if original.kind not in {"episode", "graph_snapshot"}:
            continue
        forged = original.model_copy(
            update={"sources": tuple(s for s in original.sources if s.kind != omitted)}
        )
        with pytest.raises(ValueError, match="^memory_accepted_sources_incomplete$"):
            x.sources.verify_record(forged)
    assert counts(x.m) == before


@pytest.mark.parametrize("omitted", ["records", "graph", "scores", "receipt", "all"])
def test_legacy_bad_aggregate_historical_only(installed, monkeypatch, omitted):
    x = installed
    key = "r01-legacy-" + omitted
    snapshot = x.sources.snapshot()
    complete = x.writer.prepare(run_id=x.rid, request_key=key)
    facts = tuple(
        s
        for s in complete[-1].sources
        if s.owner == "research_fact" and omitted not in ("all", s.kind)
    )
    values = project(
        snapshot, x.sources.snapshot_reference(snapshot), request_key=key, fact_sources=facts
    )
    # Isolated owned test PG ONLY: bypass the envelope guard to inject exactly the
    # old illegal aggregate; original per-source, SQL, request and Audit checks remain.
    # This is corruption setup, never evidence for a positive business write.
    with monkeypatch.context() as patch:
        patch.setattr(x.sources, "verify_record", lambda record: None)
        saved = x.m.writer.create_episode(key, values)
    before = counts(x.m)
    assert x.m.reader.read(saved.record.key, historical=True).record == values[-1]
    assert not x.writer.replay(saved.record.key, historical=True)["current"]
    for record in values:
        if record.kind in {"episode", "graph_snapshot"}:
            with pytest.raises(ValueError, match="^memory_accepted_sources_incomplete$"):
                x.m.reader.read(record.key)
    with pytest.raises(ValueError, match="^memory_accepted_sources_incomplete$"):
        x.writer.replay(saved.record.key)
    if omitted == "all":
        monkeypatch.setattr(x.sources, "research", None)
        with pytest.raises(ValueError, match="^memory_fact_reader_required$"):
            x.m.reader.read(saved.record.key)
    assert counts(x.m) == before


def test_complete_bytes_recovery_conflict_and_withdrawal(installed):
    from packages.grounding.models import canonical

    x = installed
    key = "r01-compatible"
    snapshot = x.sources.snapshot()
    batch = x.m.owners.fact_reader.path_reader.key
    facts = tuple(
        x.sources.reference("research_fact", batch, kind)
        for kind in ("records", "graph", "scores", "receipt")
    )
    old = project(
        snapshot, x.sources.snapshot_reference(snapshot), request_key=key, fact_sources=facts
    )
    assert [canonical(r) for r in x.writer.prepare(run_id=x.rid, request_key=key)] == [
        canonical(r) for r in old
    ]
    saved = x.writer.write(run_id=x.rid, request_key=key)
    before = counts(x.m)
    assert x.m.writer.create_episode(key, old) == saved
    assert x.writer.write(run_id=x.rid, request_key=key) == saved
    assert counts(x.m) == before
    with pytest.raises(ValueError, match="^memory_key_content_conflict$"):
        x.writer.write(
            run_id=x.rid,
            request_key=key,
            supplemental=(("simulation", "writer-existing-cf", "counterfactual_report"),),
        )
    assert counts(x.m) == before
    assert x.writer.replay(saved.record.key)["current"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_reader",
        "missing_path",
        "missing_acceptance",
        "missing_documents",
        "object_io",
        "database_io",
    ],
)
def test_dependency_vs_technical_failure(installed, monkeypatch, fault):
    from db.memory.repository import MemoryUnavailable
    from packages.memory_writer import WriterUnavailable

    x = installed
    values = x.writer.prepare(run_id=x.rid, request_key="r01-port-" + fault)
    before = counts(x.m)
    if fault == "missing_reader":
        monkeypatch.setattr(x.m.owners, "fact_reader", None)
        code = "memory_fact_reader_required"
    elif fault == "missing_path":
        monkeypatch.setattr(x.m.owners.fact_reader, "path_reader", None)
        code = "memory_source_port_unavailable"
    elif fault == "missing_acceptance":
        monkeypatch.setattr(x.m.owners.fact_reader.path_reader, "acceptance", None)
        code = "memory_source_port_unavailable"
    elif fault == "missing_documents":
        monkeypatch.setattr(x.m.owners.fact_reader.path_reader.acceptance, "documents", None)
        code = "memory_source_port_unavailable"
    else:

        def fail(*args, **kwargs):
            raise (
                OSError("synthetic owner I/O")
                if fault == "object_io"
                else RuntimeError("synthetic DB I/O")
            )

        monkeypatch.setattr(x.m.owners, "accepted_sources", fail)
        code = "memory_source_unavailable"
    if fault == "missing_reader":
        assert x.writer.write(run_id=x.rid, request_key="r01-port-" + fault)["reason"] == code
    elif fault in {"missing_path", "missing_acceptance", "missing_documents"}:
        with pytest.raises(ValueError, match="^" + code + "$"):
            x.writer.prepare(run_id=x.rid, request_key="r01-port-" + fault)
    else:
        with pytest.raises(WriterUnavailable, match="^memory_writer_source_unavailable$"):
            x.writer.write(run_id=x.rid, request_key="r01-port-" + fault)
    # Runtime-only segment leads to the common envelope check before individual facts.
    with pytest.raises(
        MemoryUnavailable if fault.endswith("io") else ValueError, match="^" + code + "$"
    ):
        x.m.writer.create_episode("r01-port-" + fault, values)
    assert counts(x.m) == before


def test_other_real_acceptance_batch_and_stale_objects_rejected(installed, monkeypatch):
    from agents.expert_agent.reporting import read_research_report
    from packages.grounding.models import digest

    x = installed
    reader = x.m.owners.fact_reader
    acceptance = reader.path_reader.acceptance
    other_key = "r01-other-real-batch"
    from tests.agent_provider.support import review_from_wire, submission

    # Another original Fake request on the current, not-yet-consumed parent.
    # Persist the real accepted objects, but do not install that new graph.
    parent = x.m.x.f.reader.current()
    x.m.x.f.provider.transport.replies.append(review_from_wire)
    candidate = x.m.x.f.ports.review.call(
        parent.material.scope, deadline=x.m.pg.binding.deadline, round_id="B"
    )
    submitted = submission(x.m.x.f.reader, candidate, x.m.x.f.budget)
    assert acceptance.submit(submitted, key=other_key)["status"] == "submitted_not_accepted"
    assert acceptance.accept(key=other_key).status == "accepted"
    report = read_research_report(acceptance, key=other_key, parent=parent)
    raw = acceptance.documents.read("accepted:" + other_key)
    complete = x.writer.prepare(run_id=x.rid, request_key="r01-wrong-batch")
    facts = tuple(s for s in complete[-1].sources if s.owner == "research_fact")
    stale = []
    for i, source in enumerate(facts):
        ref = acceptance.documents.reference(
            "accepted:" + other_key, i, raw[i], role=source.kind, version=source.version
        )
        stale.append(
            source.model_copy(
                update=dict(
                    key=other_key,
                    object_code=ref.code,
                    object_checksum=ref.checksum,
                    checksum=digest(report[source.kind]),
                    dependencies=tuple(
                        digest(["research_fact", x.rid, other_key, kind])
                        for kind in ("records", "graph", "scores", "receipt")
                    ),
                )
            )
        )
    before = counts(x.m)
    snapshot = x.sources.snapshot()
    for index in range(5):
        mixed = (
            tuple(stale)
            if index == 4
            else tuple(stale[i] if i == index else s for i, s in enumerate(facts))
        )
        values = project(
            snapshot,
            x.sources.snapshot_reference(snapshot),
            request_key="r01-wrong-batch",
            fact_sources=mixed,
        )
        with pytest.raises(ValueError, match="^memory_source_binding$"):
            x.m.writer.create_episode("r01-wrong-batch", values)
        with pytest.raises(ValueError, match="^memory_source_binding$"):
            x.sources.verify_record(values[-1])
    # Wrong installed batch also cannot become the prepare oracle.
    with monkeypatch.context() as patch:
        patch.setattr(reader.path_reader, "key", other_key)
        patch.setattr(reader.path_reader, "parent", parent)
        with pytest.raises(ValueError, match="^memory_source_binding$"):
            x.writer.prepare(run_id=x.rid, request_key="r01-wrong-install")
    assert counts(x.m) == before


def test_accepted_withdrawal_preserves_historical_bytes(installed):
    from packages.grounding.models import canonical

    x = installed
    saved = x.writer.write(run_id=x.rid, request_key="r01-withdraw")
    original = canonical(saved.record)
    source = next(s for s in saved.record.sources if s.kind == "scores")
    x.m.writer.withdraw("r01-withdraw-source", source, reason="source_withdrawn")
    with pytest.raises(ValueError, match="^memory_parent_invalid$|^memory_source_withdrawn$"):
        x.writer.replay(saved.record.key)
    assert canonical(x.m.reader.read(saved.record.key, historical=True).record) == original
    assert not x.writer.replay(saved.record.key, historical=True)["current"]


def test_legacy_v10_bytes_and_relation_guard(installed):
    from packages.grounding.models import canonical, digest
    from packages.memory.models import Dependency
    from tests.memory.test_memory import record

    x = installed
    parent = record(x.m, "r01-v10-parent", kind="graph_snapshot")
    saved = x.m.writer.create(parent)
    assert canonical(x.m.reader.read(parent.key).record) == canonical(parent)
    child = record(
        x.m,
        "r01-v10-child",
        parents=(Dependency(key=parent.key, checksum=digest(parent), relation="episode_member"),),
    )
    result = x.m.writer.create(child)
    before = counts(x.m)
    assert x.m.writer.create(child) == result
    assert canonical(x.m.reader.read(child.key, historical=True).record) == canonical(child)
    bad = child.model_copy(update=dict(key="r01-v10-bad", kind="graph_snapshot"))
    with pytest.raises(ValueError, match="^memory_relation_kind$"):
        x.m.writer.create(bad)
    assert counts(x.m) == before and saved.record == parent


def test_unaccepted_root_does_not_require_fact_reader(installed, monkeypatch):
    from tests.memory.test_writer import clone, cmd

    x = installed
    f = clone(x, "r01-root", planner=True)
    f.planner.step(f.rid, cmd(f.planner, f.rid, "stop"))
    assert f.sources.snapshot()["control"]["graph"]["version"] != "case-evidence-graph/2.0.0"
    f.sources.research = x.m.owners
    monkeypatch.setattr(x.m.owners, "fact_reader", None)
    result = f.writer.write(run_id=f.rid, request_key="r01-root")
    assert result.record.detail.scores.missing_reason == "no_accepted_research_scores"
    assert not any(s.owner == "research_fact" for s in result.record.sources)
    assert f.writer.replay(result.record.key)["current"]


def test_post_commit_owner_change_is_historical_not_compensated(installed, monkeypatch):
    from packages.evidence_details.current_rejection import CurrentResearchRejected

    x = installed
    key = "r01-post-commit"
    values = x.writer.prepare(run_id=x.rid, request_key=key)
    before = counts(x.m)
    write = x.m.writer._write
    with monkeypatch.context() as patch:

        def changed(*args):
            raise CurrentResearchRejected("memory_source_changed")

        def commit_then_change(*args):
            result = write(*args)
            patch.setattr(x.sources, "required_fact_sources", changed)
            return result

        patch.setattr(x.m.writer, "_write", commit_then_change)
        with pytest.raises(CurrentResearchRejected, match="^memory_source_changed$"):
            x.m.writer.create_episode(key, values)
        after = counts(x.m)
        assert after["memory.record"] - before["memory.record"] == len(values)
        assert after["audit.event"] - before["audit.event"] == 1
        assert not x.writer.replay(values[-1].key, historical=True)["current"]
        with pytest.raises(CurrentResearchRejected, match="^memory_source_changed$"):
            x.writer.replay(values[-1].key)
        assert counts(x.m) == after
    # The injected transient owner fault is gone; exact-key retry adds no Audit.
    assert x.m.writer.create_episode(key, values).record == values[-1]
    assert counts(x.m) == after
