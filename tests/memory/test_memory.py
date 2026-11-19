"""10A new boundary probes; no historical 08/09 test suites rerun."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from db import manage
from db.memory.repository import MemoryRepository, MemoryUnavailable
from packages.grounding.models import digest
from packages.memory.models import Dependency, EmbeddingIndex, Record, Review, read_formal
from tests.database.knowledge_support_04a import connect, sql


def record(m, key, *, kind="episode", parents=()):
    src = m.owners.reference("research_fact", m.x.reader.path_reader.key, "graph")
    return Record(
        key=key,
        kind=kind,
        case_code=src.case_code,
        run_id=src.run_id,
        sources=(src,),
        applicability=src.applicability,
        nature="source_fact",
        parents=parents,
    )


def test_pg_create_roundtrip_and_atomic_idempotency(memory):
    m = memory
    value = record(m, "roundtrip")
    one = m.writer.create(value)
    assert one == m.writer.create(value)
    assert m.reader.read(value.key) == one
    assert one.state == "candidate" and not one.eligible_for_planning
    m.writer.create(record(m, "graph-example", kind="graph_snapshot"))
    with pytest.raises(ValueError, match="memory_key_content_conflict"):
        m.writer.create(value.model_copy(update={"kind": "graph_snapshot"}))
    for row in m.x.reader().summary.original_report.values():
        assert row is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("eligible_for_retrieval", True),
        ("eligible_for_planning", True),
        ("state", "published"),
        ("purpose", "clinical"),
        ("schema_version", "research-memory/2.0.0"),
    ],
)
def test_constructed_revalidation(memory, field, value):
    raw = record(memory, "forged").model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ValueError):
        constructed = Record.model_construct(**raw)
        constructed.__dict__[field] = value
        memory.writer.create(constructed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_code", "CASE:OID_FOREIGN"),
        ("run_id", "RUN:OID_FOREIGN"),
        ("checksum", "sha256:" + "a" * 64),
        ("version", "case-evidence-graph/9.0.0"),
        ("nature", "agent_opinion"),
    ],
)
def test_source_tamper(memory, field, value):
    r = record(memory, "tamper-" + field)
    source = r.sources[0].model_copy(update={field: value})
    with pytest.raises(ValueError):
        memory.writer.create(r.model_copy(update={"sources": (source,)}))


def test_graph_control_and_versioned_fact_references(memory):
    m = memory
    for kind in ("records", "graph", "scores", "receipt"):
        s = m.owners.reference("research_fact", m.x.reader.path_reader.key, kind)
        assert m.owners.verify(s) == s
    e = m.x.h.planner.replay(m.pg.binding.run_id).events[0]
    s = m.owners.reference("planner_event", e.command.key, "control_event")
    r = record(m, "step").model_copy(update={"kind": "trajectory_step", "sources": (s,)})
    assert m.writer.create(r).record.sources == (s,)
    execute = next(
        e for e in m.x.h.planner.replay(m.pg.binding.run_id).events if e.command.action == "execute"
    )
    tool_source = m.owners.reference("planner_event", execute.command.key, "tool_execution_event")
    t = r.model_copy(update={"key": "tool", "kind": "tool_experience", "sources": (tool_source,)})
    assert m.writer.create(t).record.kind == "tool_experience"


def test_review_single_case_not_general_or_clinical(memory):
    m = memory
    r = m.writer.create(record(m, "reviewed-item"))
    review = Review(
        key="review-one",
        target=r.record.key,
        target_checksum=r.checksum,
        decision="approve",
        basis="engineering_review",
        reason="REVIEW:SYNTHETIC",
    )
    assert m.reviewer.review(review) == review
    assert m.reviewer.review(review) == review
    assert m.reader.read(r.record.key).state == "candidate"
    with pytest.raises(ValueError, match="memory_single_case_candidate"):
        m.reviewer.change_state("review-transition", r.record.key, "reviewed")
    with pytest.raises(ValueError, match="memory_publication_policy_unfrozen"):
        m.reviewer.change_state("publish", r.record.key, "published")
    assert m.reviewer.change_state("reject", r.record.key, "rejected")["state"] == "rejected"
    assert m.reader.read(r.record.key, historical=True).state == "rejected"
    with pytest.raises(ValueError, match="memory_parent_invalid"):
        m.reader.read(r.record.key)
    stopped = m.reviewer.change_state("reject-deprecate", r.record.key, "deprecated")
    assert stopped["previous"] == "rejected" and stopped["state"] == "deprecated"
    assert m.reviewer.change_state("reject-deprecate", r.record.key, "deprecated") == stopped


def test_index_metadata_only_and_binding(memory):
    m = memory
    r = m.writer.create(record(m, "index-item"))
    idx = EmbeddingIndex(
        key="index",
        target=r.record.key,
        target_checksum=r.checksum,
        content_checksum=r.checksum,
        model_code="MODEL:UNINSTANTIATED",
        model_version="metadata-only",
        dimension=16,
    )
    assert m.writer.index(idx) == idx
    assert m.writer.index(idx) == idx
    assert m.reader.read_index(idx.key) == idx
    with pytest.raises(ValueError, match="memory_index_binding"):
        m.writer.index(
            idx.model_copy(update={"key": "bad-index", "target_checksum": "sha256:" + "a" * 64})
        )
    with pytest.raises(ValidationError):
        EmbeddingIndex.model_validate({**idx.model_dump(), "vector": [0.0]})


def test_transactions_audit_failure_no_partial_record(memory):
    m = memory
    # A pre-existing conflicting Audit identity forces the last operation to fail.
    from datetime import UTC, datetime

    from db.audit.repository import AuditEvent, PostgresAuditWriter
    from db.runtime.repository.case_pipeline import identity

    manage.bootstrap_audit_principals(
        m.db.admin, manage.test_target("audit_writer").user, manage.test_target("audit_reader").user
    )
    audit = PostgresAuditWriter(manage.test_target("audit_writer"))
    audit.append_event(
        AuditEvent(
            event_id=identity("MESSAGE", "memory:create:rollback"),
            event_type="OFFLINE_REVIEW_RECORDED",
            actor="MEMORY",
            correlation_code="SOURCE:MEMORY",
            idempotency_key=identity("TASK", "memory:create:rollback"),
            occurred_at=datetime.now(UTC),
            payload={"test": "conflict"},
        )
    )
    with pytest.raises(MemoryUnavailable, match="memory_database_unavailable"):
        m.writer.create(
            record(
                m,
                "rollback",
                parents=(
                    Dependency(key="roundtrip", checksum=m.reader.read("roundtrip").checksum),
                ),
            )
        )
    for table, column in [
        ("record", "key"),
        ("source", "record_key"),
        ("status", "key"),
        ("dependency", "child"),
    ]:
        assert sql(
            m.db.admin, f"SELECT count(*) FROM memory.{table} WHERE {column}='rollback'"
        ) == [(0,)]
    assert sql(m.db.admin, "SELECT count(*) FROM memory.request WHERE key='create:rollback'") == [
        (0,)
    ]


@pytest.mark.parametrize("capability", ["writer", "reviewer", "reader"])
def test_independent_permissions(memory, capability):
    target = manage.test_target("memory_" + capability)
    with connect(target) as c:
        c.execute("SET ROLE mmagent_memory_" + capability)
        c.execute("SELECT 1 FROM memory.record LIMIT 1")
        import psycopg

        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("SET ROLE mmagent_memory_owner")
    for wrong in ("runtime", "bootstrap"):
        with pytest.raises(manage.DatabaseCommandError):
            MemoryRepository(
                manage.test_target(wrong), capability=capability, sources=memory.registry
            )


def test_runtime_cannot_write_memory_and_fk(memory):
    import psycopg

    with connect(manage.test_target("runtime")) as c:
        c.execute("SET ROLE mmagent_runtime_app")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("SELECT * FROM memory.record")
    r = record(memory, "fk").model_copy(update={"run_id": "RUN:OID_FOREIGN"})
    with connect(manage.test_target("memory_writer")) as c:
        c.execute("SET ROLE mmagent_memory_writer")
        from psycopg.types.json import Jsonb

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            c.execute(
                "INSERT INTO memory.record(key,kind,case_code,run_id,checksum,payload) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (r.key, r.kind, r.case_code, r.run_id, digest(r), Jsonb(r.model_dump(mode="json"))),
            )


@pytest.mark.parametrize("name", ["Episode", "TrajectoryStep", "CounterfactualPair"])
def test_formal_v1_exact_byte_dispatch(name):
    raw = Path("packages/contracts/examples/v1/" + name + ".valid.json").read_bytes()
    value, returned = read_formal(raw)
    assert returned == raw and value.contract_type == name
    assert json.loads(returned)["schema_version"] == "1.0"


def test_unknown_owner_error_not_withdrawal(memory, monkeypatch):
    m = memory
    r = m.writer.create(record(m, "unknown"))

    def broken(_):
        raise OSError("password=should-not-leak")

    monkeypatch.setattr(m.registry, "verify", broken)
    with pytest.raises(MemoryUnavailable, match="^memory_source_unavailable$"):
        m.writer.synchronize(r.record.key)
    assert m.reader.read(r.record.key, historical=True).state == "candidate"


def test_simulation_and_review_sources(memory):
    from tests.counterfactual.support import candidate, run

    m = memory
    result = run(m.x, candidate(targets=(m.x.parent.policy.items[0].id,)))
    assert result["status"] == "candidate"
    source = m.owners.reference("simulation", "one", "counterfactual_report")
    value = record(m, "simulation-pair").model_copy(
        update={"kind": "counterfactual_pair", "sources": (source,), "nature": "simulation"}
    )
    assert m.writer.create(value).record.nature == "simulation"
    key = m.x.h.args["event"].code + ":result"
    review_source = m.owners.reference("agent_review", key, "review_result")
    target = m.writer.create(record(m, "agent-opinion-target"))
    review = Review(
        key="agent-opinion",
        target=target.record.key,
        target_checksum=target.checksum,
        decision="approve",
        basis="agent_opinion",
        reason="REVIEW:FAKE",
        source=review_source,
    )
    assert m.reviewer.review(review) == review
    with pytest.raises(ValueError, match="memory_agent_opinion_not_engineering_review"):
        m.reviewer.review(
            review.model_copy(update={"key": "agent-disguised", "basis": "engineering_review"})
        )
    assert m.reader.read_review(review.key) == review
    assert m.reader.read(target.record.key).state == "candidate"
    assert source.version == "counterfactual-report/1.1.0"
    assert review_source.version == "research-review-result/2.0.0"


def test_two_original_cases_reviewed_still_not_published(memory):
    from apps.memory_sources import ResearchSources
    from tests.counterfactual.support import fixture
    from tests.memory.support import second_case

    m = memory
    pg = second_case(m.pg)
    x = fixture(pg=pg)
    owner = ResearchSources(
        fact_reader=x.reader,
        applicability=("USE:SYNTHETIC",),
        simulation=x.service,
        documents=pg.documents,
    )
    m.registry.owners[pg.binding.run_id] = owner
    source = owner.reference("research_fact", x.reader.path_reader.key, "graph")
    one = m.writer.create(record(m, "case-one"))
    other = Record(
        key="case-two",
        kind="episode",
        case_code=source.case_code,
        run_id=source.run_id,
        applicability=source.applicability,
        sources=(source,),
        nature="source_fact",
    )
    two = m.writer.create(other)
    abstract = Record(
        key="two-case-pattern",
        kind="abstract_pattern",
        case_code=one.record.case_code,
        run_id=one.record.run_id,
        applicability=one.record.applicability,
        sources=one.record.sources + two.record.sources,
        nature="abstract_candidate",
        parents=(
            Dependency(key=one.record.key, checksum=one.checksum),
            Dependency(key=two.record.key, checksum=two.checksum),
        ),
    )
    saved = m.writer.create(abstract)
    with pytest.raises(ValueError, match="memory_review_required"):
        m.reviewer.change_state("no-review", abstract.key, "reviewed")
    rv = Review(
        key="two-case-engineering",
        target=abstract.key,
        target_checksum=saved.checksum,
        decision="approve",
        basis="engineering_review",
        reason="REVIEW:ENGINEERING_ONLY",
    )
    m.reviewer.review(rv)
    assert m.reviewer.change_state("two-reviewed", abstract.key, "reviewed")["state"] == "reviewed"
    assert m.reviewer.change_state("two-reviewed", abstract.key, "reviewed")["state"] == "reviewed"
    value = m.reader.read(abstract.key)
    assert value.state == "reviewed" and not value.eligible_for_retrieval
    with pytest.raises(ValueError, match="memory_publication_policy_unfrozen"):
        m.reviewer.change_state("two-publish", abstract.key, "published")


def test_recovery_metadata_through_original_service(memory):
    from packages.planner_control.actions import RankedActions
    from packages.validation_recovery import (
        BranchRoute,
        Failure,
        NearestLegalActions,
        RecoveryPolicy,
        ValidationRecovery,
    )
    from tests.agent_provider.support import ref

    m = memory
    h = m.x.h
    run_id = m.pg.binding.run_id
    policy = RecoveryPolicy(
        routes=(BranchRoute(branch="A", forks=("root",)), BranchRoute(branch="B", forks=("root",))),
        source=next(r for r in m.pg.binding.locks if r.role == "rule"),
    )
    h.planner.ranking = RankedActions(NearestLegalActions(m.x.f.source, policy))
    for action in ("retrieve", "instantiate", "identify"):
        h.planner.step(run_id, h.command(action))
    failure = Failure(
        category="missing_input",
        target="GOAL:SYNTHETIC",
        reason="synthetic_missing",
        sources=(ref("observation", "OBS:SYNTHETIC"),),
    )
    service = ValidationRecovery(
        planner=h.planner,
        review=h.service,
        documents=m.pg.documents,
        policy=policy,
        failure_reader=lambda state: failure,
        summary_reader=m.x.reader.path_reader,
    )
    service.resolve(run_id, key="memory-recovery")
    m.owners.recovery = service
    source = m.owners.reference("recovery", "memory-recovery", "recovery-stop-report")
    r = record(m, "recovery-item").model_copy(
        update={"kind": "recovery_experience", "sources": (source,), "nature": "missing_evidence"}
    )
    saved = m.writer.create(r)
    assert saved.record.sources[0].kind == "recovery-stop-report"
    report = m.pg.documents.read("recovery:memory-recovery:report")[0]
    assert report["incident"]["failure"]["category"] == "missing_input"
    assert report["clinical_effect"] is None
    original = service.failure_reader
    service.failure_reader = lambda state: failure.model_copy(update={"reason": "changed_origin"})
    try:
        with pytest.raises(ValueError, match="memory_recovery_source_changed"):
            m.owners.verify(source)
    finally:
        service.failure_reader = original


def test_db_payload_alignment_and_no_self_granted_state(memory):
    import psycopg
    from psycopg.types.json import Jsonb

    m = memory
    r = record(m, "db-guard")
    with connect(manage.test_target("memory_writer")) as c:
        c.execute("SET ROLE mmagent_memory_writer")
        with pytest.raises(psycopg.errors.CheckViolation):
            c.execute(
                "INSERT INTO memory.record(key,kind,case_code,run_id,checksum,payload) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (
                    "wrong-key",
                    r.kind,
                    r.case_code,
                    r.run_id,
                    digest(r),
                    Jsonb(r.model_dump(mode="json")),
                ),
            )
    m.writer.create(r)
    with connect(manage.test_target("memory_writer")) as c:
        c.execute("SET ROLE mmagent_memory_writer")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE memory.status SET state='published' WHERE key='db-guard'")
    with connect(manage.test_target("memory_reviewer")) as c:
        c.execute("SET ROLE mmagent_memory_reviewer")
        with pytest.raises(
            psycopg.errors.RaiseException, match="memory_publication_policy_unfrozen"
        ):
            c.execute(
                "UPDATE memory.status SET state='published',revision=revision+1 WHERE "
                "key='db-guard'"
            )


def test_readback_revalidates_corrupt_payload(memory):
    m = memory
    r = m.writer.create(record(m, "corrupt-read"))
    # Administrative corruption is a negative probe only; no privileged positive writes.
    sql(
        m.db.admin,
        "UPDATE memory.record SET "
        "creator=session_user,payload=jsonb_set(payload,'{nature}','\"simulation\"') "
        "WHERE key='corrupt-read'",
    )
    with pytest.raises(ValueError):
        m.reader.read(r.record.key, historical=True)
    sql(
        m.db.admin,
        "UPDATE memory.record SET "
        "creator=session_user,payload=jsonb_set(payload,'{nature}','\"source_fact\"') "
        "WHERE key='corrupt-read'",
    )


def test_transitive_withdrawal_and_no_rehash_restoration(memory):
    m = memory
    a = m.writer.create(record(m, "ancestor"))
    b = m.writer.create(
        record(m, "child", parents=(Dependency(key=a.record.key, checksum=a.checksum),))
    )
    c = m.writer.create(
        record(
            m,
            "grandchild",
            parents=(Dependency(key=b.record.key, checksum=b.checksum, relation="revision_of"),),
        )
    )
    abstract = record(
        m, "abstract", parents=(Dependency(key=a.record.key, checksum=a.checksum),)
    ).model_copy(update={"kind": "abstract_pattern", "nature": "abstract_candidate"})
    m.writer.create(abstract)
    assert set(m.reader.descendants(a.record.key)) == {
        "ancestor",
        "child",
        "grandchild",
        "abstract",
    }
    source = a.record.sources[0]
    result = m.writer.withdraw("withdraw-one", source)
    assert m.writer.withdraw("withdraw-one", source) == result
    for key in (a.record.key, b.record.key, c.record.key, "abstract"):
        assert m.reader.read(key, historical=True).state == "deprecated"
        with pytest.raises(ValueError, match="memory_parent_invalid"):
            m.reader.read(key)
    with pytest.raises(ValueError, match="memory_source_withdrawn"):
        m.writer.create(record(m, "new-checksum-cannot-restore"))


def test_current_owner_withdrawal_unknown_not_reclassified(memory):
    m = memory
    m.x.f.reader.withdrawn = True
    try:
        from packages.evidence_details.current_rejection import current_rejection

        with pytest.raises(ValueError) as info:
            m.owners.reference("research_fact", m.x.reader.path_reader.key, "graph")
        assert current_rejection(info.value) is not None
        saved = m.writer.synchronize("roundtrip")
        assert saved.state == "deprecated"
    finally:
        m.x.f.reader.withdrawn = False
    with pytest.raises(ValueError, match="memory_parent_invalid"):
        m.reader.read("roundtrip")


def test_lifecycle_and_writer_review_permission(memory):
    from packages.memory.lifecycle import transition

    with pytest.raises(ValueError, match="memory_illegal_transition"):
        transition("candidate", "candidate")
    with pytest.raises(ValueError, match="memory_illegal_transition"):
        transition("deprecated", "candidate")
    with pytest.raises(ValueError, match="memory_publication_policy_unfrozen"):
        transition("reviewed", "published")
    assert transition("candidate", "reviewed") == "reviewed"
    assert transition("published", "deprecated") == "deprecated"
    with pytest.raises(ValueError, match="memory_reviewer_required"):
        memory.writer.review({})


def test_original_audit_reader_and_new_principal_boundaries(memory):
    import psycopg

    from db.audit.repository import PostgresAuditReader

    reader = PostgresAuditReader(manage.test_target("audit_reader"))
    events = reader.list_correlation("SOURCE:MEMORY")
    assert events and all(e.actor == "MEMORY" for e in events)
    creates = [e for e in events if e.payload.get("operation") == "create"]
    assert creates and all(e.payload["sources"] for e in creates)
    assert all(e.payload["sources"][0]["object_checksum"].startswith("sha256:") for e in creates)
    assert reader.list_correlation("SOURCE:PLANNER_CONTROL")
    for channel, statement in (
        ("memory_reader", "DELETE FROM memory.record"),
        ("memory_writer", "DELETE FROM memory.record"),
        ("memory_reviewer", "DELETE FROM memory.review"),
        ("memory_reviewer", "INSERT INTO memory.record(key) VALUES('forbidden')"),
    ):
        with connect(manage.test_target(channel)) as c:
            c.execute("SET ROLE mmagent_" + channel)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(statement)
