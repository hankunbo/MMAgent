"""05A isolated real PostgreSQL; centralized session lease and shared connection helper."""

import uuid

import psycopg
import pytest

from db import manage
from db.grounding.repository import GroundingConflictError, GroundingRepository
from packages.grounding import GroundingError, Mapping, State
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.retrieval.knowledge.models import RetrievalContext
from tests.database.knowledge_support_04a import connect, free_port, prepare_database, sql
from tests.grounding.test_models import reseal


@pytest.fixture(scope="module")
def database(database_lease):
    with database_lease.cluster(free_port()):
        db = prepare_database()
        manage.bootstrap_grounding_principals(
            db.admin, *(manage.test_target(c).user for c in manage.GROUNDING_CHANNELS)
        )
        manage.bootstrap_runtime_principal(db.admin, manage.test_target("runtime").user)
        yield db


@pytest.fixture
def setup(database):
    m, p, c, k, _ = fixture(code="BINDING:OID_" + uuid.uuid4().hex.upper())
    catalogs = SyntheticCatalogReader([c])
    writer = GroundingRepository(
        manage.test_target("grounding_writer"),
        capability="writer",
        policy=p,
        knowledge=k,
        catalogs=catalogs,
    )
    reader = GroundingRepository(
        manage.test_target("grounding_reader"),
        capability="reader",
        policy=p,
        knowledge=k,
        catalogs=catalogs,
    )
    return m, writer, reader, k, catalogs


def key():
    return uuid.uuid4().hex


def counts(db):
    return sql(
        db.admin,
        "SELECT (SELECT count(*) FROM grounding.mapping_version), "
        "(SELECT count(*) FROM grounding.request), (SELECT count(*) FROM audit.event), "
        "(SELECT count(*) FROM grounding.review)",
    )[0]


def test_migrate_empty_and_repeat(database):
    first = {d: manage.recorded_migrations(database.migration, d) for d in manage.DOMAINS}
    manage.migrate(database.migration)
    assert first == {d: manage.recorded_migrations(database.migration, d) for d in manage.DOMAINS}
    assert len(first["grounding"]) == 2 and len(first["runtime"]) == 14


def test_roundtrip_idempotency_review_and_withdrawal(database, setup):
    m, w, r, k, c = setup
    ctx = RetrievalContext()
    request = key()
    draft = w.register_draft(m, ctx, request_key=request)
    assert w.register_draft(m, ctx, request_key=request) == draft == r.get_exact(m.reference())
    with pytest.raises(GroundingError, match="mapping_not_admissible"):
        r.query_admissible((m.reference(),), ctx)
    changed = reseal(Mapping, {**m.model_dump(mode="json"), "version": "1.0.1"})
    with pytest.raises(GroundingConflictError):
        w.register_draft(changed, ctx, request_key=request)
    review_key = key()
    active = w.machine_review(m.reference(), ctx, request_key=review_key)
    assert w.machine_review(m.reference(), ctx, request_key=review_key) == active
    assert r.query_admissible((m.reference(),), ctx) == (active,)
    w.invalidate(m.reference(), state=State.WITHDRAWN, request_key=key())
    with pytest.raises(GroundingError):
        r.query_admissible((m.reference(),), ctx)
    history = r.replay(m.reference())
    assert (
        history.snapshot.record == m
        and len(history.reviews) == 2
        and not history.currently_admissible
    )
    with pytest.raises(GroundingError):
        w.machine_review(m.reference(), ctx, request_key=key())


@pytest.mark.parametrize("stage", ["audit", "request"])
def test_mid_transaction_rollback(database, setup, stage):
    m, w, r, k, c = setup
    before = counts(database)
    table = "audit.event" if stage == "audit" else "grounding.request"
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_grounding_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected transaction failure'; END $$",
    )
    sql(
        database.admin,
        f"CREATE TRIGGER fail_grounding BEFORE INSERT ON {table} "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_grounding_test()",
    )
    try:
        with pytest.raises(psycopg.Error):
            w.register_draft(m, RetrievalContext(), request_key=key())
        assert counts(database) == before
        assert sql(
            database.admin,
            "SELECT count(*) FROM grounding.mapping_identity WHERE code=%s",
            (m.code,),
        ) == [(0,)]
    finally:
        sql(database.admin, f"DROP TRIGGER fail_grounding ON {table}")
        sql(database.admin, "DROP FUNCTION public.fail_grounding_test()")


def test_sources_disabled_tools_and_new_versions(database, setup):
    m, w, r, k, c = setup
    ctx = RetrievalContext()
    w.register_draft(m, ctx, request_key=key())
    w.machine_review(m.reference(), ctx, request_key=key())
    k.withdrawn = True
    with pytest.raises(GroundingError):
        r.query_admissible((m.reference(),), ctx)
    assert r.replay(m.reference()).snapshot.record == m
    k.withdrawn = False
    c.disable(m.candidates[0].implementation_ref)
    with pytest.raises(GroundingError):
        r.query_admissible((m.reference(),), ctx)
    newer = reseal(Mapping, {**m.model_dump(mode="json"), "version": "1.1.0"})
    w.new_version(m.reference(), newer, ctx, request_key=key())
    assert r.get_exact(m.reference()).record == m
    with pytest.raises(GroundingConflictError):
        w.register_draft(m, ctx, request_key=key())


def test_permissions_sql_constraints_and_no_cross_domain_write(database, setup):
    m, w, r, k, c = setup
    ctx = RetrievalContext()
    w.register_draft(m, ctx, request_key=key())
    with pytest.raises(GroundingError, match="writer_required"):
        r.register_draft(m, ctx, request_key=key())
    w.machine_review(m.reference(), ctx, request_key=key())
    for role in ("grounding_writer", "grounding_reader"):
        with connect(manage.test_target(role)) as conn:
            conn.execute("SET ROLE mmagent_" + role)
            for statement in [
                "UPDATE knowledge.entity_version SET checksum=checksum",
                "UPDATE runtime.run SET status=status",
                "DELETE FROM audit.event",
                "CREATE TABLE grounding.unauthorized(x int)",
                "SET ROLE mmagent_grounding_owner",
            ]:
                with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
                    conn.execute(statement)
    with connect(manage.test_target("grounding_writer")) as conn:
        conn.execute("SET ROLE mmagent_grounding_writer")
        for statement, args in [
            ("INSERT INTO grounding.mapping_identity VALUES(%s)", (m.code,)),
            ("INSERT INTO grounding.mapping_identity VALUES(NULL)", ()),
            ("UPDATE grounding.mapping_state SET state=%s WHERE code=%s", ("validated", m.code)),
        ]:
            with pytest.raises(psycopg.Error), conn.transaction():
                conn.execute(statement, args)
    for table in ("mapping_version", "policy", "review", "request"):
        with pytest.raises(psycopg.errors.CheckViolation):
            sql(
                database.admin,
                f"UPDATE grounding.{table} SET "
                + (
                    "version=version"
                    if table in ("mapping_version", "policy")
                    else "payload=payload"
                    if table == "review"
                    else "checksum=checksum"
                ),
            )


def test_real_knowledge_reader_exact_release_and_withdrawal(database):
    from db.grounding.knowledge import PublicKnowledgeReader
    from db.knowledge.repository import KnowledgeNotFoundError
    from tests.database.knowledge_support_04a import graph_records, review_subject, seed_published

    prefix = "GROUNDING_" + key().upper()
    base = seed_published(database, prefix)
    graph = graph_records(base, prefix)
    database.editor.put_many(tuple(graph.values()))
    for name in (
        "concept",
        "target",
        "template",
        "goal",
        "node",
        "edge",
        "rule",
        "requirement",
        "stop_rule",
        "release",
    ):
        record = graph[name]
        review = review_subject(database, base, record)
        database.publisher.publish(
            record.reference(), review.reference(), base["policy"].reference()
        )
    m, p, c, _, _ = fixture()
    fields = {
        name + "_ref": graph[name].reference().model_dump(mode="json")
        for name in ("requirement", "template", "release")
    }
    fields.update(
        claim_refs=[base["claim"].reference().model_dump(mode="json")],
        source_refs=[r.model_dump(mode="json") for r in graph["requirement"].source_refs],
        applicability_refs=[
            r.model_dump(mode="json") for r in graph["requirement"].applicability_refs
        ],
    )
    m = reseal(Mapping, {**m.model_dump(mode="json"), **fields})
    adapter = PublicKnowledgeReader(database.reader, database.editor)
    ctx = RetrievalContext(
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
    )
    result = adapter.read(m, ctx, replay=False)
    assert result.requirement_ref == m.requirement_ref and result.mode == "published"
    database.publisher.withdraw(base["source_version"].reference(), "synthetic withdrawal test")
    with pytest.raises(KnowledgeNotFoundError):
        adapter.read(m, ctx, replay=False)
    assert adapter.read(m, ctx, replay=True).mode == "replay"


def test_concurrent_same_request_commits_once(database, setup):
    from concurrent.futures import ThreadPoolExecutor

    m, w, _, _, _ = setup
    request = key()
    before = counts(database)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(w.register_draft, m, RetrievalContext(), request_key=request)
            for _ in range(2)
        ]
        assert jobs[0].result() == jobs[1].result()
    after = counts(database)
    assert tuple(a - b for a, b in zip(after, before, strict=True)) == (1, 1, 1, 0)


def test_review_audit_failure_rolls_back_state_and_receipt(database, setup):
    m, w, _, _, _ = setup
    w.register_draft(m, RetrievalContext(), request_key=key())
    before = counts(database)
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_review_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'test'; END $$",
    )
    sql(
        database.admin,
        "CREATE TRIGGER fail_review BEFORE INSERT ON audit.event "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_review_test()",
    )
    try:
        with pytest.raises(psycopg.Error):
            w.machine_review(m.reference(), RetrievalContext(), request_key=key())
        assert counts(database) == before
        assert w.get_exact(m.reference()).state == State.DRAFT
    finally:
        sql(database.admin, "DROP TRIGGER fail_review ON audit.event")
        sql(database.admin, "DROP FUNCTION public.fail_review_test()")


def test_wrong_exact_version_checksum_and_illegal_copy_leave_no_writes(database, setup):
    from packages.grounding import Ref, validate

    m, w, r, _, _ = setup
    w.register_draft(m, RetrievalContext(), request_key=key())
    before = counts(database)
    for field, value in [("version", "0.1.0"), ("checksum", "sha256:" + "0" * 64)]:
        ref = validate(Ref, {**m.reference().model_dump(mode="json"), field: value})
        with pytest.raises(psycopg.errors.NoDataFound):
            r.get_exact(ref)
    for update in [{"patient_id": "CASE:OID_PATIENT1"}, {"claim_refs": ()}]:
        with pytest.raises(ValueError):
            w.register_draft(m.model_copy(update=update), RetrievalContext(), request_key=key())
    assert counts(database) == before
