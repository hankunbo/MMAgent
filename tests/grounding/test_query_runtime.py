"""05B real PG binding, Published Knowledge, stale results and atomic lock failures."""

import json

import psycopg
import pytest

from db import manage
from db.knowledge.models import KnowledgeReference
from db.knowledge.repository import KnowledgeNotFoundError
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding import CatalogSnapshot, GroundingError, Mapping, State, digest
from packages.grounding.query_models import QueryResult
from tests.database.knowledge_support_04a import sql
from tests.disease_packs import support
from tests.grounding.query_support import metadata, request_for, updated
from tests.grounding.test_database import database as database
from tests.grounding.test_database import key
from tests.grounding.test_models import reseal
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated


def scoped_mapping(integrated):
    data, old, writer, catalogs = integrated

    def right_eye(value):
        if isinstance(value, dict):
            return {k: "RIGHT" if k == "laterality" else right_eye(v) for k, v in value.items()}
        if isinstance(value, list):
            return [right_eye(v) for v in value]
        return value

    catalog = reseal(
        CatalogSnapshot,
        {
            **right_eye(catalogs.read_snapshot(old.catalog_ref).model_dump(mode="json")),
            "version": "1.2.0",
        },
    )
    catalogs.add(catalog)
    mapping = reseal(
        Mapping,
        {
            **right_eye(old.model_dump(mode="json")),
            "version": "1.2.0",
            "catalog_ref": catalog.reference(),
        },
    )
    writer.new_version(old.reference(), mapping, support.context(), request_key=key())
    writer.machine_review(mapping.reference(), support.context(), request_key=key())
    return data, mapping, writer, catalogs


def query_request(database, mapping, writer, lock):
    row = sql(
        database.admin,
        "SELECT r.case_code,c.patient_hash,e.encounter_code,c.laterality "
        "FROM runtime.run r JOIN runtime.case_record c USING(case_code) "
        "JOIN runtime.encounter e USING(case_code) WHERE r.run_id=%s",
        (lock.run_id,),
    )[0]
    scope = dict(
        zip(("case_code", "patient_hash", "encounter_code", "laterality"), row, strict=True)
    )
    q = request_for(
        mapping,
        writer.policy,
        lock=lock,
        scope=scope,
        target_node_code=writer.knowledge.read(
            mapping, support.context(), replay=False
        ).target_node_code,
        context=support.context(),
    )
    return updated(
        q,
        artifacts=[
            metadata(q, i.constraint, str(n).zfill(4))
            for n, i in enumerate(mapping.candidates[0].inputs)
            if i.required
        ],
    )


def counts(database):
    return sql(
        database.admin,
        "SELECT (SELECT count(*) FROM runtime.run_grounding_lock),"
        "(SELECT count(*) FROM audit.event),(SELECT count(*) FROM grounding.request)",
    )[0]


def test_new_and_old_runs_path_admission_results_and_failures(database, integrated, tmp_path):
    data, mapping, writer, catalogs = scoped_mapping(integrated)
    args = create_run(database, data, mapping, writer)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    service = GroundingQueryService(locks, writer)
    lock = locks.bind(**args)
    q = query_request(database, mapping, writer, lock)
    initial_counts = counts(database)
    result = service.query(q)
    assert len(result.candidates) == 2 and result.status == "candidates"
    assert service.validate_result(q, result) == result
    assert counts(database) == initial_counts  # Queries/validation never write Runtime/Audit.
    assert locks.bind(**args) == lock and counts(database) == initial_counts
    # Same run idempotency identity with different locked content cannot replace the lock.
    with pytest.raises(psycopg.errors.UniqueViolation):
        locks.bind(**{**args, "graph_checksum": digest("different")})
    assert counts(database) == initial_counts and locks.read(args["run_id"]) == lock
    for changes in (
        {"scope": {**q.scope.model_dump(mode="json"), "patient_hash": digest("different")}},
        {"scope": {**q.scope.model_dump(mode="json"), "encounter_code": "ENCOUNTER:OID_OTHER001"}},
        {"target_node_code": "EVIDENCE:SYNTHETIC:WRONG"},
    ):
        with pytest.raises(ValueError):
            service.query(updated(q, artifacts=[], **changes))
    with pytest.raises(ValueError):
        service.validate_result(q, result.model_copy(update={"candidates": ()}))
    # Even a resealed semantically plausible omission must fail full-set recomputation.
    forged = reseal(
        QueryResult,
        {
            **result.model_dump(mode="json"),
            "candidates": [result.candidates[0].model_dump(mode="json")],
        },
    )
    with pytest.raises(GroundingError, match="stale_or_forged"):
        service.validate_result(q, forged)
    # Existing result under a changed data availability binding is an old cache, not authorization.
    with pytest.raises(GroundingError, match="stale_or_forged"):
        service.validate_result(updated(q, artifacts=[]), result)
    catalogs.disable(mapping.candidates[0].implementation_ref)
    one = service.query(q)
    assert len(one.candidates) == len(one.rejected) == 1
    new_args = create_run(database, data, mapping, writer)
    new_lock = locks.bind(**new_args)  # First bind succeeds despite one revoked OR candidate.
    new_q = query_request(database, mapping, writer, new_lock)
    assert len(service.query(new_q).candidates) == 1
    with pytest.raises(GroundingError, match="stale_or_forged"):
        service.validate_result(q, result)
    catalogs.disable(mapping.candidates[1].implementation_ref)
    assert service.query(q).status == service.query(new_q).status == "gaps"
    all_args = create_run(database, data, mapping, writer)
    all_lock = locks.bind(**all_args)
    assert service.query(query_request(database, mapping, writer, all_lock)).status == "gaps"
    assert (
        locks.read(args["run_id"]) == lock
    )  # Neither safety state nor future catalogs replace it.
    # A failed reader is a technical exception; no cached empty result or partial writes.
    original = catalogs.current_safety

    def fail(_):
        raise OSError("synthetic catalog transport failure")

    before = counts(database)
    catalogs.current_safety = fail
    try:
        with pytest.raises(OSError, match="transport"):
            service.query(q)
    finally:
        catalogs.current_safety = original
    assert counts(database) == before
    # First binding plus Audit is still atomic under the pathwise entry.
    failure_args = create_run(database, data, mapping, writer)
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_query_lock() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION '05B audit failure'; END $$",
    )
    sql(
        database.admin,
        "CREATE TRIGGER fail_query_lock BEFORE INSERT ON audit.event "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_query_lock()",
    )
    before = counts(database)
    try:
        with pytest.raises(psycopg.Error):
            locks.bind(**failure_args)
        assert counts(database) == before
    finally:
        sql(database.admin, "DROP TRIGGER fail_query_lock ON audit.event")
        sql(database.admin, "DROP FUNCTION public.fail_query_lock()")
    # Whole mapping invalidation blocks both established queries and new binding.
    writer.invalidate(mapping.reference(), state=State.DISABLED, request_key=key())
    with pytest.raises(GroundingError, match="mapping_not_admissible"):
        service.query(q)
    with pytest.raises(GroundingError, match="mapping_not_admissible"):
        locks.bind(**failure_args)
    catalogs._disabled.clear()  # Isolated synthetic fixture reset only.
    writer.machine_review(mapping.reference(), support.context(), request_key=key())
    withdrawn_mapping = reseal(Mapping, {**mapping.model_dump(mode="json"), "version": "1.2.1"})
    writer.new_version(mapping.reference(), withdrawn_mapping, support.context(), request_key=key())
    writer.machine_review(withdrawn_mapping.reference(), support.context(), request_key=key())
    withdrawn_args = create_run(database, data, withdrawn_mapping, writer)
    withdrawn_lock = locks.bind(**withdrawn_args)
    withdrawn_q = query_request(database, withdrawn_mapping, writer, withdrawn_lock)
    unbound_args = create_run(database, data, withdrawn_mapping, writer)
    writer.invalidate(withdrawn_mapping.reference(), state=State.WITHDRAWN, request_key=key())
    with pytest.raises(GroundingError, match="mapping_not_admissible"):
        service.query(withdrawn_q)
    with pytest.raises(GroundingError, match="mapping_not_admissible"):
        locks.bind(**unbound_args)
    assert locks.read(args["run_id"]) == lock
    # Current Knowledge withdrawal is never bypassed by pathwise or historical reads.
    database.publisher.withdraw(
        KnowledgeReference.model_validate_json(mapping.source_refs[0].model_dump_json()),
        "synthetic 05B withdrawal",
    )
    with pytest.raises(KnowledgeNotFoundError):
        service.query(q)
    with pytest.raises(KnowledgeNotFoundError):
        locks.bind(**failure_args)
    assert locks.read(args["run_id"]) == lock
    (tmp_path / "query-runtime.synthetic.json").write_text(
        json.dumps(
            dict(
                synthetic=True,
                initial=result.model_dump(mode="json"),
                revoked_or=one.model_dump(mode="json"),
                lock=lock.model_dump(mode="json"),
            ),
            indent=2,
        )
        + "\n"
    )
