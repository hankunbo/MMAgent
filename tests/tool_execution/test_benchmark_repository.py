"""Real PG engineering host/transaction tests, with harmless trusted subprocesses."""

from copy import deepcopy
from uuid import uuid4

import pytest
from psycopg import IntegrityError
from psycopg.types.json import Jsonb

from db import manage
from db.runtime.repository.benchmark import BenchmarkToolExecutions
from packages.contracts import Artifact
from packages.grounding.models import validate
from tests.grounding.test_database import database as database
from tests.tool_execution.test_trusted_local import setup as setup
from tests.tool_execution.trusted_support import prepared


@pytest.fixture
def pg_prepared(setup, database):
    p = prepared(setup)
    doc = p.plan.document
    repo = BenchmarkToolExecutions(manage.test_target("runtime"))
    # Test plans use fixed IDs, isolated database fixture gives each test a clean host.
    with repo._connect() as c:
        present = c.execute(
            "SELECT 1 FROM runtime.benchmark_host WHERE run_id=%s", (doc.run_id,)
        ).fetchone()
    if not present:
        repo.create_host(
            doc.run_id,
            doc.scope.case_code,
            {"fixture": True},
            doc.catalog_ref.model_dump(mode="json"),
            "shadow",
        )
    else:
        # Another test cannot adopt its host/catalog. Each fixture catalog includes unique paths.
        pytest.fail("benchmark test host must be unique")
    for record in p.f.owners.records.values():
        a = validate(
            Artifact,
            {
                **record.artifact.model_dump(mode="json"),
                "scope": {
                    "scope_type": "RUN",
                    "case_code": doc.scope.case_code,
                    "run_id": doc.run_id,
                },
            },
        )
        repo.register_input(a)
    p.admin = database.admin
    p.service._repo = repo
    p.repo = repo
    return p


def test_real_pg_readback_idempotency_and_no_patient_rows(pg_prepared):
    p = pg_prepared
    key = "benchmark-" + uuid4().hex
    outcome = p.service.execute(p.plan, key)
    assert outcome.state == "recorded", outcome.model_dump_json()
    assert outcome.workflow.status == "SUCCEEDED", outcome.model_dump_json()
    recovered = p.repo.recover(p.plan.document.run_id, key)
    assert recovered["result"] == outcome.workflow.model_dump(mode="json")
    assert p.service.execute(p.plan, key) == outcome
    # Failed derivation rolls back both its newly inserted artifact and Audit event.
    original = outcome.workflow.steps[0].result.artifacts[0]
    broken = validate(
        Artifact,
        {
            **original.model_dump(mode="json"),
            "asset_code": "ASSET:OID_" + uuid4().hex.upper(),
            "derived_from": ["ASSET:OID_MISSING_SOURCE"],
        },
    )
    with pytest.raises(IntegrityError):
        p.repo.register_input(broken)
    with p.repo._connect() as c:
        assert (
            c.execute(
                "SELECT 1 FROM runtime.benchmark_artifact WHERE asset_code=%s", (broken.asset_code,)
            ).fetchone()
            is None
        )
    wrong_scope = validate(
        Artifact,
        {
            **original.model_dump(mode="json"),
            "asset_code": "ASSET:OID_" + uuid4().hex.upper(),
            "scope": {
                "scope_type": "RUN",
                "run_id": "RUN:OID_DIFFERENT_HOST",
                "case_code": "CASE:OID_DIFFERENT_HOST",
            },
        },
    )
    with pytest.raises(IntegrityError):
        p.repo.register_input(wrong_scope)
    old_request = {k: v for k, v in recovered.items() if k not in ("steps", "status", "result")}
    # A new catalog cannot be retrofitted onto the old host's durable lock.
    with p.repo._connect() as c:
        c.execute(
            "UPDATE runtime.benchmark_host SET catalog_ref=%s WHERE run_id=%s",
            (Jsonb({"checksum": "old_catalog"}), p.plan.document.run_id),
        )
    with pytest.raises(ValueError, match="benchmark_catalog_lock_mismatch"):
        p.repo.reserve(old_request, p.plan.document.steps)
    with p.repo._connect() as c:
        c.execute(
            "UPDATE runtime.benchmark_host SET catalog_ref=%s WHERE run_id=%s",
            (Jsonb(p.plan.document.catalog_ref.model_dump(mode="json")), p.plan.document.run_id),
        )
    with p.repo._connect() as c:
        assert (
            c.execute("SELECT count(*) AS n FROM runtime.benchmark_derivation").fetchone()["n"] > 0
        )
        assert c.execute("SELECT count(*) AS n FROM runtime.case_record").fetchone()["n"] == 0
        bad = deepcopy(recovered)
        bad["result"]["status"] = "CANCELLED"
        c.execute(
            "UPDATE runtime.benchmark_attempt SET document=%s WHERE attempt_id=%s",
            (Jsonb(bad), bad["attempt_id"]),
        )
    from tests.database.knowledge_support_04a import sql

    assert int(sql(p.admin, "SELECT count(*) FROM audit.event")[0][0]) > 0
    with pytest.raises(ValueError):
        p.repo.recover(p.plan.document.run_id, key)
