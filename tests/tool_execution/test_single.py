"""The first 06D vertical slice uses real compiler/admission, child process and PG."""

from types import SimpleNamespace

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection
from tests.grounding.query_support import NOW
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs
from tests.tool_execution.support import prepared


def test_single_actual_process_qc_and_atomic_registration(database, integrated, tmp_path):
    data, mapping, writer, catalogs = scoped_mapping(integrated)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    grounding = GroundingQueryService(locks, writer)
    owners, objects = Owners(), LocalObjectStore(tmp_path / "objects")
    q = register_inputs(query_request(database, mapping, writer, lock), owners, objects)
    catalog = catalogs.read_snapshot(lock.catalog_ref)
    invocations = register_deployments(q, catalog, owners, tmp_path)
    result = grounding.query(q)
    repository = RuntimeToolExecutions(manage.test_target("runtime"))
    for record in owners.records.values():
        repository.register_artifact(record.artifact)
    base = SimpleNamespace(
        q=q,
        catalogs=catalogs,
        grounding=grounding,
        owners=owners,
        objects=objects,
        result=result,
        service=SimpleNamespace(clock=lambda: NOW),
        selection=AdmissionSelection(
            candidate_refs=tuple(c.identity for c in result.candidates), invocations=invocations
        ),
    )
    ready = prepared(tmp_path, repository=repository, base=base)
    workflow = ready.executor.execute_bundle(ready.plan.to_bundle_json(), "single")
    assert workflow.status == "SUCCEEDED", workflow.model_dump_json(indent=2)
    step = workflow.steps[0]
    assert step.result.qc_status.value == "PASS"
    assert [s.value for s in step.status_history] == [
        "REQUESTED",
        "DRY_RUN_VALIDATED",
        "QUEUED",
        "RUNNING",
        "SUCCEEDED",
    ]
    assert not ready.plan.document.execution_authorized
    recovered = ready.executor.recover(q.lock.run_id, "single")
    assert recovered["status"] == "CONFIRMED"
    assert ready.executor.execute(ready.plan, "single") == workflow
    with repository._connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) AS n FROM runtime.artifact_derivation WHERE output_asset_code=%s",
                (step.result.artifacts[0].asset_code,),
            ).fetchone()["n"]
            > 0
        )
    from tests.database.knowledge_support_04a import sql

    assert (
        sql(
            database.admin,
            "SELECT count(*) FROM audit.event WHERE run_id=%s "
            "AND event_type='TOOL_EXECUTION_RECORDED'",
            (q.lock.run_id,),
        )[0][0]
        == 6
    )
