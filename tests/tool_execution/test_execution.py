"""Real harmless children and real PG faults; never a subprocess mock."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.contracts import QCStatus
from packages.grounding.models import digest
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection
from packages.tool_execution import ExecutionError, TechnicalQCRegistry
from packages.tool_workflow import ExecutionPlan
from tests.database.knowledge_support_04a import sql
from tests.grounding.query_support import NOW, updated
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs
from tests.tool_execution.support import prepared, runtime_files


@pytest.fixture(scope="module")
def mapping_ready(integrated):
    return scoped_mapping(integrated)


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    return runtime_files(tmp_path_factory.mktemp("06d-runtime-closure"))


@pytest.fixture(scope="module")
def host_request(database, mapping_ready):
    data, mapping, writer, _ = mapping_ready
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    return query_request(database, mapping, writer, lock), GroundingQueryService(locks, writer)


@pytest.fixture
def make_ready(database, mapping_ready, tmp_path, closure, host_request):
    def make(mode="success", repo_type=RuntimeToolExecutions, fresh=False):
        root = tmp_path / uuid4().hex
        root.mkdir()
        data, mapping, writer, catalogs = mapping_ready
        q, grounding = host_request
        if fresh:
            locks = RuntimeGroundingLocks(manage.test_target("runtime"))
            lock = locks.bind(**create_run(database, data, mapping, writer))
            q = query_request(database, mapping, writer, lock)
            grounding = GroundingQueryService(locks, writer)
        lock = q.lock
        owners, objects = Owners(), LocalObjectStore(root / "objects")
        q = updated(
            q,
            artifacts=[
                a.model_copy(update={"asset_code": "ASSET:OID_" + uuid4().hex.upper()})
                for a in q.artifacts
            ],
        )
        q = register_inputs(q, owners, objects)
        catalog = catalogs.read_snapshot(lock.catalog_ref)
        invocations = register_deployments(q, catalog, owners, root)
        result = grounding.query(q)
        repo = repo_type(manage.test_target("runtime"))
        for record in owners.records.values():
            repo.register_artifact(record.artifact)
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
        p = prepared(root, mode=mode, repository=repo, base=base, files=closure)
        p.repo = repo
        return p

    return make


@pytest.mark.parametrize(
    "mode,status,error",
    [
        ("success", "SUCCEEDED", None),
        ("isolation", "SUCCEEDED", None),
        ("nonzero", "FAILED", "process_nonzero"),
        ("timeout", "TIMED_OUT", "timeout"),
        ("logs", "FAILED", "log_size_limit"),
        ("missing", "FAILED", "execution_validation_failed"),
        ("oversize", "FAILED", "resource_limit"),
        ("wrong", "FAILED", "output_format_invalid"),
        ("symlink", "FAILED", "execution_validation_failed"),
        ("memory", "FAILED", "process_nonzero"),
        ("cpu", "FAILED", "resource_limit"),
    ],
)
def test_actual_process_failure_matrix(make_ready, monkeypatch, mode, status, error):
    p = make_ready(mode)
    monkeypatch.setenv("DATABASE_URL", "sensitive-test-value-not-to-inherit")
    monkeypatch.setenv("API_KEY", "test-private-value")
    w = p.executor.execute(p.plan, "matrix-" + mode)
    r = w.steps[0]
    assert r.result.status.value == status, r.model_dump_json(indent=2)
    assert r.error_code == error
    if mode == "memory":
        assert r.exit_code == 42  # fixture reached the actual MemoryError handler
    if mode == "cpu":
        assert r.exit_code == -24
    assert r.result.decisive_evidence is False
    assert bool(r.result.artifacts) == (status == "SUCCEEDED")
    recovered = p.executor.recover(p.plan.document.run_id, "matrix-" + mode)
    assert recovered["status"] == "CONFIRMED"
    assert recovered["steps"][0]["process_identity"]["pid"] > 0
    assert p.executor.execute(p.plan, "matrix-" + mode) == w


@pytest.mark.parametrize("mode", ["badlabels", "shape"])
def test_actual_qc_failure_is_not_pass(make_ready, mode):
    p = make_ready(mode)
    w = p.executor.execute(p.plan, "qc-" + mode)
    assert w.steps[0].result.status.value == "SUCCEEDED"
    assert w.steps[0].result.qc_status == QCStatus.FAIL
    assert w.steps[0].qc_records[0].initial_qc_status == "PENDING"


def test_cancel_running_and_process_group_cleanup(make_ready):
    p = make_ready("cancel")
    cancel = threading.Event()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(p.executor.execute, p.plan, "cancel", cancel=cancel)
        deadline = time.monotonic() + 20
        while not list((p.root / "attempts").glob("*/*/process.json")):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        cancel.set()
        w = future.result(timeout=10)
    assert w.status == "CANCELLED" and w.steps[0].result.status.value == "CANCELLED"
    identity = p.executor.recover(p.plan.document.run_id, "cancel")["steps"][0]["process_identity"]
    assert not Path(f"/proc/{identity['pid']}").exists()


def test_cancel_before_launch(make_ready):
    p = make_ready()
    cancel = threading.Event()
    cancel.set()
    w = p.executor.execute(p.plan, "cancel-before", cancel=cancel)
    assert w.steps[0].status_history[-1].value == "CANCELLED"
    assert not list((p.root / "attempts").glob("*/*/process.json"))


def test_fork_child_cannot_outlive_supervised_attempt(make_ready):
    p = make_ready("fork")
    w = p.executor.execute(p.plan, "fork")
    assert w.status == "SUCCEEDED", w.model_dump_json(indent=2)
    assert not list((p.root / "attempts").rglob("*.escaped"))
    proc = p.executor.recover(p.plan.document.run_id, "fork")["steps"][0]["process_identity"]
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        assert not (int(fields[2]) == proc["pid"] and fields[0] not in ("Z", "X"))


@pytest.mark.parametrize(
    "change", ["plan", "input", "resource", "runtime", "missing_qc", "policy", "document_only"]
)
def test_revalidation_and_owner_policy_refuse_before_launch(make_ready, change):
    p = make_ready()
    plan = p.plan
    if change == "plan":
        bundle = json.loads(plan.to_bundle_json())
        bundle["plan"]["steps"][0]["argv"][-1] = "nonzero"
        bundle["plan"]["checksum"] = digest(
            {k: v for k, v in bundle["plan"].items() if k != "checksum"}
        )
        plan = ExecutionPlan.from_bundle_json(json.dumps(bundle))
    elif change == "input":
        record = next(iter(p.f.owners.records.values()))
        p.f.objects._path(record.artifact.uri).write_bytes(b"replaced")
    elif change == "resource":
        p.script.write_text("raise SystemExit(0)")
    elif change == "runtime":
        item = p.policy.runtime_files[0]
        # Change a controlled declaration, never the actual shared interpreter.
        p.executor._policy = p.policy.model_copy(
            update={
                "runtime_files": (
                    item.model_copy(update={"checksum": digest("changed")}),
                    *p.policy.runtime_files[1:],
                )
            }
        )
        p.executor._runner.policy = p.executor._policy
    elif change == "missing_qc":
        p.executor._qc = TechnicalQCRegistry({})
    elif change == "policy":
        p.executor._policy = p.policy.model_copy(
            update={"allowed_plan_checksums": (digest("unregistered"),)}
        )
    else:
        plan = p.plan.document
    if change == "runtime":
        w = p.executor.execute(plan, "reject")
        assert w.steps[0].result.status.value == "BLOCKED"
    else:
        with pytest.raises((ValueError, KeyError)):
            p.executor.execute(plan, "reject")
    assert not list((p.root / "attempts").glob("*/*/process.json"))


def test_same_key_concurrent_never_double_launch(make_ready):
    p = make_ready("cancel")
    cancel = threading.Event()
    with ThreadPoolExecutor() as pool:
        first = pool.submit(p.executor.execute, p.plan, "concurrent", cancel=cancel)
        deadline = time.monotonic() + 20
        while not list((p.root / "attempts").glob("*/*/process.json")):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        with pytest.raises(ExecutionError, match="concurrency_limit"):
            p.executor.execute(p.plan, "concurrent")
        cancel.set()
        w = first.result(timeout=10)
    assert len(list((p.root / "attempts").glob("*/*/process.json"))) == 1
    assert p.executor.execute(p.plan, "concurrent") == w


class FailingStepCommit(RuntimeToolExecutions):
    fail = True

    def advance(self, attempt, step_id, expected, target, **kwargs):
        if target.value == "SUCCEEDED" and self.fail:
            self.fail = False
            # Actual PG rejects Audit AFTER inserting artifacts/derivations in the
            # same transaction. Caller retains the exact durable result for retry.
            from db.runtime.repository import tool_execution

            original = tool_execution.event

            def bad_event(*args, **kw):
                result = original(*args, **kw)
                result["event_checksum"] = "invalid"
                return result

            from unittest.mock import patch

            with patch.object(tool_execution, "event", bad_event):
                return super().advance(attempt, step_id, expected, target, **kwargs)
        return super().advance(attempt, step_id, expected, target, **kwargs)


def test_pg_atomic_rollback_then_resume_commit_without_process_replay(make_ready, database):
    p = make_ready(repo_type=FailingStepCommit)
    run = p.plan.document.run_id
    audit_before = sql(
        database.admin,
        "SELECT count(*) FROM audit.event WHERE run_id=%s AND event_type='TOOL_EXECUTION_RECORDED'",
        (run,),
    )[0][0]
    before = sql(
        database.admin,
        "SELECT (SELECT count(*) FROM runtime.artifact_metadata),"
        "(SELECT count(*) FROM runtime.artifact_derivation)",
    )[0]
    with pytest.raises(psycopg.Error):
        p.executor.execute(p.plan, "commit-failure")
    assert (
        sql(
            database.admin,
            "SELECT (SELECT count(*) FROM runtime.artifact_metadata),"
            "(SELECT count(*) FROM runtime.artifact_derivation)",
        )[0]
        == before
    )
    recovered = p.executor.recover(run, "commit-failure")
    assert recovered["status"] == "ACTIVE" and recovered["steps"][0]["status"] == "RUNNING"
    assert (
        sql(
            database.admin,
            "SELECT count(*) FROM audit.event WHERE run_id=%s AND "
            "event_type='TOOL_EXECUTION_RECORDED'",
            (run,),
        )[0][0]
        == audit_before + 4
    )
    with pytest.raises(ExecutionError, match="recovery_required"):
        p.executor.execute(p.plan, "commit-failure")
    w = p.executor.recover_commit(p.plan, "commit-failure")
    assert w.status == "SUCCEEDED"
    assert len(list((p.root / "attempts").glob("*/*/process.json"))) == 1
    assert p.executor.recover_commit(p.plan, "commit-failure") == w


class LostConfirmation(RuntimeToolExecutions):
    def confirm(self, workflow):
        super().confirm(workflow)
        raise OSError("simulated lost acknowledgement after committed PG transaction")


def test_lost_pg_acknowledgement_is_recovered_without_rerun(make_ready):
    p = make_ready(repo_type=LostConfirmation)
    with pytest.raises(OSError):
        p.executor.execute(p.plan, "lost-ack")
    result = p.executor.recover_commit(p.plan, "lost-ack")
    assert result.status == "SUCCEEDED"
    assert len(list((p.root / "attempts").glob("*/*/process.json"))) == 1


def test_object_promotion_failure_publishes_no_artifact(make_ready, database, monkeypatch):
    p = make_ready()
    before = sql(database.admin, "SELECT count(*) FROM runtime.artifact_metadata")[0][0]

    def reject(*args):
        raise OSError("object media failed")

    monkeypatch.setattr(p.f.objects, "promote", reject)
    w = p.executor.execute(p.plan, "object-failure")
    assert w.status == "FAILED" and not w.steps[0].result.artifacts
    assert sql(database.admin, "SELECT count(*) FROM runtime.artifact_metadata")[0][0] == before
    assert p.executor.cleanup_unreferenced(p.plan, "object-failure")
    assert not list((p.f.objects.root / "staging").iterdir())


def test_runtime_role_cannot_directly_write_execution_or_audit(make_ready):
    p = make_ready()
    for table in ("runtime.tool_attempt", "runtime.tool_step_execution", "audit.event"):
        with p.repo._connect() as connection:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute(f"DELETE FROM {table}")


def test_pg_reservation_serializes_independent_service_instances(make_ready):
    from packages.tool_execution import ToolExecutionService

    p = make_ready("cancel")
    other = ToolExecutionService(
        compiler=p.compiler,
        catalogs=p.f.catalogs,
        objects=p.f.objects,
        qc=p.qc,
        repository=p.repo,
        policy=p.policy,
        workspace=p.root / "second-service-workspace",
    )
    cancel = threading.Event()
    with ThreadPoolExecutor() as pool:
        first = pool.submit(p.executor.execute, p.plan, "pg-concurrent", cancel=cancel)
        deadline = time.monotonic() + 20
        while not list((p.root / "attempts").glob("*/*/process.json")):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        try:
            with pytest.raises(ExecutionError, match="recovery_required"):
                other.execute(p.plan, "pg-concurrent")
        finally:
            cancel.set()
        first.result(timeout=10)
    assert not list((p.root / "second-service-workspace").glob("*/*/process.json"))


def test_pg_same_key_different_policy_conflicts(make_ready):
    from packages.tool_execution import ToolExecutionService

    p = make_ready()
    first = p.executor.execute(p.plan, "same-key-conflict")
    changed = p.policy.model_copy(update={"log_bytes": p.policy.log_bytes + 1})
    other = ToolExecutionService(
        compiler=p.compiler,
        catalogs=p.f.catalogs,
        objects=p.f.objects,
        qc=p.qc,
        repository=p.repo,
        policy=changed,
        workspace=p.root / "attempts",
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        other.execute(p.plan, "same-key-conflict")
    assert p.executor.recover_commit(p.plan, "same-key-conflict") == first


def test_actual_pg_two_step_derivation_uses_new_upstream_product(database, tmp_path, closure):
    from db.runtime.repository import RunRecord
    from packages.contracts import (
        CaseBundle,
        GECEGInstance,
        ImmutablePolicyReference,
        VersionedReference,
    )
    from packages.grounding.models import validate
    from tests.database.test_runtime_repository_02b import _example

    repo = RuntimeToolExecutions(manage.test_target("runtime"))
    p = prepared(tmp_path, chain=True, repository=repo, files=closure)
    doc = p.plan.document
    case = _example(CaseBundle, "CaseBundle.valid.json").model_dump(mode="json")
    case.update(
        case_code=doc.scope.case_code,
        encounter_code=doc.scope.encounter_code,
        patient_hash=doc.scope.patient_hash,
        laterality=doc.scope.laterality,
        artifacts=[r.artifact.model_dump(mode="json") for r in p.f.owners.records.values()],
    )
    case["scope"]["case_code"] = doc.scope.case_code
    repo.create_case_bundle(validate(CaseBundle, case), "TASK:OID_06DCHAINCASE")
    graph = _example(GECEGInstance, "GECEGInstance.valid.json").model_dump(mode="json")
    graph.update(case_code=doc.scope.case_code, laterality=doc.scope.laterality)
    graph["scope"].update(case_code=doc.scope.case_code, run_id=None)
    graph = validate(GECEGInstance, graph)
    repo.create_graph_snapshot(graph, "TASK:OID_06DCHAINGRAPH")
    policy = p.f.q.lock.policy
    repo.create_run(
        RunRecord(
            run_id=doc.run_id,
            case_code=doc.scope.case_code,
            graph_code=graph.graph_code,
            graph_version=1,
            template_ref=graph.template_ref,
            tool_catalog_ref=VersionedReference(**doc.catalog_ref.model_dump(exclude={"kind"})),
            policy_refs=(
                ImmutablePolicyReference(
                    policy_code=policy.code, version=policy.version, checksum=policy.checksum
                ),
            ),
            rule_refs=(graph.edges[0].rule_ref,),
        )
    )
    w = p.executor.execute(p.plan, "pg-dag")
    assert w.status == "SUCCEEDED", w.model_dump_json(indent=2)
    upstream = w.steps[0].result.artifacts[0]
    downstream = w.steps[1].result.artifacts[0]
    assert upstream.asset_code in downstream.derived_from
    assert upstream.checksum != downstream.checksum
    assert (
        sql(
            database.admin,
            "SELECT count(*) FROM runtime.artifact_derivation "
            "WHERE output_asset_code=%s AND input_asset_code=%s",
            (downstream.asset_code, upstream.asset_code),
        )[0][0]
        == 1
    )
    assert sql(database.admin, "SELECT to_regclass('medgraph.workflow_run')")[0][0] is None
    # Result consistency: every confirmation must match the exact committed plan
    # prefix/order. A coherent partial summary cannot hide an already committed step.
    from packages.tool_execution.models import WorkflowResult

    before_audit = sql(database.admin, "SELECT count(*) FROM audit.event")[0][0]
    with pytest.raises(ValueError, match="order"):
        repo.confirm(w.model_copy(update={"steps": tuple(reversed(w.steps))}))
    partial = w.model_dump(mode="json")
    partial.update(
        status="PARTIAL_FAILED", steps=partial["steps"][:1], unstarted_steps=[w.steps[1].step_id]
    )
    coherent = validate(WorkflowResult, partial)
    with pytest.raises(ValueError, match="committed_step_set"):
        repo.confirm(coherent)
    assert sql(database.admin, "SELECT count(*) FROM audit.event")[0][0] == before_audit
    assert repo.recover(doc.run_id, "pg-dag")["result"] == w.model_dump(mode="json")
    # Preserve review examples only in the test's owned temporary directory.
    (tmp_path / "plan-bundle.json").write_text(p.plan.to_bundle_json())
    (tmp_path / "workflow-example.json").write_text(w.model_dump_json(indent=2))


def test_terminal_parent_run_refuses_new_attempt(make_ready, database):
    p = make_ready(fresh=True)
    run = p.plan.document.run_id
    # A terminal host-run is a real PG scope fact. This isolated failure injection
    # changes only the new synthetic run created by this test.
    sql(database.admin, "UPDATE runtime.run SET status='FAILED_TECHNICAL' WHERE run_id=%s", (run,))
    with pytest.raises(psycopg.errors.CheckViolation):
        p.executor.execute(p.plan, "terminal-parent")
    assert p.executor.recover(run, "terminal-parent") is None
    assert not list((p.root / "attempts").glob("*/*/process.json"))
