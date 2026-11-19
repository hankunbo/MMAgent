"""J software fixture: real isolated PG and harmless CPU PNG, no medical graph/model."""

import json
from dataclasses import replace

import psycopg
import pytest

from db import manage
from db.audit.repository import PostgresAuditReader
from db.runtime.repository.ports import IdempotencyConflictError, RunRecord
from db.runtime.repository.research import ResearchRunRepository, ResearchRuntimeExecutions
from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.case_score_ledger.models import ScoreDefinition, ScoreEnvelope
from packages.case_score_ledger.service import CaseScoreService, ToolObservation
from packages.contracts import CaseBundle
from packages.grounding.models import digest, seal
from packages.research_runtime import ResearchRunContext
from packages.research_runtime.service import ResearchRuntimeService
from packages.tool_admission.models import ResearchRequest
from packages.tool_execution import ToolExecutionService
from packages.tool_workflow import ResearchContext
from tests.database.knowledge_support_04a import sql
from tests.disease_packs.support import anonymous_case, code
from tests.grounding.test_database import database as database
from tests.tool_admission.support import reseal
from tests.tool_execution.support import prepared
from tests.tool_workflow.support import choose


def assembly(tmp_path):
    runtime = ResearchRunRepository(manage.test_target("runtime"))
    ready = prepared(tmp_path / "fixture")
    f, q = ready.f, ready.f.q
    case = anonymous_case("GOAL:SYNTHETIC:RESEARCH")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=q.scope.case_code,
                scope=case.scope.model_copy(update={"case_code": q.scope.case_code}),
                encounter_code=q.scope.encounter_code,
                patient_hash=q.scope.patient_hash,
                laterality=q.scope.laterality,
                artifacts=tuple(r.artifact for r in f.owners.records.values()),
            )
        ).model_dump_json()
    )
    runtime.create_case_bundle(case, code("TASK"))
    c = f.mapping.candidates[0]
    invocation = json.loads(ready.plan._context_json)["selection"]["invocations"][0]
    req = seal(
        ResearchRequest,
        code="RESEARCH:J:SOFTWARE_FIXTURE",
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=q.lock.policy,
        lock=dict(run_id=code("RUN"), catalog_ref=q.lock.catalog_ref),
        scope=q.scope,
        window_start=q.window_start,
        window_end=q.window_end,
        snapshot_at=q.snapshot_at,
        satisfied_terms=q.satisfied_terms,
        artifacts=q.artifacts,
        candidate=c,
        inputs=[
            dict(
                port=i.port,
                asset_codes=[
                    a.asset_code for a in q.artifacts if a.artifact_ref == i.constraint.artifact_ref
                ],
            )
            for i in c.inputs
        ],
        invocation=invocation,
    )
    f.owners.requests[(req.code, req.version, req.checksum)] = req
    service = ResearchRuntimeService(
        runtime=runtime, research=f.owners, inputs=f.owners, compiler=ready.compiler, synthetic=True
    )
    return runtime, ready, req, service


def test_pg_research_context_end_to_end(database, tmp_path, monkeypatch):
    # Empty cluster already migrated by the unified owned fixture; replay preserves ledger.
    before = {d: manage.recorded_migrations(database.migration, d) for d in manage.DOMAINS}
    manage.migrate(database.migration)
    assert before == {d: manage.recorded_migrations(database.migration, d) for d in manage.DOMAINS}
    assert "0012" in before["runtime"]
    manage.bootstrap_audit_principals(
        database.admin,
        manage.test_target("audit_writer").user,
        manage.test_target("audit_reader").user,
    )
    rt, ready, req, service = assembly(tmp_path)
    rid = req.lock.run_id
    audits = PostgresAuditReader(manage.test_target("audit_reader"))

    def create(request=req, key="create"):
        return service.create(request.code, request.version, request.checksum, key)

    def counts():
        return sql(
            database.admin,
            "SELECT (SELECT count(*) FROM runtime.run),"
            "(SELECT count(*) FROM runtime.tool_attempt),"
            "(SELECT count(*) FROM runtime.idempotency_request),"
            "(SELECT count(*) FROM audit.event)",
        )[0]

    initial = counts()
    # Wrong controlled source and cross-scope artifact cannot create a run.
    with monkeypatch.context() as m:
        m.setattr(ready.f.owners, "read_request", lambda *a: reseal(req, purpose="debug"))
        with pytest.raises(ValueError, match="registration_identity"):
            create()
    source = next(iter(ready.f.owners.records.values()))
    with monkeypatch.context() as m:
        m.setattr(
            ready.f.owners,
            "read_input",
            lambda *a: source.model_copy(
                update={
                    "artifact": source.artifact.model_copy(update={"checksum": digest("forged")})
                }
            ),
        )
        with pytest.raises(ValueError, match="artifact_metadata_binding_mismatch"):
            create()
    assert counts() == initial
    # Real transaction Audit fault: no run or key remains.
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_j_audit() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'j_audit_failure'; END $$",
    )
    sql(
        database.admin,
        "CREATE TRIGGER fail_j BEFORE INSERT ON audit.event "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_j_audit()",
    )
    try:
        with pytest.raises(psycopg.Error, match="j_audit_failure"):
            create()
        assert counts() == initial
    finally:
        sql(database.admin, "DROP TRIGGER fail_j ON audit.event")
    context = create()
    assert context == create() == service.read(rid)
    assert sql(database.admin, "SELECT count(*) FROM runtime.graph_snapshot")[0][0] == 0
    assert sql(database.admin, "SELECT count(*) FROM runtime.benchmark_host")[0][0] == 0
    with rt._connect() as connection:
        row = connection.execute("SELECT * FROM runtime.run WHERE run_id=%s", (rid,)).fetchone()
    assert row["graph_code"] is row["template_code"] is row["rule_refs"] is None
    assert row["status"] == "CREATED"
    assert (
        sum(
            e.payload.get("context_checksum") == context.checksum
            for e in audits.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
        )
        == 1
    )
    with pytest.raises(ValueError, match="clinical_context_required"):
        rt.current_graph(rid)
    with pytest.raises(ValueError):
        RunRecord.model_validate_json(context.model_dump_json())
    for field, value in (("purpose", "debug"), ("version", "1.0.1")):
        changed = reseal(req, **{field: value})
        ready.f.owners.requests[(changed.code, changed.version, changed.checksum)] = changed
        with pytest.raises(psycopg.Error, match="research_key_content_conflict"):
            create(changed)
    # Role denial and discriminator/old-lock immutability are database constraints.
    with pytest.raises(psycopg.Error, match="permission denied"):
        sql(
            manage.test_target("knowledge_reader"),
            "SELECT runtime.create_research_run('{}','TASK:OID_DENIED1','{}')",
        )
    for statement in (
        "UPDATE runtime.run SET context_version='clinical-run/1.0.0' WHERE run_id=%s",
        "UPDATE runtime.run SET status='COMPLETED_SUPPORTED' WHERE run_id=%s",
        "UPDATE runtime.run SET research_context='{}' WHERE run_id=%s",
    ):
        with pytest.raises(psycopg.Error):
            with rt._connect() as connection:
                connection.execute(statement, (rid,))
    admitted = ready.compiler.admission.admit_research(req)
    plan = service.compile(rid, choose(admitted))
    assert plan.document.purpose == "shadow" and not plan.document.decisive_evidence
    executions = ResearchRuntimeExecutions(
        manage.test_target("runtime"), compiler=ready.compiler, plans=(plan.to_bundle_json(),)
    )
    policy = ready.policy.model_copy(
        update=dict(purpose="shadow", allowed_plan_checksums=(plan.checksum,))
    )
    executor = ToolExecutionService(
        compiler=ready.compiler,
        catalogs=ready.f.catalogs,
        objects=ready.f.objects,
        qc=ready.qc,
        repository=executions,
        policy=policy,
        workspace=ready.root / "research-attempts",
    )
    # Ordinary executor cannot reserve technical run, including valid shadow plan.
    with monkeypatch.context() as m:
        m.setattr(executor, "_repo", RuntimeToolExecutions(manage.test_target("runtime")))
        with pytest.raises(psycopg.Error, match="research_execution_port_required"):
            executor.execute(plan, "wrong-port")
    workflow = executor.execute(plan, "fixed-tool")
    assert workflow.status == "SUCCEEDED" and not workflow.decisive_evidence
    assert executor.execute(plan, "fixed-tool") == workflow
    artifact = workflow.steps[0].result.artifacts[0]
    assert artifact.research_only and artifact.scope.graph_code is None
    with pytest.raises(psycopg.Error, match="research_artifact_purpose"):
        with rt._connect() as connection:
            connection.execute(
                "UPDATE runtime.artifact_metadata SET research_only=false WHERE asset_code=%s",
                (artifact.asset_code,),
            )
    d = ScoreDefinition(
        code="fixture.raw",
        layer="model",
        metric="logit",
        definition="Software fixture only; negative PNG first pixel",
        unit="logit",
        direction="higher",
    )
    observation = ToolObservation(
        key="fixed-tool",
        plan_json=plan.to_bundle_json(),
        compiler=ready.compiler,
        outputs=((artifact.artifact_type_code, d, "synthetic_png_logit/1.0.0", "value"),),
    )

    def scores(observations=(observation,), **kw):
        return CaseScoreService(
            runtime=rt,
            executions=executions,
            objects=ready.f.objects,
            observations=observations,
            synthetic=True,
            context_version="research-run/1.0.0",
            **kw,
        )

    ledger = scores()
    with pytest.raises(ValueError, match="clinical_port_forbidden"):
        scores(graph_reader=object())
    envelope = ledger.save(rid, "j-ledger")
    assert envelope == ledger.read(rid, "j-ledger") == ledger.save(rid, "j-ledger")
    assert envelope.graph_status == "unavailable" and envelope.graph is None
    assert not envelope.accepted_validations and envelope.original_report is None
    assert next(e.value for e in envelope.entries if e.definition.metric == "logit") == -1.0
    assert all(
        e.status == "unavailable"
        for e in envelope.entries
        if e.definition.layer in {"graph", "proposition", "disease"}
    )
    # Source reconstruction, same-key conflict, dependency semantics, no exception fallback.
    forged = seal(
        ScoreEnvelope,
        **{
            **envelope.model_dump(mode="json", exclude={"checksum"}),
            "run_status": "FAILED_TECHNICAL",
        },
    )
    with pytest.raises(ValueError, match="not_current"):
        rt.read_score(rid, "j-ledger", ledger.policy_checksum, lambda: forged, ready.f.objects)
    duplicate = replace(observation, key="optional-unexecuted")
    optional = scores((replace(observation, optional_dependencies=(duplicate.key,)), duplicate))
    assert (
        next(
            e.value
            for e in optional.build(rid).entries
            if e.definition.metric == "logit" and e.status == "available"
        )
        == -1.0
    )
    required = scores((replace(observation, required_dependencies=(duplicate.key,)), duplicate))
    assert all(
        e.value is None for e in required.build(rid).entries if e.definition.metric == "logit"
    )
    with pytest.raises(IdempotencyConflictError, match="content_conflict"):
        optional.save(rid, "j-ledger")
    changed_definition = d.model_copy(update={"definition": "Conflicting meaning"})
    with pytest.raises(ValueError, match="score_definition_conflict"):
        scores(
            (
                replace(
                    observation,
                    outputs=observation.outputs
                    + (
                        (
                            artifact.artifact_type_code,
                            changed_definition,
                            "synthetic_png_logit/1.0.0",
                            "value",
                        ),
                    ),
                ),
            )
        )
    stable = counts()
    objects_before = set((ready.f.objects.root / "objects").iterdir())
    for owner, name in ((rt, "_score_audit"), (ready.f.objects, "promote")):
        with monkeypatch.context() as m:

            def fail(*a):
                raise RuntimeError("j_object_or_audit_failure")

            m.setattr(owner, name, fail)
            with pytest.raises(RuntimeError, match="j_object_or_audit_failure"):
                ledger.save(rid, "failure-" + name)
        assert counts() == stable
        assert set((ready.f.objects.root / "objects").iterdir()) == objects_before
        assert not list((ready.f.objects.root / "staging").iterdir())
    # A real durable unknown attempt stays unknown; execution count never increases.
    attempt = executions.recover(rid, "fixed-tool")
    document = {k: attempt[k] for k in ("run_id", "case_code", "request_hash", "plan_checksum")}
    document.update(
        attempt_id="1" * 32,
        idempotency_key="unknown",
        decision={
            **attempt["decision"],
            "attempt_id": "1" * 32,
            "execution_binding_checksum": digest(
                dict(attempt="1" * 32, request=attempt["request_hash"])
            ),
        },
    )
    executions.reserve(document, plan.document.steps)
    unknown = scores((replace(observation, key="unknown"),))
    assert unknown.build(rid).technical[0].status == "UNKNOWN_JOB"
    n = counts()[1]
    with pytest.raises(ValueError, match="attempt_unconfirmed_recovery_required"):
        executor.execute(plan, "unknown")
    assert counts()[1] == n
    # SQL reservation rejects purpose and scope tampering, before fast-path replay.
    for field, value in (
        ("purpose", "decisive_research"),
        ("scope_checksum", digest("other scope")),
    ):
        bad = {**document, "decision": {**document["decision"], field: value}}
        with pytest.raises(psycopg.Error, match="research_execution_context_mismatch"):
            executions.reserve(bad, plan.document.steps)
    # Recompiled but differently registered research request cannot reuse original run lock.
    changed = reseal(req, purpose="debug")
    ready.f.owners.requests[(changed.code, changed.version, changed.checksum)] = changed
    other_admitted = ready.compiler.admission.admit_research(changed)
    other = ready.compiler.compile(
        ResearchContext(request=changed, admission=other_admitted), choose(other_admitted)
    )
    with pytest.raises(ValueError, match="research_context_plan_mismatch"):
        context.validate_plan(other)
    with pytest.raises(ValueError, match="research_context_plan_mismatch"):
        context.validate_plan(ready.plan)
    changed_step = plan.document.steps[0].model_copy(
        update={
            "implementation_ref": other.document.steps[0].implementation_ref.model_copy(
                update={"version": "99.0.0"}
            )
        }
    )
    with pytest.raises(ValueError, match="research_plan_node_mismatch"):
        executions.reserve(document, (changed_step,))
    assert sql(database.admin, "SELECT count(*) FROM runtime.graph_snapshot")[0][0] == 0
    assert not sql(database.admin, "SELECT * FROM runtime.run_knowledge_lock")
    assert not sql(database.admin, "SELECT * FROM runtime.run_grounding_lock")
    assert ResearchRunContext.model_validate_json(context.model_dump_json()) == context
    # A separate legal legacy control, only after the graph-free positive proof above.
    from tests.database.test_runtime_repository_02b import _create_case_graph, _create_run_steps

    clinical_case, clinical_graph = _create_case_graph(rt)
    clinical_run, _ = _create_run_steps(rt, clinical_case, clinical_graph)
    original = clinical_run.model_dump_json()
    with pytest.raises(ValueError, match="research_context_required"):
        rt.read_research_run(clinical_run.run_id)
    with pytest.raises(ValueError, match="clinical_context_required"):
        rt.create_run(context)
    with pytest.raises(psycopg.Error, match="research_context_immutable"):
        with rt._connect() as c:
            c.execute(
                "UPDATE runtime.run SET context_version='research-run/1.0.0' WHERE run_id=%s",
                (clinical_run.run_id,),
            )
    assert clinical_run.model_dump_json() == original
    assert rt.current_graph(clinical_run.run_id) == clinical_graph
    # Application cannot insert a research discriminator or omit legacy clinical fields.
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with rt._connect() as c:
            c.execute(
                "INSERT INTO runtime.run(run_id,context_version) VALUES(%s,'research-run/1.0.0')",
                (code("RUN"),),
            )
    with pytest.raises(psycopg.errors.RaiseException, match="run version locks are immutable"):
        with rt._connect() as c:
            c.execute(
                "UPDATE runtime.run SET graph_code=NULL WHERE run_id=%s", (clinical_run.run_id,)
            )
    with pytest.raises(psycopg.errors.CheckViolation, match="run_context_shape"):
        with rt._connect() as c:
            c.execute(
                "INSERT INTO runtime.run(run_id,case_code,status,contract_schema_version,"
                "tool_catalog_code,tool_catalog_version,policy_refs,template_code,"
                "template_version,rule_refs) SELECT %s,case_code,status,contract_schema_version,"
                "tool_catalog_code,tool_catalog_version,policy_refs,template_code,"
                "template_version,rule_refs FROM runtime.run WHERE run_id=%s",
                (code("RUN"), clinical_run.run_id),
            )
    import os
    from pathlib import Path

    output = os.environ.get("MMAGENT_J_CONTEXT_EVIDENCE_DIR")
    if output:
        Path(output, "synthetic-chain.json").write_text(
            json.dumps(
                dict(
                    synthetic=True,
                    actual_model_executed=False,
                    research_context=context.model_dump(mode="json"),
                    plan=json.loads(plan.to_bundle_json()),
                    envelope=envelope.model_dump(mode="json"),
                    workflow=workflow.model_dump(mode="json"),
                    legacy_run_json=original,
                    audit=[
                        e.model_dump(mode="json")
                        for e in audits.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
                    ],
                ),
                indent=2,
                default=str,
            )
        )
