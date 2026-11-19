"""R01: new single-case orchestration over real owned PG and harmless CPU execution."""

import json
from threading import Event

import psycopg
import pytest

from db import manage
from db.runtime.repository.research import ResearchRuntimeExecutions
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.case_score_ledger.models import ScoreDefinition
from packages.case_score_ledger.service import CaseScoreService, ToolObservation
from packages.contracts import CaseBundle
from packages.grounding.models import digest, seal
from packages.research_runtime.service import ResearchRuntimeService
from packages.research_runtime.single_case import ResearchSingleCaseService
from packages.tool_admission.models import ResearchRequest
from packages.tool_execution import ToolExecutionService
from tests.database.knowledge_support_04a import sql
from tests.disease_packs.support import anonymous_case, code
from tests.grounding.test_database import database as database
from tests.tool_execution.support import prepared
from tests.tool_workflow.support import choose


def installed(tmp_path, mode="success"):
    ready = prepared(tmp_path / "fixture", mode=mode)
    runtime = ResearchLifecycleRepository(manage.test_target("runtime"))
    f, q = ready.f, ready.f.q
    scope = q.scope.model_copy(
        update=dict(case_code=code("CASE"), encounter_code=code("ENCOUNTER"))
    )
    records = []
    for old in f.owners.records.values():
        aid = code("ASSET")
        artifact = old.artifact.model_copy(
            update=dict(
                asset_code=aid,
                scope=old.artifact.scope.model_copy(update=dict(case_code=scope.case_code)),
            )
        )
        metadata = old.metadata.model_copy(
            update=dict(
                asset_code=aid,
                case_code=scope.case_code,
                encounter_code=scope.encounter_code,
            )
        )
        records.append(
            type(old).model_validate_json(
                old.model_copy(update=dict(artifact=artifact, metadata=metadata)).model_dump_json()
            )
        )
    f.owners.records = {r.artifact.asset_code: r for r in records}
    case = anonymous_case("GOAL:SYNTHETIC:RESEARCH")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=scope.case_code,
                scope=case.scope.model_copy(update=dict(case_code=scope.case_code)),
                encounter_code=scope.encounter_code,
                patient_hash=scope.patient_hash,
                laterality=scope.laterality,
                artifacts=tuple(r.artifact for r in records),
            )
        ).model_dump_json()
    )
    runtime.create_case_bundle(case, code("TASK"))
    candidate = f.mapping.candidates[0]
    request = seal(
        ResearchRequest,
        code="RESEARCH:R01:" + code("TASK").split(":")[1],
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=q.lock.policy,
        lock=dict(run_id=code("RUN"), catalog_ref=q.lock.catalog_ref),
        scope=scope,
        window_start=q.window_start,
        window_end=q.window_end,
        snapshot_at=q.snapshot_at,
        satisfied_terms=q.satisfied_terms,
        artifacts=tuple(r.metadata for r in records),
        candidate=candidate,
        inputs=[
            dict(
                port=i.port,
                asset_codes=[
                    r.metadata.asset_code
                    for r in records
                    if r.metadata.artifact_ref == i.constraint.artifact_ref
                ],
            )
            for i in candidate.inputs
        ],
        invocation=json.loads(ready.plan._context_json)["selection"]["invocations"][0],
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    research = ResearchRuntimeService(
        runtime=runtime, research=f.owners, inputs=f.owners, compiler=ready.compiler, synthetic=True
    )
    choice = choose(ready.compiler.admission.admit_research(request))
    services = {}

    def bind(plan, key):
        if key in services:
            return services[key]
        executions = ResearchRuntimeExecutions(
            manage.test_target("runtime"), compiler=ready.compiler, plans=(plan.to_bundle_json(),)
        )
        executor = ToolExecutionService(
            compiler=ready.compiler,
            catalogs=ready.f.catalogs,
            objects=ready.f.objects,
            qc=ready.qc,
            repository=executions,
            policy=ready.policy.model_copy(
                update=dict(purpose="shadow", allowed_plan_checksums=(plan.checksum,))
            ),
            workspace=ready.root / "single-case-attempts",
        )
        output_type = ready.qc.policy_for(plan.document.steps[0].outputs[0]).artifact_type_code
        scores = CaseScoreService(
            runtime=runtime,
            executions=executions,
            objects=ready.f.objects,
            observations=(
                ToolObservation(
                    key=key,
                    plan_json=plan.to_bundle_json(),
                    compiler=ready.compiler,
                    outputs=(
                        (
                            output_type,
                            ScoreDefinition(
                                code="fixture.raw",
                                layer="model",
                                metric="logit",
                                unit="logit",
                                definition="Synthetic PNG software value; no disease score",
                                direction="higher",
                            ),
                            "synthetic_png_logit/1.0.0",
                            "value",
                        ),
                    ),
                ),
            ),
            synthetic=True,
            context_version="research-run/1.0.0",
        )
        services[key] = executor, scores
        return services[key]

    driver = ResearchSingleCaseService(
        runtime=runtime,
        research=research,
        choice=choice,
        bind_plan=bind,
        config_checksum=digest("R01 synthetic installation"),
        budget_seconds=60,
    )
    return driver, request, services


def test_single_case_lifecycle_pg(database, tmp_path, monkeypatch):
    before = manage.recorded_migrations(database.migration, "runtime")
    assert "0012" in before
    manage.migrate(database.migration)
    assert before == manage.recorded_migrations(database.migration, "runtime")
    driver, req, services = installed(tmp_path / "success")
    report = driver.run_registered(req, key="single")
    assert report["terminal"] and report["run_status"] == "COMPLETED_INSUFFICIENT_EVIDENCE"
    assert report["technical_status"] == "SUCCEEDED"
    assert report["graph"] is None and not report["accepted_validations"]
    state = driver.runtime.lifecycle(req.lock.run_id)
    assert state["phase"] == "FINISHED"
    assert driver.run_registered(req, key="single") == report
    assert driver.runtime.lifecycle(req.lock.run_id)["deadline"] == state["deadline"]
    with pytest.raises(psycopg.errors.CheckViolation, match="research_run_context_conflict"):
        driver.run_registered(req, key="different")
    assert (
        sql(
            database.admin,
            "SELECT count(*) FROM runtime.tool_attempt WHERE run_id='" + req.lock.run_id + "'",
        )[0][0]
        == 1
    )

    # H persistence failure after tool completion leaves an explicit recoverable SCORING phase.
    driver, req, services = installed(tmp_path / "score-failure")
    original = driver.runtime.save_score
    with monkeypatch.context() as m:
        m.setattr(
            driver.runtime,
            "save_score",
            lambda *a: (_ for _ in ()).throw(RuntimeError("score_store_failed")),
        )
        with pytest.raises(RuntimeError, match="score_store_failed"):
            driver.run_registered(req, key="score-failure")
    assert driver.runtime.lifecycle(req.lock.run_id)["phase"] == "SCORING"
    assert driver.runtime.run_status(req.lock.run_id) != "CREATED"
    assert driver.runtime.save_score == original
    executor = services["score-failure:tool"][0]
    with monkeypatch.context() as m:
        m.setattr(
            executor, "execute", lambda *a, **k: pytest.fail("must not execute while scoring")
        )
        assert driver.run_registered(req, key="score-failure")["terminal"]

    # Cancel before launch: no attempt, no fabricated negative score, audited safe stop.
    driver, req, _ = installed(tmp_path / "cancel")
    cancel = Event()
    cancel.set()
    report = driver.run_registered(req, key="cancel", cancel=cancel)
    assert report["run_status"] == "STOPPED_UNSAFE" and report["technical_status"] == "NOT_EXECUTED"

    for mode in ("nonzero", "timeout"):
        driver, req, _ = installed(tmp_path / mode, mode=mode)
        report = driver.run_registered(req, key=mode)
        assert report["run_status"] == "FAILED_TECHNICAL"
        assert report["technical_status"] in {"FAILED", "TIMED_OUT"}
        assert all(e["value"] is None for e in report["model_entries"])

    driver, req, _ = installed(tmp_path / "admission-rejected")
    with monkeypatch.context() as m:
        m.setattr(
            driver.research.compiler,
            "compile",
            lambda *a: (_ for _ in ()).throw(ValueError("not_admitted")),
        )
        report = driver.run_registered(req, key="reject")
        assert report == driver.run_registered(req, key="reject")
    assert report["run_status"] == "FAILED_TECHNICAL" and report["score_checksum"] is None

    # Actual unconfirmed commit journal; next recovery can only commit, never launch.
    driver, req, services = installed(tmp_path / "unknown")
    original_bind = driver.bind_plan

    with monkeypatch.context() as m:
        # Limit the injected durable commit fault to this context.
        def broken(plan, key):
            executor, scores = original_bind(plan, key)
            m.setattr(
                executor._repo,
                "confirm",
                lambda *a: (_ for _ in ()).throw(RuntimeError("ack_lost")),
            )
            return executor, scores

        m.setattr(driver, "bind_plan", broken)
        report = driver.run_registered(req, key="unknown")
    assert report["technical_status"] == "UNKNOWN_JOB" and not report["terminal"]
    assert report["run_status"] == "EXECUTING"
    executor, _ = services["unknown:tool"]
    with monkeypatch.context() as m:
        m.setattr(executor, "execute", lambda *a, **k: pytest.fail("UNKNOWN replay forbidden"))
        recovered = driver.run_registered(req, key="unknown", recover_only=True)
    assert recovered["technical_status"] == "SUCCEEDED" and recovered["terminal"]

    # Run transition + lifecycle state + Audit all roll back together on Audit failure.
    driver, req, _ = installed(tmp_path / "audit-failure")
    context = driver.research.create(req.code, req.version, req.checksum, "audit")
    plan = driver.research.compile(context.run_id, driver.choice)
    state = driver.runtime.begin_execution(context, plan, "audit", driver.config_checksum, 60)
    sql(
        database.admin,
        "CREATE FUNCTION public.fail_r01_audit() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'r01_audit_failed'; END $$",
    )
    sql(
        database.admin,
        "CREATE TRIGGER fail_r01 BEFORE INSERT ON audit.event "
        "FOR EACH ROW EXECUTE FUNCTION public.fail_r01_audit()",
    )
    try:
        with pytest.raises(Exception, match="r01_audit_failed"):
            driver.runtime.record_lifecycle(state, phase="PREPARED", target_status="NORMALIZING")
    finally:
        sql(database.admin, "DROP TRIGGER fail_r01 ON audit.event")
    assert driver.runtime.lifecycle(context.run_id) == state
    assert driver.runtime.run_status(context.run_id) == "CREATED"

    # Original 07A reads only synthetic Training. The new owner reads visible artifacts,
    # never evaluator labels, notes or summaries. An incompatible fixture profile rejects.
    from db.runtime.repository.dataset_input import DatasetInputRepository
    from packages.dataset_adapters.service import DatasetPreparationService, VisibleInputReader
    from packages.object_store import LocalObjectStore
    from packages.research_runtime.registration import VisibleResearchRequests
    from tests.dataset_adapters.support import fair
    from tests.tool_admission.support import reseal

    driver, template, _ = installed(tmp_path / "visible-07A")
    original_record = next(iter(driver.research.inputs.records.values()))
    visible_store = LocalObjectStore(tmp_path / "visible")
    evaluation_store = LocalObjectStore(tmp_path / "evaluation")
    dataset_repo = DatasetInputRepository(driver.runtime)
    adapter, _ = fair(tmp_path / "synthetic-data")
    preparation = DatasetPreparationService(
        dataset_repo, visible_store, evaluation_store, catalog=adapter.catalog
    )
    registered = VisibleResearchRequests(
        runtime=driver.runtime,
        visible_reader=VisibleInputReader(dataset_repo, visible_store),
        objects=visible_store,
        candidate=template.candidate,
        invocation=template.invocation,
        catalog_ref=template.lock.catalog_ref,
        policy=template.compatibility_policy,
        input_binding=reseal(
            original_record.type_binding,
            contract_type_code="ARTIFACT:SLO_IMAGE",
            contract_unit="px",
        ),
        modality=original_record.metadata.modality,
        scope_terms=template.satisfied_terms,
        config_checksum=driver.config_checksum,
    )
    driver.preparation, driver.register_request = preparation, registered.register
    driver.research.research = driver.research.inputs = registered
    driver.research.compiler.admission.research = registered
    driver.research.compiler.admission.inputs = registered
    # This old harmless fixture requires the old input owner/profile; rejection is mandatory.
    sample = adapter.prepare("data_00000.npz", split="training")
    result = driver.run(sample, key="visible")
    assert result["terminal"] and result["technical_status"] == "NOT_EXECUTED"
    assert result["reason"] == "admission_or_compile_rejected"
    request = registered.for_key("visible")
    context = driver.research.read(request.lock.run_id)
    assert context.request == request
    assert not any(
        s in request.model_dump_json() for s in ("SECRET_DIAGNOSIS", "SECRET_SUMMARY", "glaucoma")
    )
    assert driver.run(sample, key="visible") == result
