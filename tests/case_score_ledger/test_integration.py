"""Actual PG: synthetic tool output -> owner ledger -> Runtime/Audit -> source-checked read."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from db import manage
from db.runtime.repository.case_scores import CaseScoreRepository
from db.runtime.repository.ports import IdempotencyConflictError
from db.runtime.repository.tool_execution import RuntimeToolExecutions
from packages.case_score_ledger.models import ScoreDefinition, ScoreEnvelope
from packages.case_score_ledger.service import (
    CaseScoreService,
    PipelineGraphReader,
    ToolObservation,
)
from packages.grounding.models import canonical, digest, seal
from tests.deterministic_pipeline.test_pipeline import invoke


def setup(p):
    report = invoke(p, "scores")
    state = p.runtime.read_request(p.runtime.key(p.case.case_code, "scores"))
    events = [p.service._read(r) for r in state.trace]
    plans = [e for e in events if e["kind"] == "plan"]
    d = ScoreDefinition(
        code="synthetic.raw_logit",
        layer="model",
        metric="logit",
        definition=(
            "Synthetic fixture only: negative of observed PNG first pixel; no disease classifier"
        ),
        unit="logit",
        direction="higher",
    )
    observations = tuple(
        ToolObservation(
            key=e["action"],
            plan_json=json.dumps(e["bundle"]),
            compiler=p.tools.ready.compiler,
            outputs=(
                (p.data["requirement"].artifact_type_code, d, "synthetic_png_logit/1.0.0", "value"),
            ),
        )
        for e in plans
    )
    runtime = CaseScoreRepository(manage.test_target("runtime"))
    service = CaseScoreService(
        runtime=runtime,
        executions=RuntimeToolExecutions(manage.test_target("runtime")),
        objects=p.objects,
        observations=observations,
        synthetic=True,
        graph_reader=PipelineGraphReader(p.service, p.runtime.key(p.case.case_code, "scores")),
    )
    return report, state, service


def counts(runtime):
    with runtime._connect() as c:
        return c.execute(
            "SELECT count(*) AS n FROM runtime.idempotency_request "
            "WHERE operation='case_score_ledger'"
        ).fetchone()["n"]


def test_pg_closed_loop_conflicts_replay_mutations_and_atomicity(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    report, state, service = setup(p)
    original_report_bytes = p.objects.read(state.report.uri)
    envelope = service.save(state.run_id, "ledger")
    assert envelope == service.read(state.run_id, "ledger") == service.save(state.run_id, "ledger")
    assert envelope.synthetic and envelope.graph_status == "accepted"
    raw = next(e for e in envelope.entries if e.definition.metric == "logit")
    assert raw.value == -1.0 and raw.calibration == "uncalibrated"
    supported = next(e for e in envelope.entries if e.definition.metric == "support")
    assert supported.value.contributing_nodes and supported.source.accepted_event_ids
    # Owner 03 counts and order are exactly preserved despite negative raw model output.
    for e in envelope.entries:
        if e.definition.metric in report["components"]:
            assert e.value.model_dump(mode="json") == report["components"][e.definition.metric]
    assert p.objects.read(state.report.uri) == original_report_bytes
    assert json.loads(original_report_bytes)["schema_version"] == "deterministic-case-report/2.0.0"
    events = p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")
    assert sum(e.payload.get("envelope_checksum") == envelope.checksum for e in events) == 1
    # Never persist entire score object, pixels, source text or caller scores in JSONB.
    with service.runtime._connect() as c:
        row = c.execute(
            "SELECT * FROM runtime.idempotency_request WHERE idempotency_key=%s",
            (service.runtime._score_key(state.run_id, "ledger"),),
        ).fetchone()
    assert set(row["response_payload"]) == {"uri", "checksum", "byte_size"}
    saved_ref = row["response_payload"]
    stored_path = p.objects.root / saved_ref["uri"].removeprefix("object://knowledge/")
    original = stored_path.read_bytes()
    forged = json.loads(original)
    forged["entries"][0]["value"] = -2.0
    forged = seal(ScoreEnvelope, **{k: v for k, v in forged.items() if k != "checksum"})
    stored_path.write_bytes(canonical(forged))
    with pytest.raises(ValueError, match="score_object_checksum"):
        service.read(state.run_id, "ledger")
    stored_path.write_bytes(original)
    # Rehashed valid envelopes do not constitute source authority, even at repository callback.
    with pytest.raises(ValueError, match="not_current"):
        service.runtime.read_score(
            state.run_id, "ledger", service.policy_checksum, lambda: forged, p.objects
        )
    before = counts(service.runtime)
    objects_before = set((p.objects.root / "objects").iterdir())
    audit_before = len(p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE"))

    def fail(*args):
        raise RuntimeError("synthetic score Audit failure")

    with monkeypatch.context() as m:
        m.setattr(service.runtime, "_score_audit", fail)
        with pytest.raises(RuntimeError, match="Audit failure"):
            service.save(state.run_id, "atomic-audit")
    assert counts(service.runtime) == before
    assert set((p.objects.root / "objects").iterdir()) == objects_before
    assert len(p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")) == audit_before
    with monkeypatch.context() as m:
        m.setattr(p.objects, "promote", fail)
        with pytest.raises(RuntimeError):
            service.save(state.run_id, "atomic-object")
    assert counts(service.runtime) == before and not list((p.objects.root / "staging").iterdir())
    assert set((p.objects.root / "objects").iterdir()) == objects_before
    # An existing key cannot be reused for a different installed scoring policy.
    with pytest.raises(IdempotencyConflictError, match="content_conflict"):
        service.runtime.save_score(
            state.run_id, "ledger", digest("other policy"), lambda: envelope, p.objects
        )
    repeated = CaseScoreService(
        runtime=service.runtime,
        executions=service.executions,
        objects=p.objects,
        observations=(
            replace(service.observations[0], outputs=service.observations[0].outputs * 2),
        ),
        synthetic=True,
    ).build(state.run_id)
    assert sum(e.definition.metric == "logit" for e in repeated.entries) == 1
    coverage = next(e for e in repeated.entries if e.definition.metric == "coverage")
    assert (coverage.numerator, coverage.denominator) == (1, 1)
    # Optional unavailable enhancement preserves the independently confirmed result.
    duplicate = replace(service.observations[0], key="unexecuted-enhancement")
    optional = replace(service.observations[0], optional_dependencies=(duplicate.key,))
    parallel = CaseScoreService(
        runtime=service.runtime,
        executions=service.executions,
        objects=p.objects,
        observations=(optional, duplicate),
        synthetic=True,
    )
    value = parallel.build(state.run_id)
    assert (
        next(
            e for e in value.entries if e.definition.metric == "logit" and e.status == "available"
        ).value
        == -1.0
    )
    required = replace(optional, required_dependencies=(duplicate.key,), optional_dependencies=())
    dependent = CaseScoreService(
        runtime=service.runtime,
        executions=service.executions,
        objects=p.objects,
        observations=(required, duplicate),
        synthetic=True,
    )
    assert all(
        e.value is None
        for e in dependent.build(state.run_id).entries
        if e.definition.metric == "logit"
    )
    assert any(
        e.missing_reason == "required_dependency_failed"
        for e in dependent.build(state.run_id).entries
    )
    # Unknown durable job is displayed without execute/recover_commit (neither port is present).
    recover = service.executions.recover
    with monkeypatch.context() as m:
        m.setattr(service.executions, "recover", lambda *args: dict(status="ACTIVE"))
        unknown = CaseScoreService(
            runtime=service.runtime,
            executions=service.executions,
            objects=p.objects,
            observations=service.observations,
            synthetic=True,
        ).build(state.run_id)
        assert unknown.technical[0].status == "UNKNOWN_JOB"
    assert service.executions.recover == recover
    # Post-construction registry drift, scope forgery and wrong artifact readback fail closed.
    with monkeypatch.context() as m:
        m.setattr(service, "synthetic", False)
        with pytest.raises(ValueError, match="registry_drift"):
            service.build(state.run_id)
    existing = service.runtime.artifacts
    with monkeypatch.context() as m:
        m.setattr(
            service.runtime,
            "artifacts",
            lambda *args: tuple(
                a.model_copy(update={"checksum": digest("tampered")}) for a in existing(*args)
            ),
        )
        with pytest.raises(ValueError, match="artifact_scope_or_fact"):
            service.build(state.run_id)
    research_result = research_readback(p, service, state)
    # Pure engineering bindings cannot become a medical Knowledge artifact even via editor.
    from tests.case_score_ledger.test_engineering_stop import fixture as engineering_fixture

    binding, _, _, _ = engineering_fixture(p.objects.root.parent)
    with pytest.raises(TypeError, match="exact registered Knowledge model"):
        p.database.editor.put(binding)
    reset_result = reset_current_validation(p, service, state)
    # Frozen old report still has precisely its original bytes.
    assert p.objects.read(state.report.uri) == original_report_bytes
    import os

    evidence_path = os.environ.get("MMAGENT_H_EVIDENCE_DIR")
    if evidence_path:
        target = Path(evidence_path) / "synthetic-integration.json"
        target.write_text(
            json.dumps(
                dict(
                    synthetic=True,
                    actual_model_executed=False,
                    research_envelope=research_result.model_dump(mode="json"),
                    reset_envelope=reset_result.model_dump(mode="json"),
                    envelope=envelope.model_dump(mode="json"),
                    original_report_checksum=state.report.checksum,
                    score_reference=saved_ref,
                    audit_event_ids=[
                        e.event_id
                        for e in events
                        if e.payload.get("envelope_checksum") == envelope.checksum
                    ],
                ),
                indent=2,
            )
        )


def research_readback(p, service, state):
    """Original 06B research -> 06C -> CPU 06D -> patient Runtime, without a graph reader."""
    from packages.grounding.models import validate
    from packages.grounding.query_models import QueryRequest
    from packages.tool_admission.models import ResearchRequest
    from packages.tool_execution import ToolExecutionService
    from packages.tool_workflow import ResearchContext
    from tests.tool_workflow.support import choose

    ready = p.tools.ready
    context = json.loads(ready.plan._context_json)
    q = validate(QueryRequest, context["request"])
    candidate = context["query_result"]["candidates"][0]["candidate"]
    from db.runtime.repository import RunRecord
    from packages.contracts import ImmutablePolicyReference, VersionedReference
    from packages.graph_engine import InstantiationContext, canonical_checksum, instantiate_template
    from tests.disease_packs.support import code

    research_run = code("RUN")
    graph = instantiate_template(
        p.data["template"].graph_template,
        p.case,
        InstantiationContext(
            graph_code=code("GRAPH"),
            graph_version=1,
            template_checksum=canonical_checksum(p.data["template"].graph_template),
            created_at=p.data["created_at"],
            generator_version="H.synthetic-research",
            source_codes=("SOURCE:SYNTHETIC_FIXTURE",),
            resolutions=(),
            rule_versions=(p.data["rule"].reference().to_versioned_reference(),),
            run_id=research_run,
        ),
    ).instance
    research_run_record = RunRecord(
        run_id=research_run,
        case_code=p.case.case_code,
        graph_code=graph.graph_code,
        graph_version=1,
        template_ref=graph.template_ref,
        tool_catalog_ref=VersionedReference(**q.lock.catalog_ref.model_dump(exclude={"kind"})),
        policy_refs=(
            ImmutablePolicyReference(
                policy_code=q.lock.policy.code,
                version=q.lock.policy.version,
                checksum=q.lock.policy.checksum,
            ),
        ),
        rule_refs=(p.data["rule"].reference().to_versioned_reference(),),
    )
    # Reuse the original Runtime transaction: deferred graph/run FKs commit together.
    with service.runtime._connect() as connection:
        service.runtime._create_graph_snapshot(connection, graph, code("TASK"))
        service.runtime._create_run(connection, research_run_record)
    # No Knowledge Release/Mapping lock or clinical graph qualification.
    req = seal(
        ResearchRequest,
        code="RESEARCH:H:SYNTHETIC",
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=q.lock.policy,
        lock=dict(run_id=research_run, catalog_ref=q.lock.catalog_ref),
        scope=q.scope,
        window_start=q.window_start,
        window_end=q.window_end,
        snapshot_at=q.snapshot_at,
        satisfied_terms=q.satisfied_terms,
        artifacts=q.artifacts,
        candidate=candidate,
        inputs=[
            dict(
                port=i["port"],
                asset_codes=[
                    a.asset_code
                    for a in q.artifacts
                    if a.artifact_ref.model_dump(mode="json") == i["constraint"]["artifact_ref"]
                ],
            )
            for i in candidate["inputs"]
        ],
        invocation=context["selection"]["invocations"][0],
    )
    ready.f.owners.requests[(req.code, req.version, req.checksum)] = req
    admitted = ready.compiler.admission.admit_research(req)
    assert admitted.status == "preparation_only"
    plan = ready.compiler.compile(
        ResearchContext(request=req, admission=admitted), choose(admitted)
    )
    policy = ready.policy.model_copy(
        update=dict(purpose="shadow", allowed_plan_checksums=(plan.checksum,))
    )
    executor = ToolExecutionService(
        compiler=ready.compiler,
        catalogs=ready.f.catalogs,
        objects=p.objects,
        qc=ready.qc,
        repository=service.executions,
        policy=policy,
        workspace=ready.root / "h-research-attempts",
    )
    workflow = executor.execute(plan, "h-shadow")
    assert workflow.status == "SUCCEEDED" and not workflow.decisive_evidence
    observation = replace(service.observations[0], key="h-shadow", plan_json=plan.to_bundle_json())
    research_service = CaseScoreService(
        runtime=service.runtime,
        executions=service.executions,
        objects=p.objects,
        observations=(observation,),
        synthetic=True,
    )
    result = research_service.save(research_run, "research-ledger")
    assert result.graph_status == "unavailable"
    assert next(e for e in result.entries if e.definition.metric == "logit").value == -1.0
    assert research_service.read(research_run, "research-ledger") == result
    # Three real CPU fixture outputs: two share an input root; one is independently
    # registered synthetic input. Same score code must retain all three sources.
    from tests.tool_admission.support import register_inputs, reseal

    executor.execute(plan, "h-shadow-same-root")
    from tests.grounding.query_support import updated

    independent_q = updated(
        q, artifacts=[a.model_copy(update={"asset_code": code("ASSET")}) for a in q.artifacts]
    )
    independent_q = register_inputs(independent_q, ready.f.owners, p.objects)
    for a in independent_q.artifacts:
        service.runtime.register_artifact(ready.f.owners.records[a.asset_code].artifact)
    replacements = dict(
        zip(
            (a.asset_code for a in q.artifacts),
            (a.asset_code for a in independent_q.artifacts),
            strict=True,
        )
    )
    independent_request = reseal(
        req,
        code="RESEARCH:H:INDEPENDENT",
        artifacts=independent_q.artifacts,
        inputs=[
            {**i.model_dump(mode="json"), "asset_codes": [replacements[a] for a in i.asset_codes]}
            for i in req.inputs
        ],
    )
    ready.f.owners.requests[
        (independent_request.code, independent_request.version, independent_request.checksum)
    ] = independent_request
    independent_admission = ready.compiler.admission.admit_research(independent_request)
    independent_plan = ready.compiler.compile(
        ResearchContext(request=independent_request, admission=independent_admission),
        choose(independent_admission),
    )
    independent_executor = ToolExecutionService(
        compiler=ready.compiler,
        catalogs=ready.f.catalogs,
        objects=p.objects,
        qc=ready.qc,
        repository=service.executions,
        policy=policy.model_copy(update={"allowed_plan_checksums": (independent_plan.checksum,)}),
        workspace=ready.root / "h-independent-attempts",
    )
    independent_executor.execute(independent_plan, "h-shadow-independent")
    observations = (
        observation,
        replace(observation, key="h-shadow-same-root"),
        replace(
            observation, key="h-shadow-independent", plan_json=independent_plan.to_bundle_json()
        ),
    )
    services = [
        CaseScoreService(
            runtime=service.runtime,
            executions=service.executions,
            objects=p.objects,
            observations=items,
            synthetic=True,
        )
        for items in (observations, tuple(reversed(observations)))
    ]
    multiple = services[0].save(research_run, "multiple-sources")
    assert services[1].read(research_run, "multiple-sources") == multiple
    assert services[1].save(research_run, "multiple-sources") == multiple
    scores = {
        e.source.plan_checksum + e.source.attempt_id: e
        for e in multiple.entries
        if e.definition.metric == "logit"
    }
    assert len(scores) == 3
    assert len({e.source.artifact_code for e in scores.values()}) == 3
    assert len({e.source.correlation_group for e in scores.values()}) == 2
    shared = [e for e in scores.values() if e.source.plan_checksum == plan.checksum]
    assert (
        len(shared) == 2
        and shared[0].source.correlation_group == shared[1].source.correlation_group
    )
    return result


def reset_current_validation(p, service, state):
    """Persist actual original-03 reset events; returning to the same status stays invalidated."""
    from types import SimpleNamespace

    from db.audit.repository import AuditEvent, GraphVersionAppend
    from packages.deterministic_pipeline.reporting import accepted_validations
    from tests.deterministic_pipeline.test_review_07c import transition

    graph = p.runtime.current_graph(state.run_id)
    audits = list(p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE"))
    events = [p.service._read(r) for r in state.trace]
    values = accepted_validations(graph, events, audits, p.runtime.get_graph_snapshot)
    target = values[0]["item"].target_node_code
    f = SimpleNamespace(validations=values, audits=audits)
    for before, after, reset in (
        ("SUPPORTED", "PENDING", "EXCLUDED"),
        ("PENDING", "EXCLUDED", None),
        ("EXCLUDED", "PENDING", "SUPPORTED"),
        ("PENDING", "SUPPORTED", None),
    ):
        changed, event = transition(f, graph, target, before, after, reset_to=reset)
        document = AuditEvent.model_validate(
            {k: getattr(event, k) for k in AuditEvent.model_fields}
        )
        p.audit.append_graph_version(
            GraphVersionAppend(
                run_id=state.run_id,
                base_graph_version=graph.graph_version,
                graph=changed,
                audit_event=document,
            )
        )
        graph = changed
    value = service.build(state.run_id)
    assert value.graph_status == "unavailable" and not value.accepted_validations
    assert next(e for e in value.entries if e.definition.metric == "logit").value == -1.0
    with pytest.raises(ValueError, match="not_current"):
        service.read(state.run_id, "ledger")
    return value


def test_engineering_budget_receipt_actual_pg(assembled_pipeline, monkeypatch):
    from tests.case_score_ledger.test_engineering_stop import fixture
    from tests.deterministic_pipeline.test_budget_attribution import (
        test_no_new_execution_after_budget_expires_during_compile,
    )

    p = assembled_pipeline
    # Exercise original deadline safety and prove it stops without an engineering approval.
    test_no_new_execution_after_budget_expires_during_compile(p, monkeypatch)
    key = p.runtime.key(p.case.case_code, "expired-before-launch")
    state = p.runtime.read_request(key)
    binding, reader, review_file, _ = fixture(p.objects.root.parent, runtime_config=p.config)
    service = CaseScoreService(
        runtime=CaseScoreRepository(manage.test_target("runtime")),
        executions=RuntimeToolExecutions(manage.test_target("runtime")),
        objects=p.objects,
        observations=(),
        synthetic=True,
        graph_reader=PipelineGraphReader(p.service, key),
        engineering_stop=(reader, binding.checksum),
    )
    envelope = service.save(state.run_id, "engineering-budget")
    assert service.read(state.run_id, "engineering-budget") == envelope
    assert envelope.engineering_stop.trigger == "time_budget"
    assert not envelope.engineering_stop.grants_medical_authority
    assert envelope.engineering_stop.synthetic and envelope.graph_status == "unavailable"
    import os

    if directory := os.environ.get("MMAGENT_H_EVIDENCE_DIR"):
        (Path(directory) / "synthetic-engineering-integration.json").write_text(
            envelope.model_dump_json(indent=2)
        )
    review_file.write_text("{}")
    with pytest.raises(ValueError):
        service.read(state.run_id, "engineering-budget")
    # A withdrawn receipt cannot undo the already durable original immediate safety stop.
    assert p.service._read(state.report)["stop_reason"] == "time_budget"
