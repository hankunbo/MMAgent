"""Reusable success preparation: original public single-case, 06 and H owners.

Two registrations share installed tool bytes, not case/patient/input IDs or run facts.
No prior fixture/SQL or historical proof is rewritten.
"""

import json
import struct
import zlib

from db import manage
from db.runtime.repository.research import ResearchRuntimeExecutions
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.case_score_ledger.models import ScoreDefinition
from packages.case_score_ledger.service import CaseScoreService, ToolObservation
from packages.contracts import CaseBundle
from packages.grounding.models import digest, seal
from packages.object_store import checksum
from packages.research_runtime.service import ResearchRuntimeService
from packages.research_runtime.single_case import ResearchSingleCaseService
from packages.tool_admission.models import ResearchRequest
from packages.tool_execution import ToolExecutionService
from tests.disease_packs.support import anonymous_case, code
from tests.tool_execution.support import prepared
from tests.tool_workflow.support import choose


def synthetic_input(identity):
    """Distinct deterministic synthetic bytes, independent of any desired outcome."""
    pixels = bytes.fromhex(digest(identity)[7:19])

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" + pixels[:3] + b"\0" + pixels[3:]))
        + chunk(b"IEND", b"")
    )


def registered_request(ready, identity):
    if not hasattr(ready, "task_input_records"):
        ready.task_input_records = tuple(ready.f.owners.records.values())
    f, q = ready.f, ready.f.q
    scope = q.scope.model_copy(
        update=dict(
            case_code=code("CASE"),
            encounter_code=code("ENCOUNTER"),
            patient_hash=digest("10e-independent-synthetic:" + identity),
        )
    )
    records = []
    for old in ready.task_input_records:
        aid = code("ASSET")
        raw = synthetic_input(identity + ":" + old.artifact.asset_code)
        uri = ready.f.objects.promote(ready.f.objects.stage(raw), checksum(raw))
        artifact = old.artifact.model_copy(
            update=dict(
                asset_code=aid,
                uri=uri,
                checksum=checksum(raw),
                provenance=old.artifact.provenance.model_copy(
                    update={
                        "input_checksums": (checksum(raw),),
                        "source_codes": (
                            "SOURCE:TASK_SYNTHETIC:" + digest(identity)[7:23].upper(),
                        ),
                    }
                ),
                scope=old.artifact.scope.model_copy(update=dict(case_code=scope.case_code)),
            )
        )
        metadata = old.metadata.model_copy(
            update=dict(
                asset_code=aid,
                checksum=checksum(raw),
                case_code=scope.case_code,
                encounter_code=scope.encounter_code,
                patient_hash=scope.patient_hash,
            )
        )
        records.append(
            type(old).model_validate_json(
                old.model_copy(
                    update=dict(artifact=artifact, metadata=metadata, byte_size=len(raw))
                ).model_dump_json()
            )
        )
    f.owners.records.update({r.artifact.asset_code: r for r in records})
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
    return case, request


def installed(tmp_path, *, ready=None, identity="support", mode="success"):
    ready = ready or prepared(tmp_path / "fixture", mode=mode)
    case, request = registered_request(ready, identity)
    runtime = ResearchLifecycleRepository(manage.test_target("runtime"))
    runtime.create_case_bundle(case, code("TASK"))
    f = ready.f
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
        completion_goal="GOAL:SYNTHETIC",
    )
    return driver, request, services


def prepare_support(*, root, ready, registry, repository, identity, key):
    """Reusable synthetic success pool entry, through public run/Writer owners only."""
    from types import SimpleNamespace

    from apps.memory_execution_sources import ExecutionSources
    from packages.memory_writer import MemoryWriter

    driver, request, services = installed(root, ready=ready, identity=identity)
    report = driver.run_registered(request, key=key)
    proof = driver.read_task_completion(request.lock.run_id)
    source = ExecutionSources(
        runtime=driver.runtime,
        run_id=request.lock.run_id,
        applicability=("USE:SYNTHETIC",),
        task_completion=driver,
    )
    registry.owners[request.lock.run_id] = source
    writer = MemoryWriter(repository=repository, sources=source)
    episode = writer.write(run_id=request.lock.run_id, request_key=key + "-episode")
    return SimpleNamespace(
        driver=driver,
        request=request,
        services=services,
        report=report,
        proof=proof,
        source=source,
        writer=writer,
        episode=episode,
        key=key,
    )
