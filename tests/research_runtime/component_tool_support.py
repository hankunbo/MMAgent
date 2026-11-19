"""Explicit synthetic JPEG byte consumer using original component/runtime owners.

The fixture installs technical JPEG/PNG declarations; it does not turn unknown
case metadata into facts or relax admission. Consequently the unqualified GAMMA
component remains rejected by current admission's laterality/acquisition gates.
``choice(request)`` deliberately reports that rejection rather than returning a
fabricated plan. No medical model, external service, or GPU is used.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository, ResearchRuntimeExecutions
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.grounding.models import digest, seal, validate
from packages.grounding.synthetic import SyntheticCatalogReader, ref
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeDocuments
from packages.research_runtime.component_inputs import ComponentResearchRequests
from packages.research_runtime.registration import single_choice
from packages.research_runtime.service import ResearchRuntimeService
from packages.research_runtime.task_context import TaskRuns
from packages.tool_admission import InvocationInput, ToolAdmissionService
from packages.tool_admission.models import AdmissionPolicy
from packages.tool_execution import (
    ExecutionPolicy,
    PNGPolicy,
    TechnicalQCRegistry,
    ToolExecutionService,
)
from packages.tool_workflow import CommandAllowlist, ToolWorkflowCompiler
from packages.tool_workflow.models import OutputSpec
from tests.dataset_adapters.test_multimodal import gamma, jpeg
from tests.dataset_adapters.test_registration import service
from tests.grounding.query_support import NOW
from tests.research_runtime.test_component_inputs import policy
from tests.research_runtime.test_task_context import intent
from tests.tool_admission.support import fixture, reseal
from tests.tool_execution.support import runtime_files
from tests.tool_workflow.support import Allowlists, config

# Read and consume every byte of the exact synthetic GAMMA JPEG. This is a byte
# checksum visualization, not image decoding, format relabeling, or medical QC.
JPEG_CONSUMER = """import sys, struct, zlib
with open(sys.argv[1], 'rb') as stream:
    raw = stream.read()
assert raw == {expected!r}, 'unexpected synthetic component bytes'
value = sum(raw) % 2
chunk = lambda k, b: struct.pack('>I', len(b)) + k + b + struct.pack('>I', zlib.crc32(k+b))
png = b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 3, 2, 8, 0, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress((b'\\0' + bytes([value])*3)*2)) + chunk(b'IEND', b'')
with open(sys.argv[2], 'xb') as stream:
    stream.write(png)
"""


def deployment(root, *, consumer_script=None):
    """Build a new synthetic installation without changing historical fixtures."""
    root.mkdir(parents=True, exist_ok=True)
    base = fixture(root, chain=True)
    component_policy = policy()
    old_candidate = next(c for c in base.mapping.candidates if len(c.inputs) == 1)
    constraint = old_candidate.inputs[0].constraint.model_copy(
        update=dict(
            artifact_ref=component_policy.input_binding.artifact_ref,
            modality=component_policy.modality,
            laterality="UNKNOWN",
            coordinate_space="PIXEL_2D",
            unit=component_policy.input_binding.unit_term,
            qc_required=("UNASSESSED",),
            scope_terms=component_policy.scope_terms,
        )
    )
    output = old_candidate.output.model_copy(
        update=dict(
            laterality="UNKNOWN",
            coordinate_space="PIXEL_2D",
            scope_terms=component_policy.scope_terms,
        )
    )
    candidate = validate(
        type(old_candidate),
        old_candidate.model_copy(
            update=dict(
                output=output,
                inputs=(old_candidate.inputs[0].model_copy(update=dict(constraint=constraint)),),
            )
        ),
    )
    old_tool = next(
        t for t in base.catalog.tools if t.implementation_ref == candidate.implementation_ref
    )
    tool = old_tool.model_copy(update=dict(inputs=candidate.inputs, outputs=(candidate.output,)))
    catalog = reseal(
        base.catalog,
        code="CATALOG:SYNTHETIC:COMPONENT_JPEG",
        tools=(tool,),
        artifacts=tuple(dict.fromkeys((*base.catalog.artifacts, constraint.artifact_ref))),
    )
    catalogs = SyntheticCatalogReader((catalog,))
    compatibility = reseal(
        base.policy,
        allowed_terms=tuple(
            dict.fromkeys(
                (
                    *base.policy.allowed_terms,
                    *component_policy.scope_terms,
                    component_policy.modality.code,
                )
            )
        ),
    )
    old = base.owners.read_manifest(
        base.catalog.reference(), candidate.tool_ref, candidate.implementation_ref
    )
    old_profile = base.owners.read_profile(old)
    old_probe = base.owners.read_probe(old, old_profile)
    executable = Path(sys.executable).resolve()
    script = root / "synthetic_component_jpeg.py"
    script.write_text(
        JPEG_CONSUMER.format(expected=jpeg()) if consumer_script is None else consumer_script
    )
    runtime = dict(
        interpreter_uri=executable.as_uri(),
        interpreter_checksum=checksum(executable.read_bytes()),
        dependencies={},
        gpu="not_required",
    )
    resource = dict(
        resource_key="component_byte_consumer",
        resource_role="source_code",
        uri=script.as_uri(),
        sha256=checksum(script.read_bytes())[7:],
        byte_size=script.stat().st_size,
        required=True,
        metadata={"synthetic": True, "purpose": "exact_JPEG_byte_checksum_visualization"},
    )
    argv = [
        str(executable),
        str(script),
        "{input:" + candidate.inputs[0].port.port_name + "}",
        "{output:" + candidate.output_port.port_name + "}",
    ]
    command = dict(shell=False, argv=argv, timeout_seconds=2)
    native = dict(
        old.native_manifest.model_dump(mode="json"),
        entrypoint_uri=script.as_uri(),
        resources=[resource],
        invocation=command,
        runtime_environment={"admission_runtime": runtime},
    )
    manifest = reseal(
        old, catalog_ref=catalog.reference(), native_manifest=native, invocation=command
    )
    profile = reseal(
        old_profile,
        manifest_checksum=manifest.checksum,
        runtime=runtime,
        input_rules=[
            dict(
                port=candidate.inputs[0].port,
                media_types=["image/jpeg"],
                dimensions_min=[2, 3],
                dimensions_max=[2, 3],
                anatomical_targets=component_policy.scope_terms,
            )
        ],
        output_rules=[
            dict(
                port=candidate.output_port,
                media_types=["image/png"],
                dimensions_min=[2, 3],
                dimensions_max=[2, 3],
                anatomical_targets=component_policy.scope_terms,
            )
        ],
        output_target_bindings=[
            reseal(
                old_profile.output_target_bindings[0],
                constraint=candidate.output,
                anatomical_targets=component_policy.scope_terms,
            )
        ],
    )
    probe = reseal(
        old_probe,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        runtime_checksum=digest(profile.runtime),
    )
    owners = base.owners
    owners.manifests[(catalog.reference(), candidate.tool_ref, candidate.implementation_ref)] = (
        manifest
    )
    owners.profiles[manifest.checksum] = profile
    owners.probes[profile.checksum] = probe
    invocation = InvocationInput(
        implementation_ref=candidate.implementation_ref,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        parameters={},
    )
    cfg = reseal(config(), memory_bytes=268435456)
    qc_policy = PNGPolicy(
        artifact_ref=candidate.output.artifact_ref,
        artifact_type_code="ARTIFACT:SYNTHETIC:COMPONENT_BYTE_SUM",
        unit=None,
        unit_term=candidate.output.unit,
        role="derived_product",
        labels=(0, 1),
        max_pixels=6,
    )
    qc_ref = ref("policy", "POLICY:SYNTHETIC:COMPONENT_BYTE_SUM_QC").model_copy(
        update=dict(checksum=digest(qc_policy))
    )
    allowlists = Allowlists()
    allowlists.values[manifest.checksum] = seal(
        CommandAllowlist,
        version="1.0.0",
        source_ref=ref("source_version", "SOURCE:SYNTHETIC:COMPONENT_JPEG_ALLOWLIST"),
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        compiler_config_checksum=cfg.checksum,
        synthetic_only=True,
        executable_uri=executable.as_uri(),
        executable_checksum=runtime["interpreter_checksum"],
        entrypoint_uri=script.as_uri(),
        entrypoint_checksum=checksum(script.read_bytes()),
        resources=[resource],
        argv=argv,
        parameter_schema_version="1.0.0",
        parameter_schema_checksum=digest(manifest.parameter_schema),
        outputs=[
            OutputSpec(port=candidate.output_port, filename="byte-sum.png", qc_policy_ref=qc_ref)
        ],
    )
    return SimpleNamespace(
        root=root,
        base=base,
        component_policy=component_policy,
        candidate=candidate,
        catalogs=catalogs,
        catalog=catalog,
        compatibility=compatibility,
        owners=owners,
        invocation=invocation,
        manifest=manifest,
        profile=profile,
        script=script,
        config=cfg,
        allowlists=allowlists,
        qc=TechnicalQCRegistry({qc_ref: qc_policy}),
        files=runtime_files(root),
        executable=executable,
    )


def assembly(database, tmp_path, key):
    """Use the shared ``test_runtime_repository_02b.database`` tuple unchanged.

    Registration/child creation remain caller actions so tests can assert exact
    before/after counts. ``choice(request)`` and ``executor_for(plan)`` are ready
    for a legitimately admitted child; assembly never bypasses admission.
    """
    inputs = service(database, tmp_path)
    registered = inputs.register_multimodal(
        prepare("gamma", gamma(tmp_path / "source"), "0001"), key=key + "-input"
    )
    visible = registered.visible
    ready = deployment(tmp_path / "deployment")
    repo, sessions = ResearchRunRepository(database[2]), PlannerRepository(database[2])
    ctx = TaskRuns(runtime=repo, inputs=inputs).create(
        case_code=visible.case.case_code,
        key=key + "-parent",
        task=intent(),
        synthetic=True,
        policy_ref=dict(
            kind="policy",
            code="POLICY:SYNTHETIC:COMPONENT_EXECUTION_TEST",
            version="1.0.0",
            checksum=digest("synthetic component execution"),
        ),
        catalog_ref=ready.catalog.reference(),
    )
    evidence = CaseEvidenceRepository(database[2])
    docs = RuntimeDocuments(
        repo=evidence,
        contexts=repo,
        sessions=sessions,
        store=inputs.visible_store,
        run_id=ctx.run_id,
        root=visible.case.artifacts[0],
    )
    args = dict(
        runtime=repo,
        inputs=inputs,
        evidence=evidence,
        documents=docs,
        parent_run_id=ctx.run_id,
        component_id=next(c.id for c in visible.components.components if c.kind == "image"),
        request_key=key,
        component_policy=ready.component_policy,
        candidate=ready.candidate,
        invocation=ready.invocation,
        catalog_ref=ready.catalog.reference(),
        policy=ready.compatibility,
    )
    owner = ComponentResearchRequests(**args)
    admission = ToolAdmissionService(
        policy=AdmissionPolicy(max_total_bytes=268435456),
        grounding=ready.base.grounding,
        catalogs=ready.catalogs,
        manifests=ready.owners,
        profiles=ready.owners,
        probes=ready.owners,
        inputs=owner,
        research=owner,
        objects=inputs.visible_store,
        resources=LocalResourcePolicy(
            allowed_roots=(str(ready.root), str(ready.executable.parent.parent))
        ),
        clock=lambda: NOW,
    )
    compiler = ToolWorkflowCompiler(admission, ready.allowlists, ready.config)
    research = ResearchRuntimeService(
        runtime=repo, research=owner, inputs=owner, compiler=compiler, synthetic=True
    )
    executors = {}

    def choice(request):
        admitted = admission.admit_research(request)
        if not admitted.retained_roots:
            raise ValueError(
                "synthetic_component_not_admitted:"
                + ",".join(sorted({r.reason for r in admitted.rejected}))
            )
        return single_choice(admitted)

    def executor_for(plan):
        if plan.checksum not in executors:
            executions = ResearchRuntimeExecutions(
                database[2], compiler=compiler, plans=(plan.to_bundle_json(),)
            )
            execution_policy = ExecutionPolicy(
                reference=ref("policy", "POLICY:SYNTHETIC:COMPONENT_OFFLINE_EXECUTION"),
                runtime_files=ready.files,
                allowed_plan_checksums=(plan.checksum,),
                purpose=plan.document.purpose,
                synthetic_execution=True,
                cpu_seconds=1,
            )
            executors[plan.checksum] = ToolExecutionService(
                compiler=compiler,
                catalogs=ready.catalogs,
                objects=inputs.visible_store,
                qc=ready.qc,
                repository=executions,
                policy=execution_policy,
                workspace=ready.root / "attempts",
            )
        return executors[plan.checksum]

    ready.compiler = compiler
    return SimpleNamespace(
        inputs=inputs,
        registered=registered,
        visible=visible,
        repo=repo,
        sessions=sessions,
        ctx=ctx,
        docs=docs,
        evidence=evidence,
        args=args,
        owner=owner,
        childresearch=research,
        research=research,
        choice=choice,
        executor_for=executor_for,
        executors=executors,
        ready=ready,
        compiler=compiler,
        admission=admission,
    )
