"""Reviewed synthetic JPEG consumer through original native/review/runtime owners.

Only the local source snapshot is synthetic; the original native projection and
review produce a non-synthetic catalog. Matching compiler/execution flags retain
their original meanings. No real catalog, model, network, GPU, or patient is used.
The review qualifies a single observation's byte sum, preserving unknown eye/time.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from db.runtime.repository.case_evidence import CaseEvidenceRepository
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository, ResearchRuntimeExecutions
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.grounding.models import Candidate, canonical, digest, seal, validate
from packages.grounding.synthetic import ref
from packages.integrations.medgraph import (
    AdapterConfig,
    MedgraphCatalogReader,
    ProjectionPolicy,
    SnapshotStore,
)
from packages.integrations.medgraph.models import NativeSnapshot
from packages.integrations.medgraph.projection import expected_manifest
from packages.integrations.medgraph.research import ResearchReview, ReviewedCatalog
from packages.integrations.medgraph.resources import LocalResourcePolicy, observe_manifest_resources
from packages.research_runtime.agent_records import RuntimeDocuments
from packages.research_runtime.component_inputs import ComponentResearchRequests
from packages.research_runtime.registration import single_choice
from packages.research_runtime.service import ResearchRuntimeService
from packages.research_runtime.task_context import TaskRuns
from packages.tool_admission import InvocationInput, ToolAdmissionService
from packages.tool_admission.models import AdmissionPolicy, JSONOutputRule
from packages.tool_execution import (
    ExecutionPolicy,
    PNGPolicy,
    TechnicalQCRegistry,
    ToolExecutionService,
)
from packages.tool_workflow import CommandAllowlist, ToolWorkflowCompiler
from packages.tool_workflow.models import OutputSpec
from tests.dataset_adapters.test_multimodal import gamma
from tests.dataset_adapters.test_registration import service
from tests.grounding.query_support import NOW
from tests.integrations.medgraph.test_research import pin
from tests.research_runtime.component_tool_support import deployment as offline_deployment
from tests.research_runtime.test_task_context import intent
from tests.tool_admission.support import reseal
from tests.tool_workflow.support import Allowlists


def deployment(root, *, laterality="not_consumed", acquisition_time="not_consumed", bundle=None):
    """Install exact pinned audit files using the original ReviewedCatalog owner."""
    from packages.tool_admission.method_metadata import MethodMetadata

    script = None
    if bundle is not None:
        from tests.dataset_adapters.test_multimodal import jpeg

        script = (
            "import sys\n"
            "from pathlib import Path\n"
            f"assert Path(sys.argv[1]).read_bytes() == {jpeg()!r}\n"
            f"with open(sys.argv[2], 'xb') as stream: stream.write({bundle!r})\n"
        )
    ready = offline_deployment(root, consumer_script=script)
    raw = json.loads(
        (Path(__file__).parents[1] / "integrations/medgraph/native.synthetic.json").read_text()
    )
    for artifact in raw["artifacts"]:
        if artifact["id"] == 1:
            artifact["code"] = "fixture:EncodedJPEG"
            artifact["schema_json"] = {"dimensions": 2, "formats": ["jpeg"], "synthetic": True}
        else:
            artifact["code"] = "fixture:ByteSumPNG"
            artifact["schema_json"] = {
                "dimensions": 2,
                "formats": ["png"],
                "synthetic": True,
                "values": "input_byte_sum_modulo_two",
            }
            if bundle is not None:
                artifact["code"] = "fixture:FundusBundle"
                artifact["schema_json"] = {
                    "dimensions": 1,
                    "formats": ["json"],
                    "synthetic": True,
                    "values": "synthetic_four_mask_bundle_not_model_prediction",
                }
    for resource in raw["resources"]:
        resource.update(
            uri=ready.script.as_uri(),
            sha256=pin(ready.script).checksum[7:],
            byte_size=ready.script.stat().st_size,
        )
    native = validate(NativeSnapshot, raw)
    native = native.model_copy(
        update=dict(
            manifests=tuple(
                expected_manifest(native, i) for i in native.implementations if i.enabled
            )
        )
    )

    class SyntheticNativeSource:
        def __init__(self, value):
            self.value = value

        def capture(self):
            return self.value

    source = SyntheticNativeSource(native)
    store = root / "native-snapshots"
    store.mkdir()
    legacy = MedgraphCatalogReader(
        source,
        SnapshotStore(store),
        AdapterConfig(source_code="COMPONENT_CPU_FIXTURE", dsn_env="UNUSED_SYNTHETIC_FIXTURE"),
        ProjectionPolicy(),
    )
    native_catalog = legacy.capture_snapshot()
    native_tool = next(
        t for t in native_catalog.tools if t.validation_status.value == "smoke_tested"
    )
    native_manifest = legacy.read_manifest(
        native_catalog.reference(), native_tool.tool_ref, native_tool.implementation_ref
    )
    resources = LocalResourcePolicy(allowed_roots=(str(root), str(ready.executable.parent.parent)))
    native_evidence = root / "native-resource-evidence.json"
    native_evidence.write_bytes(canonical(observe_manifest_resources(native_manifest, resources)))
    audit = root / "method-metadata-review.txt"
    audit.write_text(
        "SYNTHETIC CPU SOFTWARE TEST ONLY. The installed program reads every byte "
        "of the pinned 2x3 GAMMA JPEG fixture, asserts exact equality, computes the "
        "sum modulo two, and writes a 2x3 grayscale PNG. It does not consume eye "
        "laterality or acquisition time. Scope: single_input_current_observation. "
        "Unknown laterality and null acquisition time must remain unchanged. "
        "Encoded JPEG bytes have no physical measurement unit: not_applicable. "
        "Excluded: bilateral_pairing, longitudinal_comparison, clinical_conclusion. "
        "No medical inference, decoded-image validity, real model, or GPU claim.\n"
    )
    if bundle is not None:
        audit.write_text(
            "SYNTHETIC CPU SOFTWARE TEST ONLY. Exact synthetic JPEG bytes are checked, "
            "then a fixed four-mask JSON bundle is emitted for original QC and persistence. "
            "This is not model inference, accuracy validation, or a real deployment. "
            "Eye and time are not consumed; unit is not_applicable. "
            "No bilateral pairing, longitudinal inference, or clinical conclusion.\n"
        )
    unit = dict(state="not_applicable", code=None)
    input_constraint = validate(
        type(ready.candidate.inputs[0].constraint),
        ready.candidate.inputs[0].constraint.model_copy(
            update=dict(artifact_ref=native_tool.inputs[0].constraint.artifact_ref, unit=unit)
        ),
    )
    output_constraint = validate(
        type(ready.candidate.output),
        ready.candidate.output.model_copy(
            update=dict(artifact_ref=native_tool.outputs[0].artifact_ref, unit=unit)
        ),
    )
    component_policy = reseal(
        ready.component_policy,
        input_binding=reseal(
            ready.component_policy.input_binding,
            artifact_ref=input_constraint.artifact_ref,
            unit_term=unit,
            source_ref=ref("source_version", "SOURCE:SYNTHETIC:ENCODED_JPEG_NO_PHYSICAL_UNIT"),
        ),
    )
    argv = [
        str(ready.executable),
        str(ready.script),
        "{input:" + native_tool.inputs[0].port.port_name + "}",
        "{output:" + native_tool.output_ports[0].port_name + "}",
    ]
    command = dict(shell=False, argv=argv, timeout_seconds=2)
    execution_declaration = native_manifest.native_manifest.model_copy(
        update=dict(
            entrypoint_uri=ready.script.as_uri(),
            working_directory=None,
            resources=ready.manifest.native_manifest.resources,
            invocation=command,
            runtime_environment={
                "admission_runtime": ready.profile.runtime.model_dump(mode="json")
            },
        )
    )
    execution_declaration = validate(type(execution_declaration), execution_declaration)
    parameter_schema = dict(type="object", properties={}, additionalProperties=False)
    preprocessing = dict(
        version="1.0.0", policy_code="POLICY:SYNTHETIC:EXACT_JPEG_BYTES", operations=[]
    )
    postprocessing = dict(
        version="1.0.0",
        policy_code="POLICY:SYNTHETIC:FOUR_MASK_BUNDLE"
        if bundle is not None
        else "POLICY:SYNTHETIC:BYTE_SUM_PNG",
        operations=[],
    )
    metadata = seal(
        MethodMetadata,
        purposes=("debug", "shadow"),
        laterality=laterality,
        acquisition_time=acquisition_time,
        basis=("Reviewed synthetic byte consumer; eye and time are neither read nor inferred."),
        evidence=pin(audit),
        execution_declaration_checksum=digest(execution_declaration),
        input_constraints_checksum=digest((input_constraint,)),
        output_constraints_checksum=digest((output_constraint,)),
        parameter_schema_checksum=digest(parameter_schema),
        preprocessing_checksum=digest(preprocessing),
        postprocessing_checksum=digest(postprocessing),
    )
    preprocessing["metadata_requirements"] = metadata.model_dump(mode="json")
    review = seal(
        ResearchReview,
        source_catalog_ref=native_catalog.reference(),
        source_implementation_ref=native_tool.implementation_ref,
        source_manifest_checksum=native_manifest.checksum,
        config_checksum=native_manifest.config_checksum,
        resolutions=[
            dict(
                gap=g,
                basis="Synthetic byte consumer review only; no medical validation claim.",
                evidence=pin(audit),
                disposition="formal_gate_deferred_shadow_only"
                if g == "validation_gates_missing"
                else "verified_for_shadow",
            )
            for g in native_manifest.binding.gaps
        ],
        input_constraints=(input_constraint,),
        output_constraints=(output_constraint,),
        deployment_declaration=execution_declaration,
        parameter_schema=parameter_schema,
        preprocessing_policy=preprocessing,
        postprocessing_policy=postprocessing,
        interpreter=pin(ready.executable),
        native_resource_evidence=pin(native_evidence),
    )
    record = root / "installed-method-review.json"
    record.write_bytes(canonical(review))
    catalogs = ReviewedCatalog(legacy, pin(record), resources)
    manifest, tool = catalogs.manifest, catalogs.tool
    candidate = validate(
        Candidate,
        dict(
            code="CAPABILITY:SYNTHETIC:REVIEWED_FUNDUS_BUNDLE"
            if bundle is not None
            else "CAPABILITY:SYNTHETIC:REVIEWED_JPEG_BYTE_SUM",
            capability_ref=tool.capabilities[0],
            tool_ref=tool.tool_ref,
            implementation_ref=tool.implementation_ref,
            workflow_ref=None,
            output_port=tool.output_ports[0],
            output=tool.outputs[0],
            inputs=tool.inputs,
            validation_required="smoke_tested",
            intended_use="research_only",
            execution_policy="research",
            estimate_ref=None,
        ),
    )
    profile = reseal(
        ready.profile,
        manifest_checksum=manifest.checksum,
        config_checksum=manifest.config_checksum,
        policy_checksum=manifest.policy_checksum,
        input_rules=[
            ready.profile.input_rules[0].model_copy(update=dict(port=candidate.inputs[0].port))
        ],
        output_rules=[
            JSONOutputRule(
                port=candidate.output_port,
                dimensions_min=(1,),
                dimensions_max=(1,),
                anatomical_targets=ready.profile.output_rules[0].anatomical_targets,
            )
            if bundle is not None
            else ready.profile.output_rules[0].model_copy(update=dict(port=candidate.output_port))
        ],
        output_target_bindings=[
            reseal(
                ready.profile.output_target_bindings[0],
                port=candidate.output_port,
                constraint=candidate.output,
            )
        ],
        preprocessing_checksum=digest(preprocessing),
        postprocessing_checksum=digest(postprocessing),
    )
    old_probe = ready.owners.probes[ready.profile.checksum]
    probe = reseal(
        old_probe, manifest_checksum=manifest.checksum, profile_checksum=profile.checksum
    )
    ready.owners.profiles[manifest.checksum] = profile
    ready.owners.probes[profile.checksum] = probe
    invocation = InvocationInput(
        implementation_ref=tool.implementation_ref,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        parameters={},
    )
    qc_policy = PNGPolicy(
        artifact_ref=candidate.output.artifact_ref,
        artifact_type_code="ARTIFACT:SYNTHETIC:REVIEWED_BYTE_SUM",
        unit=None,
        unit_term=candidate.output.unit,
        role="derived_product",
        labels=(0, 1),
        max_pixels=6,
    )
    qc_ref = ref("policy", "POLICY:SYNTHETIC:REVIEWED_BYTE_SUM_QC").model_copy(
        update=dict(checksum=digest(qc_policy))
    )
    if bundle is not None:
        from packages.tool_execution.fundus_bundle import FundusBundlePolicy, decode_bundle

        decoded, _ = decode_bundle(bundle)
        qc_policy = FundusBundlePolicy(
            artifact_ref=candidate.output.artifact_ref,
            artifact_type_code="ARTIFACT:SYNTHETIC:FUNDUS_BUNDLE",
            input_artifact_type_code=component_policy.input_binding.contract_type_code,
            unit_term=candidate.output.unit,
            source_checksum=decoded.source_checksum,
        )
        qc_ref = ref("policy", "POLICY:SYNTHETIC:REVIEWED_FUNDUS_BUNDLE_QC").model_copy(
            update=dict(checksum=digest(qc_policy))
        )
    allowlists = Allowlists()
    allowlists.values[manifest.checksum] = seal(
        CommandAllowlist,
        version="1.0.0",
        source_ref=ref("source_version", "SOURCE:SYNTHETIC:REVIEWED_JPEG_ALLOWLIST"),
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        compiler_config_checksum=ready.config.checksum,
        synthetic_only=False,
        executable_uri=ready.executable.as_uri(),
        executable_checksum=profile.runtime.interpreter_checksum,
        entrypoint_uri=ready.script.as_uri(),
        entrypoint_checksum=pin(ready.script).checksum,
        resources=execution_declaration.resources,
        argv=argv,
        parameter_schema_version="1.0.0",
        parameter_schema_checksum=digest(parameter_schema),
        outputs=[
            OutputSpec(
                port=candidate.output_port,
                filename="bundle.json" if bundle is not None else "byte-sum.png",
                qc_policy_ref=qc_ref,
            )
        ],
    )
    ready.__dict__.update(
        component_policy=component_policy,
        candidate=candidate,
        catalogs=catalogs,
        catalog=catalogs.catalog,
        compatibility=reseal(
            ready.compatibility, synthetic_only=False, accepted_validation=("smoke_tested",)
        ),
        invocation=invocation,
        manifest=manifest,
        profile=profile,
        allowlists=allowlists,
        qc=TechnicalQCRegistry({qc_ref: qc_policy}),
        resources=resources,
        metadata=metadata,
        review=review,
        record=record,
        audit=audit,
        native_source=source,
        legacy=legacy,
        bundle_qc=(qc_ref, qc_policy) if bundle is not None else None,
    )
    return ready


def assembly(database, tmp_path, key, *, bundle=None, task=None):
    inputs = service(database, tmp_path)
    registered = inputs.register_multimodal(
        prepare("gamma", gamma(tmp_path / "source"), "0001"), key=key + "-input"
    )
    visible, ready = registered.visible, deployment(tmp_path / "deployment", bundle=bundle)
    if bundle is not None:
        from packages.tool_execution.fundus_bundle import FundusBundleQCRegistry

        qc_ref, qc_policy = ready.bundle_qc
        ready.qc = FundusBundleQCRegistry({qc_ref: qc_policy}, inputs.visible_store)
    repo, sessions = ResearchRunRepository(database[2]), PlannerRepository(database[2])
    ctx = TaskRuns(runtime=repo, inputs=inputs).create(
        case_code=visible.case.case_code,
        key=key + "-parent",
        task=intent() if task is None else task,
        synthetic=True,
        policy_ref=dict(
            kind="policy",
            code="POLICY:SYNTHETIC:REVIEWED_COMPONENT_TEST",
            version="1.0.0",
            checksum=digest("reviewed synthetic CPU component test"),
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
        manifests=ready.catalogs,
        profiles=ready.owners,
        probes=ready.owners,
        inputs=owner,
        research=owner,
        objects=inputs.visible_store,
        resources=ready.resources,
        clock=lambda: NOW,
    )
    compiler = ToolWorkflowCompiler(admission, ready.allowlists, ready.config)
    # Original ReviewedCatalog is a native-shaped projection. The original child
    # context requires its synthetic flag to match the non-synthetic allowlist.
    research = ResearchRuntimeService(
        runtime=repo, research=owner, inputs=owner, compiler=compiler, synthetic=False
    )
    executors = {}

    def choice(request):
        result = admission.admit_research(request)
        if not result.retained_roots:
            raise ValueError(
                "reviewed_component_not_admitted:"
                + ",".join(sorted({r.reason for r in result.rejected}))
            )
        return single_choice(result)

    def executor_for(plan):
        if plan.checksum not in executors:
            executions = ResearchRuntimeExecutions(
                database[2], compiler=compiler, plans=(plan.to_bundle_json(),)
            )
            policy = ExecutionPolicy(
                reference=ref("policy", "POLICY:SYNTHETIC:REVIEWED_COMPONENT_CPU"),
                runtime_files=ready.files,
                allowed_plan_checksums=(plan.checksum,),
                purpose=plan.document.purpose,
                synthetic_execution=False,
                cpu_seconds=1,
            )
            executors[plan.checksum] = ToolExecutionService(
                compiler=compiler,
                catalogs=ready.catalogs,
                objects=inputs.visible_store,
                qc=ready.qc,
                repository=executions,
                policy=policy,
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
