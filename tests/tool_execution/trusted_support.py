"""Harmless registered research fixture using real 06B/06C/06D code.

The catalog's false synthetic flag exercises production rejection guards only in
this isolated test fixture. It is never persisted as a medical catalog or CUDA test.
"""

from pathlib import Path
from types import SimpleNamespace

from packages.grounding.models import digest, seal, validate
from packages.grounding.ports import ToolSafety
from packages.grounding.synthetic import ref
from packages.tool_admission.models import InvocationInput, ResearchRequest
from packages.tool_execution import PNGPolicy, TechnicalQCRegistry
from packages.tool_execution.trusted_models import TrustedLocalPolicy, TrustedToolReview
from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService
from packages.tool_workflow import CommandAllowlist, ResearchContext, ToolWorkflowCompiler
from packages.tool_workflow.models import OutputSpec, TrustedResearchCompilerConfig
from tests.grounding.query_support import NOW
from tests.tool_admission.support import fixture, reseal
from tests.tool_execution.test_dependencies import TestLedger
from tests.tool_execution.test_trusted_local import FakeGPU, pinned
from tests.tool_workflow.support import Allowlists, choose


def prepared(fixture_setup):
    f = fixture(fixture_setup.root)
    catalog = reseal(f.catalog, synthetic=False)

    class Catalogs:
        disabled = ()

        def read_snapshot(self, reference):
            assert reference == catalog.reference()
            return catalog

        def current_safety(self, reference):
            self.read_snapshot(reference)
            return ToolSafety(
                catalog_ref=reference, disabled_implementations=self.disabled, revision=0
            )

    catalogs = Catalogs()
    f.service.catalogs = catalogs
    cfg = seal(
        TrustedResearchCompilerConfig,
        max_steps=2,
        max_plan_bytes=8388608,
        timeout_seconds=2,
        memory_bytes=134217728,
        cpu_count=1,
        max_output_bytes=4194304,
    )
    candidate = f.mapping.candidates[0]
    old = next(
        m
        for m in f.owners.manifests.values()
        if m.implementation_ref == candidate.implementation_ref
    )
    profile = f.owners.profiles[old.checksum]
    old_probe = f.owners.probes[profile.checksum]
    # This program consumes the bound input; it is a copy operation, not segmentation.
    script = fixture_setup.script
    script.write_text(
        "import sys\ndata=[]\nfor path in sys.argv[1:-1]:\n"
        ' with open(path, "rb") as f: data.append(f.read())\n'
        'with open(sys.argv[-1], "xb") as f: f.write(data[0])\n'
    )
    runtime = dict(
        interpreter_uri=Path(fixture_setup.review.executable.path).as_uri(),
        interpreter_checksum=fixture_setup.review.executable.checksum,
        dependencies={},
        gpu="required",
    )
    resource = dict(
        resource_key="entrypoint",
        resource_role="source_code",
        uri=script.as_uri(),
        sha256=pinned(script).checksum[7:],
        byte_size=script.stat().st_size,
        required=True,
        metadata={"test_fixture": True},
    )
    argv = [fixture_setup.review.executable.path, str(script)]
    argv += ["{input:" + i.port.port_name + "}" for i in candidate.inputs]
    argv += ["{output:" + candidate.output_port.port_name + "}"]
    invocation = dict(shell=False, argv=argv, timeout_seconds=2)
    native = {
        **old.native_manifest.model_dump(mode="json"),
        "entrypoint_uri": script.as_uri(),
        "resources": [resource],
        "invocation": invocation,
        "runtime_environment": {"admission_runtime": runtime},
    }
    manifest = reseal(
        old, catalog_ref=catalog.reference(), native_manifest=native, invocation=invocation
    )
    profile = reseal(profile, manifest_checksum=manifest.checksum, runtime=runtime)
    probe = reseal(
        old_probe,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        runtime_checksum=digest(profile.runtime),
        gpu="available",
    )
    f.owners.manifests[(catalog.reference(), manifest.tool_ref, manifest.implementation_ref)] = (
        manifest
    )
    f.owners.profiles[manifest.checksum] = profile
    f.owners.probes[profile.checksum] = probe
    inv = InvocationInput(
        implementation_ref=manifest.implementation_ref,
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        parameters={},
    )
    request = seal(
        ResearchRequest,
        code="RESEARCH:TEST:TRUSTED",
        version="1.0.0",
        purpose="shadow",
        compatibility_policy=reseal(f.policy, synthetic_only=False),
        lock=dict(run_id=f.q.lock.run_id, catalog_ref=catalog.reference()),
        scope=f.q.scope,
        window_start=f.q.window_start,
        window_end=f.q.window_end,
        snapshot_at=NOW,
        satisfied_terms=f.q.satisfied_terms,
        artifacts=f.q.artifacts,
        candidate=candidate,
        inputs=[
            dict(
                port=i.port,
                asset_codes=[
                    a.asset_code
                    for a in f.q.artifacts
                    if a.artifact_ref == i.constraint.artifact_ref
                ],
            )
            for i in candidate.inputs
        ],
        invocation=inv,
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    admission = f.service.admit_research(request)
    assert admission.status == "preparation_only", admission.model_dump_json(indent=2)
    binding = profile.output_target_bindings[0]
    qc_policy = PNGPolicy(
        artifact_ref=binding.constraint.artifact_ref,
        artifact_type_code="ARTIFACT:TEST:TRUSTED",
        unit="1",
        unit_term=binding.constraint.unit,
        role="derived_product",
        labels=(0, 1),
        max_pixels=16,
    )
    qc_ref = ref("policy", "POLICY:TEST:PNG").model_copy(update={"checksum": digest(qc_policy)})
    allowlist = seal(
        CommandAllowlist,
        version="1.0.0",
        source_ref=ref("source_version", "SOURCE:TEST:REVIEW"),
        manifest_checksum=manifest.checksum,
        profile_checksum=profile.checksum,
        compiler_config_checksum=cfg.checksum,
        synthetic_only=False,
        executable_uri=runtime["interpreter_uri"],
        executable_checksum=runtime["interpreter_checksum"],
        entrypoint_uri=script.as_uri(),
        entrypoint_checksum=pinned(script).checksum,
        resources=[resource],
        argv=argv,
        parameter_schema_version="1.0.0",
        parameter_schema_checksum=digest(manifest.parameter_schema),
        outputs=[
            OutputSpec(port=candidate.output_port, filename="output.png", qc_policy_ref=qc_ref)
        ],
    )
    allowlists = Allowlists()
    allowlists.values[manifest.checksum] = allowlist
    compiler = ToolWorkflowCompiler(f.service, allowlists, cfg)
    context = ResearchContext(request=request, admission=admission)
    plan = compiler.compile(context, choose(admission))
    review = validate(
        TrustedToolReview,
        {
            **fixture_setup.review.model_dump(mode="json"),
            "implementation_ref": candidate.implementation_ref.model_dump(mode="json"),
            "manifest_checksum": manifest.checksum,
            "allowlist_checksum": allowlist.checksum,
            "entrypoint": pinned(script).model_dump(mode="json"),
            "files": [
                (pinned(script) if x.path == str(script) else x).model_dump(mode="json")
                for x in fixture_setup.review.files
            ],
        },
    )
    policy = validate(
        TrustedLocalPolicy,
        {
            **fixture_setup.policy.model_dump(mode="json"),
            "allowed_plan_checksums": [plan.checksum],
            "reviews": [review.model_dump(mode="json")],
        },
    )
    repo = TestLedger()
    service = TrustedLocalGPUResearchService(
        compiler=compiler,
        catalogs=catalogs,
        objects=f.objects,
        qc=TechnicalQCRegistry({qc_ref: qc_policy}),
        repository=repo,
        policy=policy,
        workspace=fixture_setup.root / "service-attempts",
    )
    service._runner.observation = FakeGPU()
    return SimpleNamespace(
        service=service,
        plan=plan,
        repo=repo,
        compiler=compiler,
        request=request,
        f=f,
        catalogs=catalogs,
    )
