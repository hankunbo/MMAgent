"""Synthetic nonexecuting allowlist fixture, never a deployed medical runner."""

from types import SimpleNamespace

from packages.grounding.models import digest, seal, validate
from packages.grounding.synthetic import ref
from packages.object_store import checksum
from packages.tool_admission.models import AdmissionSelection, InvocationInput
from packages.tool_workflow import (
    CommandAllowlist,
    CompilerConfig,
    FormalContext,
    ToolWorkflowCompiler,
    WorkflowChoice,
)
from packages.tool_workflow.models import OutputSpec
from tests.tool_admission.support import fixture, reseal


class Allowlists:
    def __init__(self):
        self.values = {}

    def read_allowlist(self, manifest, profile, config):
        return self.values[manifest.checksum]


def config():
    return seal(
        CompilerConfig,
        version="1.0.0",
        max_steps=128,
        max_plan_bytes=8388608,
        timeout_seconds=10,
        memory_bytes=134217728,
        cpu_count=1,
        max_output_bytes=4194304,
    )


def commands(f, root, cfg, *, parameters=None, schema=None, argv_tail=(), catalog_ref=None):
    reader, invocations = Allowlists(), []
    script = root / "synthetic-entrypoint.txt"
    script.write_bytes(b"SYNTHETIC OFFLINE FIXTURE - not executable model code\n")
    for old in list(f.owners.manifests.values()):
        if old.catalog_ref != (catalog_ref or f.q.lock.catalog_ref):
            continue
        profile = f.owners.profiles[old.checksum]
        probe = f.owners.probes[profile.checksum]
        from packages.tool_workflow.compiler import local_path

        argv = [local_path(profile.runtime.interpreter_uri), str(script)]
        argv += ["{input:" + p.port_name + "}" for p in old.ports if p.direction == "input"]
        argv += ["{output:" + p.port_name + "}" for p in old.ports if p.direction == "output"]
        argv += list(argv_tail)
        resource = dict(
            resource_key="entrypoint",
            resource_role="source_code",
            uri=script.as_uri(),
            sha256=checksum(script.read_bytes()).split(":")[1],
            byte_size=script.stat().st_size,
            required=True,
            metadata={"synthetic": True},
        )
        invocation = dict(shell=False, argv=argv, timeout_seconds=1)
        native = {
            **old.native_manifest.model_dump(mode="json"),
            "entrypoint_uri": script.as_uri(),
            "resources": [resource],
            "invocation": invocation,
        }
        manifest = reseal(
            old,
            native_manifest=native,
            invocation=invocation,
            parameter_schema=schema or old.parameter_schema,
        )
        profile = reseal(profile, manifest_checksum=manifest.checksum)
        probe = reseal(
            probe, manifest_checksum=manifest.checksum, profile_checksum=profile.checksum
        )
        f.owners.manifests[
            (manifest.catalog_ref, manifest.tool_ref, manifest.implementation_ref)
        ] = manifest
        f.owners.profiles[manifest.checksum] = profile
        f.owners.probes[profile.checksum] = probe
        invocations.append(
            InvocationInput(
                implementation_ref=manifest.implementation_ref,
                manifest_checksum=manifest.checksum,
                profile_checksum=profile.checksum,
                parameters=parameters or {},
            )
        )
        reader.values[manifest.checksum] = seal(
            CommandAllowlist,
            version="1.0.0",
            source_ref=ref("source_version", "SOURCE:SYNTHETIC:ALLOWLIST"),
            manifest_checksum=manifest.checksum,
            profile_checksum=profile.checksum,
            compiler_config_checksum=cfg.checksum,
            synthetic_only=True,
            executable_uri=profile.runtime.interpreter_uri,
            executable_checksum=profile.runtime.interpreter_checksum,
            entrypoint_uri=script.as_uri(),
            entrypoint_checksum=checksum(script.read_bytes()),
            resources=[resource],
            argv=argv,
            parameter_schema_version="1.0.0",
            parameter_schema_checksum=digest(manifest.parameter_schema),
            outputs=[
                OutputSpec(
                    port=r.port,
                    filename=r.port.port_name + ".png",
                    qc_policy_ref=ref("policy", "POLICY:SYNTHETIC:TECHNICAL_QC"),
                )
                for r in profile.output_rules
            ],
        )
    return reader, tuple(invocations)


def choose(admission, *, all_roots=False, prefer_producers=False):
    nodes = {n.identity: n for n in admission.retained}
    selected = {}
    roots = sorted(admission.retained_roots)
    if not all_roots:
        roots = roots[:1]

    def visit(identity):
        if identity in selected:
            return
        node = nodes[identity]
        inputs = []
        selected[identity] = dict(candidate_ref=identity, inputs=inputs)
        for p in node.inputs:
            sources = [dict(kind="artifact", asset_code=a.asset_code) for a in p.artifacts]
            producers = [
                dict(kind="producer", candidate_ref=i, port=nodes[i].candidate.output_port)
                for i in p.producer_refs
            ]
            sources = (producers + sources if prefer_producers else sources + producers)[
                : p.cardinality_min
            ]
            inputs.append(dict(port=p.port, sources=sources))
            for s in sources:
                if s["kind"] == "producer":
                    visit(s["candidate_ref"])

    for r in roots:
        visit(r)
    return validate(WorkflowChoice, dict(roots=roots, nodes=list(selected.values())))


def prepared(root, *, chain=False, base=None, **kwargs):
    f = base or fixture(root, chain=chain)
    cfg = config()
    reader, invocations = commands(f, root, cfg, **kwargs)
    f.selection = AdmissionSelection(
        candidate_refs=f.selection.candidate_refs, invocations=invocations
    )
    admission = f.service.admit(f.q, f.result, f.selection)
    assert admission.status == "preparation_only", admission.rejected
    context = FormalContext(
        request=f.q, query_result=f.result, selection=f.selection, admission=admission
    )
    return SimpleNamespace(
        f=f,
        context=context,
        choice=choose(admission),
        config=cfg,
        reader=reader,
        compiler=ToolWorkflowCompiler(f.service, reader, cfg),
    )
