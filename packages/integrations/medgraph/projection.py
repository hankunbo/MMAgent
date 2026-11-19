"""Deterministic conservative projections. No model inference or tool admission."""

import re

from packages.contracts import CoordinateSpace, Laterality, QCStatus
from packages.grounding.models import (
    ArtifactConstraint,
    CatalogSnapshot,
    CatalogTool,
    Input,
    PortRef,
    Ref,
    Term,
    canonical,
    digest,
    seal,
    unique,
)

from .models import (
    AdapterConfig,
    AdapterError,
    Binding,
    NativeManifest,
    NativeSnapshot,
    ProjectionPolicy,
    SnapshotEnvelope,
    json_copy,
    parse,
)

STATUS = {
    "registered": ("registered", "unvalidated"),
    "smoke_tested": ("registered", "smoke_tested"),
    "validated": ("registered", "smoke_tested"),
    "failed": ("failed", "unvalidated"),
    "deprecated": ("deprecated", "unvalidated"),
    "quarantined": ("quarantined", "unvalidated"),
}


def projected_status(status):
    return STATUS.get(status, ("quarantined", "unvalidated"))


def normalized(native):
    data = json_copy(parse(NativeSnapshot, native))
    # Only native relation arrays are sets; argv and opaque metadata arrays retain order.
    for manifest in data["manifests"]:
        manifest["resources"].sort(key=lambda r: r["resource_key"])
    for rows in data.values():
        rows.sort(key=canonical)
    value = parse(NativeSnapshot, data)
    for key, rows in data.items():
        if key != "manifests":
            unique(rows, lambda r: r["id"])
    unique(value.artifacts, lambda r: r.code)
    unique(value.tools, lambda r: r.code)
    unique(value.versions, lambda r: (r.tool_id, r.version))
    unique(value.implementations, lambda r: (r.tool_version_id, r.implementation_key))
    unique(value.ports, lambda r: (r.tool_version_id, r.direction, r.port_name))
    unique(value.resources, lambda r: (r.implementation_id, r.resource_key))
    unique(value.workflows, lambda r: (r.code, r.version))
    unique(value.steps, lambda r: (r.workflow_id, r.step_key))
    unique(value.edges, lambda r: (r.workflow_id, r.edge_key))
    unique(value.manifests, lambda r: (r.tool_code, r.version, r.implementation_key))
    # Reject orphans even if the source was not PostgreSQL (e.g. persisted tampering).
    ids = {k: {r["id"] for r in rows} for k, rows in data.items() if k != "manifests"}
    for relation, field, target in (
        ("versions", "tool_id", "tools"),
        ("ports", "tool_version_id", "versions"),
        ("ports", "artifact_type_id", "artifacts"),
        ("implementations", "tool_version_id", "versions"),
        ("resources", "implementation_id", "implementations"),
        ("steps", "workflow_id", "workflows"),
        ("steps", "tool_version_id", "versions"),
        ("edges", "workflow_id", "workflows"),
        ("edges", "artifact_type_id", "artifacts"),
    ):
        if any(r[field] not in ids[target] for r in data[relation]):
            raise AdapterError("native_orphan")
    step_scope = {s.id: s.workflow_id for s in value.steps}
    for edge in value.edges:
        for endpoint in (edge.source_step_id, edge.target_step_id):
            if endpoint is not None and step_scope.get(endpoint) != edge.workflow_id:
                raise AdapterError("native_workflow_scope_mismatch")
    return value


def adaptation_version(checksum):
    return "0.0." + str(int(checksum.removeprefix("sha256:")[:16], 16))


def external_ref(kind, identity, content, native_version=None):
    checksum = digest(content)
    prefix = {
        "artifact": "mg_a",
        "tool": "mg_t",
        "implementation": "mg_i",
        "workflow": "mg_w",
        "capability": "CAPABILITY:MEDGRAPH",
    }[kind]
    code = (
        prefix
        + ":"
        + (digest(identity)[7:].upper() if kind == "capability" else digest(identity)[7:])
    )
    version = (
        native_version
        if native_version and re.fullmatch(r"\d+\.\d+\.\d+", native_version)
        else adaptation_version(checksum)
    )
    return Ref(kind=kind, code=code, version=version, checksum=checksum)


def expected_manifest(native, implementation):
    version = next(v for v in native.versions if v.id == implementation.tool_version_id)
    tool = next(t for t in native.tools if t.id == version.tool_id)
    raw = json_copy(implementation)
    for key in ("id", "tool_version_id", "enabled"):
        raw.pop(key)
    raw.update(
        tool_code=tool.code,
        tool_name_zh=tool.name_zh,
        capability_kind=tool.kind,
        version=version.version,
    )
    raw["resources"] = [
        {k: v for k, v in json_copy(r).items() if k not in {"id", "implementation_id"}}
        for r in sorted(native.resources, key=lambda r: r.resource_key)
        if r.implementation_id == implementation.id
    ]
    return parse(NativeManifest, raw)


def implementation_content(native, implementation):
    version = next(v for v in native.versions if v.id == implementation.tool_version_id)
    tool = next(t for t in native.tools if t.id == version.tool_id)
    ports = tuple(p for p in native.ports if p.tool_version_id == version.id)
    workflows = {s.workflow_id for s in native.steps if s.tool_version_id == version.id}
    return dict(
        tool=tool,
        version=version,
        implementation=implementation,
        ports=ports,
        resources=tuple(r for r in native.resources if r.implementation_id == implementation.id),
        artifacts=tuple(a for a in native.artifacts if a.id in {p.artifact_type_id for p in ports}),
        workflows=tuple(w for w in native.workflows if w.id in workflows),
        steps=tuple(s for s in native.steps if s.workflow_id in workflows),
        edges=tuple(e for e in native.edges if e.workflow_id in workflows),
    )


def project(native, config, policy, *, synthetic=False):
    native = normalized(native)
    config = parse(AdapterConfig, config)
    policy = parse(ProjectionPolicy, policy)
    scope = config.source_code
    artifacts = {a.id: external_ref("artifact", [scope, a.code], a) for a in native.artifacts}
    versions = {v.id: v for v in native.versions}
    tool_rows = {t.id: t for t in native.tools}
    manifests = {(m.tool_code, m.version, m.implementation_key): m for m in native.manifests}
    bindings, tools = [], []
    for impl in native.implementations:
        version = versions[impl.tool_version_id]
        tool = tool_rows[version.tool_id]
        identity = [scope, tool.code, version.version, impl.implementation_key]
        content = implementation_content(native, impl)
        tool_ref = external_ref(
            "tool",
            identity[:3],
            {"tool": tool, "version": version, "ports": content["ports"]},
            version.version,
        )
        impl_ref = external_ref(
            "implementation", identity, {"native": content, "config": config, "policy": policy}
        )
        actual = manifests.pop(tuple(identity[1:]), None)
        if actual != (expected_manifest(native, impl) if impl.enabled else None):
            raise AdapterError("public_manifest_source_drift")
        gaps = [
            "applicability_not_admitted",
            "validation_gates_missing",
            "parameter_schema_not_admitted",
            "pre_post_processing_not_admitted",
            "resource_bytes_not_verified",
        ]
        if not any(d.tool_code == tool.code for d in policy.declarations):
            gaps.append("native_intended_use_missing_effective_research_ceiling_only")
        if impl.validation_status not in STATUS:
            gaps.append("unknown_validation_status")
        if impl.validation_status == "validated":
            gaps.append("legacy_validated_requires_reevaluation")
        if any(r.sha256 is None for r in content["resources"]):
            gaps.append("resource_checksum_missing")
        binding = Binding(
            tool_code=tool.code,
            native_version=version.version,
            implementation_key=impl.implementation_key,
            tool_id=tool.id,
            tool_version_id=version.id,
            implementation_id=impl.id,
            tool_ref=tool_ref,
            implementation_ref=impl_ref,
            content_checksum=digest(content),
            gaps=tuple(sorted(gaps)),
        )
        bindings.append(binding)
        inputs, outputs, output_ports = [], [], []
        for port in sorted(content["ports"], key=lambda p: (p.direction, p.port_name)):
            # No implicit mapping from free-text constraints to controlled medical terms.
            constraint = ArtifactConstraint(
                artifact_ref=artifacts[port.artifact_type_id],
                modality=Term(state="unknown", code=None),
                unit=Term(state="unknown", code=None),
                laterality=Laterality.UNKNOWN,
                coordinate_space=CoordinateSpace.UNKNOWN,
                qc_required=(QCStatus.PASS,),
                scope_terms=(),
            )
            p = PortRef(tool_ref=tool_ref, direction=port.direction, port_name=port.port_name)
            if port.direction == "input":
                if port.cardinality_max is None or port.cardinality_max > 64:
                    raise AdapterError("unbounded_native_cardinality")
                inputs.append(
                    Input(
                        port=p,
                        constraint=constraint,
                        required=port.required,
                        cardinality_min=port.cardinality_min,
                        cardinality_max=port.cardinality_max,
                    )
                )
            else:
                outputs.append(constraint)
                output_ports.append(p)
        status, validation = projected_status(impl.validation_status)
        if not impl.enabled or not version.enabled:
            status = "quarantined" if status not in {"failed", "deprecated"} else status
        capabilities = tuple(
            external_ref(
                "capability",
                [scope, tool.code, p.port_name],
                {"kind": tool.kind, "tool": tool_ref, "output": p},
            )
            for p in output_ports
        )
        workflows = tuple(workflow_reference(native, w, scope) for w in content["workflows"])
        tools.append(
            parse(
                CatalogTool,
                dict(
                    tool_ref=tool_ref,
                    implementation_ref=impl_ref,
                    capabilities=capabilities,
                    inputs=inputs,
                    outputs=outputs,
                    output_ports=output_ports,
                    workflows=workflows,
                    enabled=False,
                    catalog_status=status,
                    validation_status=validation,
                    intended_use="research_only",
                ),
            )
        )
    if manifests:
        raise AdapterError("unbound_native_manifest")
    material = {"native": native, "policy": policy, "config": config}
    catalog = seal(
        CatalogSnapshot,
        code="CATALOG:MEDGRAPH:" + scope,
        version=adaptation_version(digest(material)),
        synthetic=synthetic,
        manifest_checksum=digest(material),
        artifacts=sorted(artifacts.values(), key=canonical),
        tools=sorted(tools, key=lambda t: canonical(t.implementation_ref)),
        estimates=(),
    )
    return seal(
        SnapshotEnvelope,
        native=native,
        config=config,
        policy=policy,
        catalog=catalog,
        bindings=sorted(bindings, key=lambda b: canonical(b.implementation_ref)),
    )


def workflow_reference(native, workflow, source_code):
    return external_ref(
        "workflow",
        [source_code, workflow.code, workflow.version],
        {
            "workflow": workflow,
            "steps": tuple(s for s in native.steps if s.workflow_id == workflow.id),
            "edges": tuple(e for e in native.edges if e.workflow_id == workflow.id),
        },
        workflow.version,
    )
