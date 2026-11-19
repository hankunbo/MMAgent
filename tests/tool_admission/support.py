"""Explicit synthetic deployment and ingestion owners. No executable tool fixtures."""

import struct
import zlib
from datetime import timedelta
from types import SimpleNamespace

from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding.models import digest, seal, validate
from packages.grounding.synthetic import ref
from packages.integrations.medgraph.models import DeploymentManifest
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import LocalObjectStore, checksum
from packages.tool_admission import (
    AdmissionSelection,
    DeploymentProfile,
    EnvironmentProbe,
    InputRecord,
    InvocationInput,
    ToolAdmissionService,
)
from packages.tool_admission.models import ArtifactTypeBinding, OutputTargetBinding
from tests.grounding.query_support import NOW, paths, ready, updated


def png(height=2, width=3):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * ((width + 1) * height)))
        + chunk(b"IEND", b"")
    )


def reseal(value, **changes):
    data = value.model_dump(mode="json")
    data.pop("checksum")
    return seal(type(value), **{**data, **changes})


class Owners:
    def __init__(self):
        self.records, self.manifests, self.profiles, self.probes, self.requests = {}, {}, {}, {}, {}
        self.reads = []

    def read_input(self, scope, run_id, code):
        self.reads.append(code)
        return self.records[code]

    def read_manifest(self, catalog, tool, implementation):
        return self.manifests[(catalog, tool, implementation)]

    def read_profile(self, manifest):
        return self.profiles[manifest.checksum]

    def read_probe(self, manifest, profile):
        return self.probes[profile.checksum]

    def read_request(self, code, version, checksum):
        return self.requests[(code, version, checksum)]


class Locks:
    """Synthetic Runtime read fixture; actual GroundingQueryService is never mocked."""

    def __init__(self, q, mapping, policy, knowledge, catalogs):
        self.q, self.mapping, self.policy, self.knowledge, self.catalogs = (
            q,
            mapping,
            policy,
            knowledge,
            catalogs,
        )

    def read(self, run_id):
        assert run_id == self.q.lock.run_id
        return self.q.lock

    def validate_query_scope(self, q):
        if q.scope != self.q.scope:
            raise ValueError("synthetic_scope_mismatch")

    def query_paths(self, run_id, mappings, context):
        return paths(self.mapping, self.policy, self.knowledge, self.catalogs, context)


def register_inputs(q, owners, objects):
    metadata = []
    for m in q.artifacts:
        data = png()
        uri = objects.promote(objects.stage(data), checksum(data))
        m = validate(type(m), {**m.model_dump(mode="json"), "checksum": checksum(data)})
        a = dict(
            schema_version="1.0",
            contract_type="Artifact",
            asset_code=m.asset_code,
            artifact_type_code="ARTIFACT:SYNTHETIC:INPUT",
            uri=uri,
            checksum=m.checksum,
            media_type="image/png",
            qc_status=m.qc_status,
            dimensions=[2, 3],
            research_only=True,
            scope=dict(scope_type="CASE", case_code=m.case_code),
            laterality=m.laterality,
            coordinate_space=m.coordinate_space,
            unit="1" if m.unit.state == "known" else None,
            provenance=dict(
                actor="RULE_ENGINE",
                created_at=NOW,
                generator_version="synthetic.1.0.0",
                source_codes=["SOURCE:SYNTHETIC:INPUT"],
                input_checksums=[m.checksum],
            ),
        )
        owners.records[m.asset_code] = validate(
            InputRecord,
            dict(
                artifact=a,
                type_binding=seal(
                    ArtifactTypeBinding,
                    version="1.0.0",
                    artifact_ref=m.artifact_ref,
                    contract_type_code="ARTIFACT:SYNTHETIC:INPUT",
                    unit_term=m.unit,
                    contract_unit="1" if m.unit.state == "known" else None,
                    source_ref=ref("source_version", "SOURCE:SYNTHETIC:TYPE_BINDING"),
                ),
                metadata=m,
                byte_size=len(data),
                content_kind="image",
                source_ref=ref("source_version", "SOURCE:SYNTHETIC:INPUT"),
            ),
        )
        metadata.append(m)
    return updated(q, artifacts=metadata)


def register_deployments(q, catalog, owners, root):
    interpreter = root / "synthetic-interpreter.bin"
    interpreter.write_bytes(b"synthetic nonexecutable runtime bytes")
    invocations = []
    for n, tool in enumerate(catalog.tools, 1):
        binding = dict(
            tool_code=tool.tool_ref.code,
            native_version=tool.tool_ref.version,
            implementation_key=tool.implementation_ref.code,
            tool_id=n,
            tool_version_id=n,
            implementation_id=n,
            tool_ref=tool.tool_ref,
            implementation_ref=tool.implementation_ref,
            content_checksum=tool.implementation_ref.checksum,
            gaps=[],
        )
        invocation = dict(shell=False, argv=["synthetic-nonexecutable"], timeout_seconds=1)
        runtime = dict(
            interpreter_uri=interpreter.as_uri(),
            interpreter_checksum=checksum(interpreter.read_bytes()),
            dependencies={"synthetic": "1.0.0"},
            gpu="not_required",
        )
        native = dict(
            tool_code=tool.tool_ref.code,
            tool_name_zh="合成",
            capability_kind="synthetic",
            version=tool.tool_ref.version,
            implementation_key=tool.implementation_ref.code,
            implementation_kind="synthetic",
            entrypoint_uri=interpreter.as_uri(),
            working_directory=None,
            runtime_environment={"admission_runtime": runtime},
            invocation=invocation,
            formula_spec=None,
            hardware_requirements={"gpu": False},
            provenance={"synthetic": True},
            resources=[],
            validation_status=tool.validation_status.value,
            validation_report={"synthetic": True},
        )
        ports = [
            dict(
                id=n * 100 + j,
                tool_version_id=n,
                direction=i.port.direction,
                port_name=i.port.port_name,
                artifact_type_id=j + 1,
                required=i.required,
                cardinality_min=i.cardinality_min,
                cardinality_max=i.cardinality_max,
                constraints={},
            )
            for j, i in enumerate(tool.inputs, 1)
        ]
        ports += [
            dict(
                id=n * 100 + 90 + j,
                tool_version_id=n,
                direction="output",
                port_name=p.port_name,
                artifact_type_id=j + 20,
                required=True,
                cardinality_min=1,
                cardinality_max=1,
                constraints={},
            )
            for j, p in enumerate(tool.output_ports)
        ]
        processing = {
            "version": "1.0.0",
            "policy_code": "POLICY:SYNTHETIC:IDENTITY",
            "operations": [],
        }
        manifest = seal(
            DeploymentManifest,
            catalog_ref=catalog.reference(),
            tool_ref=tool.tool_ref,
            implementation_ref=tool.implementation_ref,
            binding=binding,
            native_manifest=native,
            invocation=invocation,
            parameter_schema={
                "type": "object",
                "properties": {"threshold": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": [],
                "additionalProperties": False,
            },
            preprocessing_policy=processing,
            postprocessing_policy=processing,
            native_manifest_state="visible",
            ports=ports,
            config_checksum=digest("synthetic-config"),
            policy_checksum=digest("synthetic-projection"),
        )
        profile = seal(
            DeploymentProfile,
            version="1.0.0",
            manifest_checksum=manifest.checksum,
            config_checksum=manifest.config_checksum,
            policy_checksum=manifest.policy_checksum,
            input_rules=[
                dict(
                    port=i.port,
                    media_types=["image/png"],
                    dimensions_min=[1, 1],
                    dimensions_max=[4096, 4096],
                    anatomical_targets=["CONCEPT:SYNTHETIC"],
                )
                for i in tool.inputs
            ],
            output_rules=[
                dict(
                    port=p,
                    media_types=["image/png"],
                    dimensions_min=[1, 1],
                    dimensions_max=[4096, 4096],
                    anatomical_targets=["CONCEPT:SYNTHETIC"],
                )
                for p in tool.output_ports
            ],
            output_target_bindings=[
                seal(
                    OutputTargetBinding,
                    version="1.0.0",
                    source_ref=ref("source_version", "SOURCE:SYNTHETIC:OUTPUT_ANATOMY"),
                    port=p,
                    constraint=constraint,
                    anatomical_targets=["CONCEPT:SYNTHETIC"],
                )
                for p, constraint in zip(tool.output_ports, tool.outputs, strict=True)
            ],
            preprocessing_checksum=digest(processing),
            postprocessing_checksum=digest(processing),
            runtime=dict(
                interpreter_uri=interpreter.as_uri(),
                interpreter_checksum=checksum(interpreter.read_bytes()),
                dependencies={"synthetic": "1.0.0"},
                gpu="not_required",
            ),
        )
        probe = seal(
            EnvironmentProbe,
            version="1.0.0",
            manifest_checksum=manifest.checksum,
            profile_checksum=profile.checksum,
            runtime_checksum=digest(profile.runtime),
            source_ref=ref("source_version", "SOURCE:SYNTHETIC:PROBE"),
            observed_at=NOW,
            expires_at=NOW + timedelta(days=1),
            interpreter="available",
            dependencies="available",
            gpu="not_required",
            resources="available",
            preconditions="satisfied",
        )
        owners.manifests[(catalog.reference(), tool.tool_ref, tool.implementation_ref)] = manifest
        owners.profiles[manifest.checksum] = profile
        owners.probes[profile.checksum] = probe
        invocations.append(
            InvocationInput(
                implementation_ref=tool.implementation_ref,
                manifest_checksum=manifest.checksum,
                profile_checksum=profile.checksum,
                parameters={},
            )
        )
    return tuple(invocations)


def fixture(tmp_path, *, chain=False):
    q, mapping, policy, catalog, knowledge, catalogs = ready(chain=chain)
    owners = Owners()
    objects = LocalObjectStore(tmp_path / "objects")
    q = register_inputs(q, owners, objects)
    locks = Locks(q, mapping, policy, knowledge, catalogs)
    grounding = GroundingQueryService(
        locks, SimpleNamespace(catalogs=catalogs, knowledge=knowledge)
    )
    invocations = register_deployments(q, catalog, owners, tmp_path)
    service = ToolAdmissionService(
        grounding=grounding,
        catalogs=catalogs,
        manifests=owners,
        profiles=owners,
        probes=owners,
        inputs=owners,
        objects=objects,
        resources=LocalResourcePolicy(allowed_roots=(str(tmp_path),)),
        research=owners,
        clock=lambda: NOW,
    )
    result = grounding.query(q)
    selection = AdmissionSelection(
        candidate_refs=tuple(
            c.identity for c in result.candidates if not c.candidate.code.endswith("INTERMEDIATE")
        ),
        invocations=invocations,
    )
    return SimpleNamespace(
        q=q,
        mapping=mapping,
        policy=policy,
        catalog=catalog,
        knowledge=knowledge,
        catalogs=catalogs,
        owners=owners,
        objects=objects,
        locks=locks,
        grounding=grounding,
        service=service,
        result=result,
        selection=selection,
    )
