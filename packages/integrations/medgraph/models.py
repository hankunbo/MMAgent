"""Native reader and deployment values; Grounding retains its only Catalog model."""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from packages.grounding.models import (
    CatalogSnapshot,
    Checksum,
    Ref,
    Value,
    Version,
    canonical,
    check_seal,
    unique,
)

Text = Annotated[str, Field(min_length=1, max_length=512)]
Object = dict[str, JsonValue]


class AdapterError(ValueError):
    """Fixed safe reason, never a DSN, SQL row or credential-bearing exception."""


def inspect_json(value, depth=0):
    if depth > 20:
        raise AdapterError("native_depth_limit")
    if isinstance(value, dict):
        if len(value) > 512:
            raise AdapterError("native_object_limit")
        for key, item in value.items():
            if re.search(
                r"password|secret|authorization|api_key|access_token|database_url|dsn", key, re.I
            ):
                raise AdapterError("credential_field_forbidden")
            inspect_json(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) > 4096:
            raise AdapterError("native_array_limit")
        for item in value:
            inspect_json(item, depth + 1)
    elif isinstance(value, str):
        if len(value) > 65536:
            raise AdapterError("native_string_limit")
        if re.search(r"://[^/\s]*@|(?:password|api_key|token|secret)=|Bearer\s", value, re.I):
            raise AdapterError("inline_credential_forbidden")


class NativeValue(Value):
    model_config = ConfigDict(serialize_by_alias=True)

    @model_validator(mode="after")
    def bounded(self):
        data = self.model_dump(mode="json")
        inspect_json(data)
        if len(canonical(data)) > 262144:
            raise AdapterError("native_record_byte_limit")
        return self


class ArtifactType(NativeValue):
    id: int = Field(gt=0)
    code: Text
    name_zh: Text
    description: str | None
    schema_document: Object = Field(alias="schema_json")


class Tool(NativeValue):
    id: int = Field(gt=0)
    code: Text
    name_zh: Text
    kind: Text
    description: str | None
    metadata: Object


class ToolVersion(NativeValue):
    id: int = Field(gt=0)
    tool_id: int = Field(gt=0)
    version: Text
    runtime: Object
    metadata: Object
    enabled: bool


class Port(NativeValue):
    id: int = Field(gt=0)
    tool_version_id: int = Field(gt=0)
    direction: Literal["input", "output"]
    port_name: Text
    artifact_type_id: int = Field(gt=0)
    required: bool
    cardinality_min: int = Field(ge=0)
    cardinality_max: int | None = Field(ge=0)
    constraints: Object


class Implementation(NativeValue):
    id: int = Field(gt=0)
    tool_version_id: int = Field(gt=0)
    implementation_key: Text
    implementation_kind: Text
    entrypoint_uri: Text
    working_directory: str | None
    runtime_environment: Object
    invocation: Object
    formula_spec: Object | None
    hardware_requirements: Object
    provenance: Object
    enabled: bool
    validation_status: Text
    validation_report: Object


class Resource(NativeValue):
    id: int = Field(gt=0)
    implementation_id: int = Field(gt=0)
    resource_key: Text
    resource_role: Text
    uri: Text
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    byte_size: int | None = Field(ge=0)
    required: bool
    metadata: Object


class Workflow(NativeValue):
    id: int = Field(gt=0)
    code: Text
    version: Text
    name_zh: Text
    description: str | None
    status: Text
    metadata: Object


class Step(NativeValue):
    id: int = Field(gt=0)
    workflow_id: int = Field(gt=0)
    step_key: Text
    tool_version_id: int = Field(gt=0)
    position_hint: int | None
    config: Object


class Edge(NativeValue):
    id: int = Field(gt=0)
    workflow_id: int = Field(gt=0)
    edge_key: Text
    source_step_id: int | None
    source_port: Text
    target_step_id: int | None
    target_port: Text
    artifact_type_id: int = Field(gt=0)
    conditions: Object


class ManifestResource(NativeValue):
    resource_key: Text
    resource_role: Text
    uri: Text
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    byte_size: int | None = Field(ge=0)
    required: bool
    metadata: Object


class NativeManifest(NativeValue):
    """Exactly the public SQL function result: deliberately no enabled or ports."""

    tool_code: Text
    tool_name_zh: Text
    capability_kind: Text
    version: Text
    implementation_key: Text
    implementation_kind: Text
    entrypoint_uri: Text
    working_directory: str | None
    runtime_environment: Object
    invocation: Object
    formula_spec: Object | None
    hardware_requirements: Object
    provenance: Object
    resources: tuple[ManifestResource, ...] = Field(max_length=128)
    validation_status: Text
    validation_report: Object


class NativeSnapshot(Value):
    artifacts: tuple[ArtifactType, ...] = Field(max_length=256)
    tools: tuple[Tool, ...] = Field(max_length=128)
    versions: tuple[ToolVersion, ...] = Field(max_length=128)
    ports: tuple[Port, ...] = Field(max_length=4096)
    implementations: tuple[Implementation, ...] = Field(max_length=128)
    resources: tuple[Resource, ...] = Field(max_length=4096)
    workflows: tuple[Workflow, ...] = Field(max_length=128)
    steps: tuple[Step, ...] = Field(max_length=4096)
    edges: tuple[Edge, ...] = Field(max_length=4096)
    manifests: tuple[NativeManifest, ...] = Field(max_length=128)


class AdapterConfig(Value):
    version: Version = "1.0.0"
    baseline_only: Literal[True] = True
    source_code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,31}$")]
    dsn_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
    connect_timeout_seconds: int = Field(default=5, ge=1, le=10)
    statement_timeout_ms: int = Field(default=5000, ge=1, le=10000)
    transaction_timeout_ms: int = Field(default=30000, ge=1, le=60000)
    max_rows: int = Field(default=512, ge=1, le=4096)
    max_snapshot_bytes: int = Field(default=8388608, ge=1024, le=16777216)


class UseDeclaration(Value):
    """Source-backed effective research restriction, never a validation Gate."""

    tool_code: Text
    source_path: Text
    source_checksum: Checksum
    excerpt: Text
    intended_use: Literal["research_only"]


class ProjectionPolicy(Value):
    version: Literal["1.0.0"] = "1.0.0"
    status_policy: Literal["legacy-conservative/1.0.0"] = "legacy-conservative/1.0.0"
    resource_policy: Literal["local-tree/1.0.0"] = "local-tree/1.0.0"
    effective_use_ceiling: Literal["research_only"] = "research_only"
    declarations: tuple[UseDeclaration, ...] = Field(default=(), max_length=128)


class Binding(Value):
    tool_code: Text
    native_version: Text
    implementation_key: Text
    tool_id: int
    tool_version_id: int
    implementation_id: int
    tool_ref: Ref
    implementation_ref: Ref
    content_checksum: Checksum
    gaps: tuple[Text, ...] = Field(max_length=128)


class SnapshotEnvelope(Value):
    schema_version: Literal["medgraph-derived/1.0.0"] = "medgraph-derived/1.0.0"
    checksum: Checksum
    native: NativeSnapshot
    config: AdapterConfig
    policy: ProjectionPolicy
    catalog: CatalogSnapshot
    bindings: tuple[Binding, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def integrity(self):
        unique(self.bindings, lambda b: b.implementation_ref)
        check_seal(self)
        return self


class Invocation(Value):
    shell: Literal[False]
    argv: tuple[Annotated[str, Field(min_length=1, max_length=4096)], ...] = Field(
        min_length=1, max_length=128
    )
    timeout_seconds: int = Field(ge=1, le=86400)


class DeploymentManifest(Value):
    schema_version: Literal["medgraph-deployment/1.0.0"] = "medgraph-deployment/1.0.0"
    checksum: Checksum
    catalog_ref: Ref
    tool_ref: Ref
    implementation_ref: Ref
    binding: Binding
    native_manifest: NativeManifest | None
    invocation: Invocation | None
    parameter_schema: Object | None
    preprocessing_policy: Object | None
    postprocessing_policy: Object | None
    native_manifest_state: Literal["visible", "disabled_in_native_function"]
    ports: tuple[Port, ...] = Field(max_length=64)
    config_checksum: Checksum
    policy_checksum: Checksum
    execution_authorized: Literal[False] = False

    @property
    def execution_manifest(self):
        return self.native_manifest

    @model_validator(mode="after")
    def integrity(self):
        check_seal(self)
        if (self.tool_ref, self.implementation_ref) != (
            self.binding.tool_ref,
            self.binding.implementation_ref,
        ):
            raise AdapterError("manifest_binding_mismatch")
        if self.catalog_ref.kind != "catalog":
            raise AdapterError("catalog_kind_required")
        if any(p.tool_version_id != self.binding.tool_version_id for p in self.ports):
            raise AdapterError("manifest_port_scope_mismatch")
        m = self.execution_manifest
        if (m is None) != (self.invocation is None):
            raise AdapterError("manifest_invocation_visibility_mismatch")
        if m and canonical(self.invocation) != canonical(m.invocation):
            raise AdapterError("manifest_invocation_binding_mismatch")
        if (m is None) != (self.native_manifest_state == "disabled_in_native_function"):
            raise AdapterError("manifest_visibility_mismatch")
        if m and (m.tool_code, m.version, m.implementation_key) != (
            self.binding.tool_code,
            self.binding.native_version,
            self.binding.implementation_key,
        ):
            raise AdapterError("manifest_native_identity_mismatch")
        return self


class ReviewedDeploymentManifest(DeploymentManifest):
    """06E overlay; native_manifest remains the exact original database document."""

    schema_version: Literal["medgraph-research-deployment/1.0.0"] = (
        "medgraph-research-deployment/1.0.0"
    )
    review_ref: Ref
    original_manifest_checksum: Checksum
    declaration_origin: Literal["deployment_owner_review_not_native"] = (
        "deployment_owner_review_not_native"
    )
    deployment_declaration: NativeManifest

    @property
    def execution_manifest(self):
        return self.deployment_declaration

    @model_validator(mode="after")
    def reviewed_identity(self):
        if self.review_ref.kind != "source_version" or self.native_manifest is None:
            raise AdapterError("research_review_source_required")
        original = self.native_manifest.model_dump(mode="json")
        deployed = self.deployment_declaration.model_dump(mode="json")
        for key in (
            "entrypoint_uri",
            "working_directory",
            "runtime_environment",
            "invocation",
            "resources",
        ):
            original.pop(key)
            deployed.pop(key)
        if original != deployed:
            raise AdapterError("research_declaration_changed_native_facts")
        return self


def parse_deployment(value):
    version = value.get("schema_version") if isinstance(value, dict) else value.schema_version
    model = (
        ReviewedDeploymentManifest
        if version == "medgraph-research-deployment/1.0.0"
        else DeploymentManifest
    )
    return parse(model, value)


def parse(model, value):
    """Suppress validation messages that could echo rejected native secrets."""
    from packages.grounding.models import validate

    try:
        return validate(model, value)
    except (ValueError, TypeError, RecursionError):
        raise AdapterError("invalid_or_unsafe_native_document") from None


def json_copy(value):
    return json.loads(canonical(value))
