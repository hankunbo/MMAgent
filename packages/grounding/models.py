"""Patient-free, immutable Grounding 1.0.0 values; no persistence or tool runtime."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.contracts import (
    CoordinateSpace,
    ExecutionPolicy,
    IntendedUse,
    Laterality,
    QCStatus,
    ToolCatalogStatus,
    ToolValidationStatus,
)

Code = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*(?::[A-Z0-9][A-Z0-9._-]*)+$", max_length=256)]
Token = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")]
Checksum = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Version = Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
Kind = Literal[
    "mapping",
    "policy",
    "catalog",
    "artifact",
    "capability",
    "tool",
    "implementation",
    "workflow",
    "evidence_requirement",
    "clinical_claim",
    "release",
    "template",
    "source_version",
    "applicability_condition",
    "estimate",
    "validation_rule",
]


class Value(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
    )


def wire(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value.model_dump(mode="json", warnings=False) if isinstance(value, BaseModel) else value


def canonical(value) -> bytes:
    return json.dumps(
        wire(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=wire,
    ).encode()


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def validate(model, value):
    """Reparse untrusted/copy/construct instances, including nested values."""

    def inspect(item):
        if isinstance(item, BaseModel):
            if set(vars(item)) - set(type(item).model_fields) or item.model_extra:
                raise ValueError("unknown fields on constructed/copied model")
            for child in vars(item).values():
                inspect(child)
        elif isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                inspect(child)

    inspect(value)
    return model.model_validate_json(canonical(value))


class Ref(Value):
    kind: Kind
    code: Token
    version: Version
    checksum: Checksum

    @model_validator(mode="after")
    def namespace(self) -> Self:
        if self.kind in {"tool", "implementation", "workflow", "artifact"}:
            # Native medgraph codes are case sensitive, e.g. retina:OpticDiscMask.
            pattern = r"[A-Za-z][A-Za-z0-9_.:-]*"
        else:
            pattern = r"[A-Z][A-Z0-9_]*(?::[A-Z0-9][A-Z0-9._-]*)+"
        if not re.fullmatch(pattern, self.code):
            raise ValueError("invalid reference namespace")
        if self.code.split(":")[0] in {"CASE", "RUN", "ASSET", "ENCOUNTER", "GRAPH", "MESSAGE"}:
            raise ValueError("patient/runtime reference forbidden in generic mapping")
        if self.kind == "mapping" and not re.fullmatch(
            r"BINDING:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}", self.code
        ):
            raise ValueError("mapping requires a stable opaque BINDING code")
        prefixes = {"mapping": "BINDING:", "policy": "POLICY:", "capability": "CAPABILITY:"}
        if self.kind in prefixes and not self.code.startswith(prefixes[self.kind]):
            raise ValueError("reference kind/namespace mismatch")
        return self


def require_kind(ref: Ref, *kinds: str) -> None:
    if ref.kind not in kinds:
        raise ValueError("reference kind mismatch")


def unique(values, key=lambda x: x) -> None:
    items = [key(x) for x in values]
    if len(items) != len(set(items)):
        raise ValueError("duplicate identity")


class State(StrEnum):
    DRAFT = "Draft"
    ACTIVE = "Active"
    DISABLED = "Disabled"
    WITHDRAWN = "Withdrawn"


class Term(Value):
    state: Literal["known", "unknown", "not_applicable", "unspecified"]
    code: Code | None

    @model_validator(mode="after")
    def known_code(self) -> Self:
        if (self.state == "known") != (self.code is not None):
            raise ValueError("only known terms have a code; null is not an unknown string")
        if self.code and not self.code.startswith("CONCEPT:"):
            raise ValueError("only controlled CONCEPT terms")
        return self


class ArtifactConstraint(Value):
    artifact_ref: Ref
    modality: Term
    laterality: Laterality
    coordinate_space: CoordinateSpace
    unit: Term
    qc_required: tuple[QCStatus, ...] = Field(min_length=1, max_length=4)
    scope_terms: tuple[Code, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def types(self) -> Self:
        require_kind(self.artifact_ref, "artifact")
        unique(self.qc_required)
        unique(self.scope_terms)
        if any(not term.startswith("CONCEPT:") for term in self.scope_terms):
            raise ValueError("scope requires controlled concept terms")
        return self


class PortRef(Value):
    tool_ref: Ref
    direction: Literal["input", "output"]
    port_name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]

    @model_validator(mode="after")
    def tool(self) -> Self:
        require_kind(self.tool_ref, "tool")
        return self


class Input(Value):
    port: PortRef
    constraint: ArtifactConstraint
    required: bool
    cardinality_min: int = Field(ge=0, le=64)
    cardinality_max: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def direction(self) -> Self:
        if self.port.direction != "input":
            raise ValueError("input direction required")
        if self.cardinality_min > self.cardinality_max or self.required != (
            self.cardinality_min > 0
        ):
            raise ValueError("required input/cardinality mismatch")
        return self


class Candidate(Value):
    code: Code
    capability_ref: Ref
    tool_ref: Ref
    implementation_ref: Ref
    workflow_ref: Ref | None
    output_port: PortRef
    output: ArtifactConstraint
    inputs: tuple[Input, ...] = Field(min_length=1, max_length=64)
    validation_required: ToolValidationStatus
    intended_use: IntendedUse
    execution_policy: ExecutionPolicy
    estimate_ref: Ref | None

    @model_validator(mode="after")
    def scope(self) -> Self:
        require_kind(self.capability_ref, "capability")
        require_kind(self.tool_ref, "tool")
        require_kind(self.implementation_ref, "implementation")
        if self.workflow_ref:
            require_kind(self.workflow_ref, "workflow")
        if self.estimate_ref:
            require_kind(self.estimate_ref, "estimate")
        if self.output_port.direction != "output" or self.output_port.tool_ref != self.tool_ref:
            raise ValueError("output port tool/version scope mismatch")
        if any(i.port.tool_ref != self.tool_ref for i in self.inputs):
            raise ValueError("input port tool/version scope mismatch")
        unique(self.inputs, lambda i: i.port.port_name)
        return self


class Policy(Value):
    kind: Literal["policy"] = "policy"
    code: Code
    version: Version
    checksum: Checksum
    baseline_only: Literal[True]
    synthetic_only: bool
    admissible_states: tuple[Literal["Active"], ...] = Field(min_length=1, max_length=1)
    type_compatibility: Literal["exact"]
    accepted_validation: tuple[ToolValidationStatus, ...] = Field(min_length=1, max_length=4)
    intended_use: Literal[IntendedUse.RESEARCH_ONLY]
    execution_policy: Literal[ExecutionPolicy.RESEARCH]
    max_bytes: int = Field(ge=1024, le=65536)
    max_candidates: int = Field(ge=1, le=64)
    max_inputs: int = Field(ge=1, le=64)
    max_artifacts: int = Field(ge=1, le=32)
    max_mappings: int = Field(ge=1, le=128)
    allowed_terms: tuple[Code, ...] = Field(min_length=1, max_length=128)

    def reference(self) -> Ref:
        return Ref(kind="policy", code=self.code, version=self.version, checksum=self.checksum)

    @model_validator(mode="after")
    def integrity(self) -> Self:
        self.reference()
        unique(self.allowed_terms)
        unique(self.accepted_validation)
        if any(not x.startswith("CONCEPT:") for x in self.allowed_terms):
            raise ValueError("policy terms must be controlled concepts")
        check_seal(self)
        return self


class Mapping(Value):
    schema_version: Literal["grounding/1.0.0"] = "grounding/1.0.0"
    kind: Literal["mapping"] = "mapping"
    code: Code
    version: Version
    checksum: Checksum
    synthetic: bool
    requirement_ref: Ref
    claim_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    release_ref: Ref
    template_ref: Ref
    source_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    applicability_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    catalog_ref: Ref
    policy_ref: Ref
    accepted_artifacts: tuple[ArtifactConstraint, ...] = Field(min_length=1, max_length=32)
    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=64)

    def reference(self) -> Ref:
        return Ref(kind="mapping", code=self.code, version=self.version, checksum=self.checksum)

    @model_validator(mode="after")
    def shape(self) -> Self:
        self.reference()
        for ref, kind in [
            (self.requirement_ref, "evidence_requirement"),
            (self.release_ref, "release"),
            (self.template_ref, "template"),
            (self.catalog_ref, "catalog"),
            (self.policy_ref, "policy"),
        ]:
            require_kind(ref, kind)
        for refs, kind in [
            (self.claim_refs, "clinical_claim"),
            (self.source_refs, "source_version"),
            (self.applicability_refs, "applicability_condition"),
        ]:
            unique(refs)
            for ref in refs:
                require_kind(ref, kind)
        unique(self.accepted_artifacts, lambda a: a.artifact_ref)
        unique(self.candidates, lambda c: c.code)
        for candidate in self.candidates:
            if candidate.output not in self.accepted_artifacts:
                raise ValueError("candidate output outside accepted artifact alternatives")
        check_seal(self)
        return self


GroundingRecord = Mapping


def check_seal(value) -> None:
    data = wire(value)
    checksum = data.pop("checksum")
    if digest(data) != checksum:
        raise ValueError("content checksum mismatch")


def seal(model, **fields):
    defaults = {k: f.default for k, f in model.model_fields.items() if not f.is_required()}
    data = {**defaults, **fields}
    data.pop("checksum", None)
    data["checksum"] = digest(data)
    return model.model_validate_json(canonical(data))


class CatalogTool(Value):
    """Derived snapshot projection, never a second editable Tool master."""

    tool_ref: Ref
    implementation_ref: Ref
    capabilities: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    inputs: tuple[Input, ...] = Field(min_length=1, max_length=64)
    outputs: tuple[ArtifactConstraint, ...] = Field(min_length=1, max_length=32)
    output_ports: tuple[PortRef, ...] = Field(min_length=1, max_length=32)
    workflows: tuple[Ref, ...] = Field(max_length=32)
    enabled: bool
    catalog_status: ToolCatalogStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse

    @model_validator(mode="after")
    def shape(self) -> Self:
        require_kind(self.tool_ref, "tool")
        require_kind(self.implementation_ref, "implementation")
        for r in self.capabilities:
            require_kind(r, "capability")
        for r in self.workflows:
            require_kind(r, "workflow")
        if len(self.outputs) != len(self.output_ports):
            raise ValueError("output/port count mismatch")
        for p in self.output_ports:
            if p.tool_ref != self.tool_ref or p.direction != "output":
                raise ValueError("catalog output scope mismatch")
        if any(i.port.tool_ref != self.tool_ref for i in self.inputs):
            raise ValueError("catalog input scope mismatch")
        unique(self.inputs, lambda i: i.port.port_name)
        unique(self.output_ports)
        unique(self.capabilities)
        unique(self.workflows)
        return self


class CatalogSnapshot(Value):
    kind: Literal["catalog"] = "catalog"
    code: Code
    version: Version
    checksum: Checksum
    synthetic: bool
    manifest_checksum: Checksum
    artifacts: tuple[Ref, ...] = Field(min_length=1, max_length=256)
    tools: tuple[CatalogTool, ...] = Field(min_length=1, max_length=128)
    estimates: tuple[Ref, ...] = Field(max_length=128)

    def reference(self) -> Ref:
        return Ref(kind="catalog", code=self.code, version=self.version, checksum=self.checksum)

    @model_validator(mode="after")
    def shape(self) -> Self:
        unique(self.artifacts)
        unique(self.tools, lambda t: t.implementation_ref)
        unique(self.estimates)
        for r in self.artifacts:
            require_kind(r, "artifact")
        for r in self.estimates:
            require_kind(r, "estimate")
        for tool in self.tools:
            if any(a.artifact_ref not in self.artifacts for a in tool.outputs):
                raise ValueError("output artifact outside catalog")
            if any(i.constraint.artifact_ref not in self.artifacts for i in tool.inputs):
                raise ValueError("input artifact outside catalog")
        check_seal(self)
        return self


class MappingSnapshot(Value):
    record: Mapping
    state: State
    revision: int = Field(ge=0)


class Review(Value):
    sequence: int = Field(ge=1)
    mapping_ref: Ref
    policy_ref: Ref
    state: State
    reason: Literal["machine_pass", "policy_rejected", "disabled", "withdrawn"]
    request_key: Token


class History(Value):
    snapshot: MappingSnapshot
    reviews: tuple[Review, ...]
    currently_admissible: Literal[False] = False
