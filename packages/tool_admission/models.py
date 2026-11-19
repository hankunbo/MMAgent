"""06B transient admission values; never execution authority or clinical evidence."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_serializer, model_validator

from packages.contracts import Artifact, CapabilityGap
from packages.contracts.models import UnitCode
from packages.grounding.models import (
    ArtifactConstraint,
    Candidate,
    Checksum,
    Code,
    Policy,
    PortRef,
    Ref,
    Term,
    Token,
    Value,
    Version,
    canonical,
    check_seal,
    require_kind,
    unique,
)
from packages.grounding.query_models import ArtifactMetadata, CaseScope, Finding, ProductDemand

Text = Annotated[str, Field(min_length=1, max_length=512)]
Purpose = Literal["decisive_research", "debug", "shadow"]
Category = Literal[
    "patient_data_missing",
    "mapping_missing",
    "unresolved_producer",
    "implementation_missing",
    "validation_insufficient",
    "environment_unavailable",
    "policy_rejection",
    "input_incompatible",
    "output_incompatible",
    "technical_error",
    "truncation",
    "input_unsatisfied",
]


class AdmissionError(ValueError):
    """Safe fixed code. Technical failures propagate, never turn into NO_TOOL."""


class AdmissionPolicy(Value):
    version: Version = "1.0.0"
    baseline_only: Literal[True] = True
    timeout_seconds: int = Field(default=30, ge=1, le=60)
    max_nodes: int = Field(default=512, ge=1, le=512)
    max_checks: int = Field(default=4096, ge=1, le=8192)
    max_input_bytes: int = Field(default=4194304, ge=1, le=4194304)
    max_total_bytes: int = Field(default=67108864, ge=1, le=2147483648)
    header_bytes: int = Field(default=65536, ge=64, le=65536)
    max_result_bytes: int = Field(default=8388608, ge=1024, le=16777216)


class ArtifactTypeBinding(Value):
    version: Version
    checksum: Checksum
    artifact_ref: Ref
    contract_type_code: Code
    unit_term: Term
    contract_unit: UnitCode | None
    source_ref: Ref

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.artifact_ref, "artifact")
        require_kind(self.source_ref, "source_version")
        check_seal(self)
        return self


class InputRecord(Value):
    """Returned by an ingestion/Runtime owner, not supplied as caller truth."""

    artifact: Artifact
    type_binding: ArtifactTypeBinding
    metadata: ArtifactMetadata
    byte_size: int = Field(ge=1, le=10737418240)
    content_kind: Literal["image", "array", "clinical_note", "unsupported"]
    source_ref: Ref

    @model_validator(mode="after")
    def coherent(self):
        a, m = self.artifact, self.metadata
        require_kind(self.source_ref, "source_version")
        if (
            a.asset_code,
            a.checksum,
            a.artifact_type_code,
            a.qc_status,
            a.laterality,
            a.coordinate_space,
            a.scope.case_code,
            a.scope.run_id,
        ) != (
            m.asset_code,
            m.checksum,
            self.type_binding.contract_type_code,
            m.qc_status,
            m.laterality,
            m.coordinate_space,
            m.case_code,
            m.run_id,
        ):
            raise ValueError("artifact_metadata_binding_mismatch")
        if m.artifact_ref != self.type_binding.artifact_ref:
            raise ValueError("artifact_type_binding_mismatch")
        if a.unit != self.type_binding.contract_unit or m.unit != self.type_binding.unit_term:
            raise ValueError("artifact_unit_binding_mismatch")
        return self


class PortRule(Value):
    """Technical target codes, not a synonym map or untyped task scope terms."""

    port: PortRef
    media_types: tuple[Literal["image/png", "image/jpeg", "application/x-npy"], ...] = Field(
        min_length=1, max_length=3
    )
    dimensions_min: tuple[int, ...] = Field(min_length=1, max_length=5)
    dimensions_max: tuple[int, ...] = Field(min_length=1, max_length=5)
    anatomical_targets: tuple[Code, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def shape(self):
        unique(self.media_types)
        unique(self.anatomical_targets)
        if len(self.dimensions_min) != len(self.dimensions_max) or any(
            low < 1 or high < low
            for low, high in zip(self.dimensions_min, self.dimensions_max, strict=True)
        ):
            raise ValueError("invalid_dimension_range")
        return self


class JSONOutputRule(PortRule):
    schema_version: Literal["json-output-rule/1.0.0"] = "json-output-rule/1.0.0"
    media_types: tuple[Literal["application/json"], ...] = ("application/json",)

    @model_validator(mode="after")
    def output_only(self):
        if self.port.direction != "output":
            raise ValueError("json_rule_is_output_only")
        return self


class RuntimeSpec(Value):
    interpreter_uri: Text
    interpreter_checksum: Checksum
    dependencies: dict[Token, Annotated[str, Field(pattern=r"^[0-9][A-Za-z0-9.!+_-]{0,127}$")]] = (
        Field(max_length=128)
    )
    gpu: Literal["required", "not_required"]


class OutputTargetBinding(Value):
    """Deployment owner's explicit anatomy subset of one exact output constraint.

    Other scope terms remain applicability conditions. No inferred clinical aliases.
    The enclosing profile binds this declaration to manifest/catalog/implementation.
    """

    version: Version
    checksum: Checksum
    source_ref: Ref
    port: PortRef
    constraint: ArtifactConstraint
    anatomical_targets: tuple[Code, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.source_ref, "source_version")
        unique(self.anatomical_targets)
        if self.port.direction != "output":
            raise ValueError("output_target_binding_direction")
        if not set(self.anatomical_targets) <= set(self.constraint.scope_terms):
            raise ValueError("output_target_binding_outside_constraint")
        check_seal(self)
        return self


class DeploymentProfile(Value):
    """Owned versioned technical declarations bound to the 06A manifest."""

    version: Version
    checksum: Checksum
    manifest_checksum: Checksum
    config_checksum: Checksum
    policy_checksum: Checksum
    input_rules: tuple[PortRule, ...] = Field(min_length=1, max_length=64)
    output_rules: tuple[PortRule | JSONOutputRule, ...] = Field(min_length=1, max_length=64)
    output_target_bindings: tuple[OutputTargetBinding, ...] = Field(default=(), max_length=64)
    preprocessing_checksum: Checksum
    postprocessing_checksum: Checksum
    runtime: RuntimeSpec

    @model_validator(mode="after")
    def coherent(self):
        unique(self.input_rules, lambda p: p.port)
        unique(self.output_rules, lambda p: p.port)
        unique(self.output_target_bindings, lambda b: (b.port, b.constraint))
        check_seal(self)
        return self

    def check_output_targets(self, candidate: Candidate, satisfied_terms: tuple[str, ...]):
        """Contextual check; caller must first verify the controlled profile binding."""
        rule = next((p for p in self.output_rules if p.port == candidate.output_port), None)
        if rule is None:
            raise AdmissionError("profile_output_port_mismatch")
        targets = set(rule.anatomical_targets)
        if not targets <= set(satisfied_terms) or not targets <= set(candidate.output.scope_terms):
            raise AdmissionError("output_anatomical_target_mismatch")
        binding = next(
            (
                b
                for b in self.output_target_bindings
                if b.port == candidate.output_port and b.constraint == candidate.output
            ),
            None,
        )
        if binding is None:
            raise AdmissionError("output_anatomical_target_unknown")
        if targets != set(binding.anatomical_targets):
            raise AdmissionError("output_anatomical_target_mismatch")
        return rule


class EnvironmentProbe(Value):
    version: Version
    checksum: Checksum
    manifest_checksum: Checksum
    profile_checksum: Checksum
    runtime_checksum: Checksum
    source_ref: Ref
    observed_at: datetime
    expires_at: datetime
    interpreter: Literal["available", "unavailable", "unknown"]
    dependencies: Literal["available", "unavailable", "unknown"]
    gpu: Literal["available", "unavailable", "unknown", "not_required"]
    resources: Literal["available", "unavailable", "unknown"]
    preconditions: Literal["satisfied", "rejected", "unknown"]

    @field_serializer("observed_at", "expires_at")
    def serialize_time(self, value):
        return value.isoformat()

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.source_ref, "source_version")
        if any(t.utcoffset() is None for t in (self.observed_at, self.expires_at)) or (
            self.expires_at <= self.observed_at
        ):
            raise ValueError("invalid_probe_time")
        check_seal(self)
        return self


class InvocationInput(Value):
    implementation_ref: Ref
    manifest_checksum: Checksum
    profile_checksum: Checksum
    parameters: dict[Token, JsonValue] = Field(max_length=64)

    @model_validator(mode="after")
    def bounded(self):
        require_kind(self.implementation_ref, "implementation")
        if len(canonical(self.parameters)) > 16384:
            raise ValueError("parameter_byte_limit")
        return self


class AdmissionSelection(Value):
    candidate_refs: tuple[Checksum, ...] = Field(max_length=128)
    invocations: tuple[InvocationInput, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.candidate_refs)
        unique(self.invocations, lambda i: i.implementation_ref)
        return self


class ResearchLock(Value):
    run_id: Annotated[str, Field(pattern=r"^RUN:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")]
    catalog_ref: Ref

    @model_validator(mode="after")
    def kinds(self):
        require_kind(self.catalog_ref, "catalog")
        return self


class ResearchInput(Value):
    port: PortRef
    asset_codes: tuple[Text, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.asset_codes)
        return self


class ResearchRequest(Value):
    """Separate exact registered request: no medical claim, mapping or Draft payload."""

    schema_version: Literal["tool-research-admission/1.0.0"] = "tool-research-admission/1.0.0"
    code: Code
    version: Version
    checksum: Checksum
    purpose: Literal["debug", "shadow"]
    compatibility_policy: Policy
    lock: ResearchLock
    scope: CaseScope
    window_start: datetime
    window_end: datetime
    snapshot_at: datetime
    satisfied_terms: tuple[Code, ...] = Field(max_length=128)
    artifacts: tuple[ArtifactMetadata, ...] = Field(max_length=256)
    candidate: Candidate
    inputs: tuple[ResearchInput, ...] = Field(min_length=1, max_length=64)
    invocation: InvocationInput
    decisive_evidence: Literal[False] = False

    @field_serializer("window_start", "window_end", "snapshot_at")
    def serialize_time(self, value):
        return value.isoformat()

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        unique(self.inputs, lambda i: i.port)
        unique(self.artifacts, lambda a: a.asset_code)
        unique(self.satisfied_terms)
        if any(
            t.utcoffset() is None for t in (self.window_start, self.window_end, self.snapshot_at)
        ) or not (self.window_start <= self.window_end <= self.snapshot_at):
            raise ValueError("invalid_research_time")
        if set(i.port for i in self.inputs) != set(i.port for i in self.candidate.inputs):
            raise ValueError("complete_research_inputs_required")
        if self.invocation.implementation_ref != self.candidate.implementation_ref:
            raise ValueError("research_implementation_mismatch")
        for a in self.artifacts:
            if (a.case_code, a.patient_hash, a.encounter_code) != (
                self.scope.case_code,
                self.scope.patient_hash,
                self.scope.encounter_code,
            ) or a.run_id not in (None, self.lock.run_id):
                raise ValueError("research_scope_mismatch")
        return self


class VerifiedInput(Value):
    asset_code: Text
    uri: Text
    checksum: Checksum
    byte_size: int = Field(gt=0)
    record_checksum: Checksum
    media_type: Text
    dimensions: tuple[int, ...] = Field(min_length=1, max_length=5)


class AdmittedPort(Value):
    port: PortRef
    required: bool
    cardinality_min: int = Field(ge=0, le=64)
    cardinality_max: int = Field(ge=1, le=64)
    artifacts: tuple[VerifiedInput, ...] = Field(max_length=256)
    producer_refs: tuple[Checksum, ...] = Field(max_length=512)
    optional_missing: bool

    @model_validator(mode="after")
    def coherent(self):
        unique(self.artifacts, lambda a: a.asset_code)
        unique(self.producer_refs)
        count = len(self.artifacts) + bool(self.producer_refs)
        if count < self.cardinality_min or self.optional_missing != (
            not self.required and not count
        ):
            raise ValueError("unsatisfied_admitted_port")
        return self


DYNAMIC_CHECKS = (
    "06D.current_safety_and_lock",
    "06D.input_and_resource_race_protection",
    "06D.environment_and_resource_reservation",
    "06D.intermediate_bytes_format_scope_qc",
    "06D.process_file_network_limits",
    "06D.output_qc_then_evidence_validation",
)


class RetainedCandidate(Value):
    identity: Checksum
    candidate: Candidate
    manifest_checksum: Checksum
    profile_checksum: Checksum
    probe_checksum: Checksum
    resources_checksum: Checksum
    invocation_checksum: Checksum
    inputs: tuple[AdmittedPort, ...] = Field(min_length=1, max_length=64)
    output_rule: PortRule | JSONOutputRule
    state: Literal["preparation_only"] = "preparation_only"
    pending_dynamic_checks: tuple[Text, ...] = DYNAMIC_CHECKS

    @model_validator(mode="after")
    def coherent(self):
        if self.output_rule.port != self.candidate.output_port:
            raise ValueError("retained_output_port_mismatch")
        expected = {i.port: i for i in self.candidate.inputs}
        unique(self.inputs, lambda i: i.port)
        if set(expected) != {i.port for i in self.inputs}:
            raise ValueError("complete_retained_ports_required")
        for i in self.inputs:
            e = expected[i.port]
            if (i.required, i.cardinality_min, i.cardinality_max) != (
                e.required,
                e.cardinality_min,
                e.cardinality_max,
            ):
                raise ValueError("retained_port_cardinality_mismatch")
        if self.pending_dynamic_checks != DYNAMIC_CHECKS:
            raise ValueError("dynamic_checks_cannot_be_removed")
        return self


class Rejection(Value):
    candidate_ref: Checksum
    tool_ref: Ref
    implementation_ref: Ref
    capability_ref: Ref
    port: PortRef | None
    asset_code: Text | None
    constraint: ArtifactConstraint
    reason: Token
    category: Category


class DeploymentGap(Value):
    origin: Literal["selected_candidate", "official_rejection"] = "selected_candidate"
    candidate_ref: Checksum
    tool_ref: Ref
    implementation_ref: Ref
    manifest_checksum: Checksum
    gaps: tuple[Text, ...] = Field(min_length=1, max_length=128)


class AdmissionResult(Value):
    schema_version: Literal["tool-admission/1.0.0"] = "tool-admission/1.0.0"
    checksum: Checksum
    purpose: Purpose
    request_checksum: Checksum
    selection_checksum: Checksum
    query_checksum: Checksum | None
    grounding_status: Literal["existing_artifact", "candidates", "gaps", "truncated"] | None
    catalog_ref: Ref
    scope: CaseScope
    run_id: Text
    policy_checksum: Checksum
    safety_checksum: Checksum
    status: Literal["preparation_only", "rejected", "truncated"]
    retained_roots: tuple[Checksum, ...] = Field(max_length=128)
    retained: tuple[RetainedCandidate, ...] = Field(max_length=512)
    rejected: tuple[Rejection, ...] = Field(max_length=8192)
    deployment_gaps: tuple[DeploymentGap, ...] = Field(max_length=512)
    upstream_findings: tuple[Finding, ...] = Field(max_length=8192)
    product_demands: tuple[ProductDemand, ...] = Field(max_length=8192)
    capability_gaps: tuple[CapabilityGap, ...] = Field(max_length=512)
    truncated: bool
    execution_authorized: Literal[False] = False
    decisive_evidence: Literal[False] = False
    clinical_effect: Literal["NONE"] = "NONE"

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        require_kind(self.catalog_ref, "catalog")
        unique(self.retained, lambda c: c.identity)
        unique(self.retained_roots)
        nodes = {c.identity: c for c in self.retained}
        if not set(self.retained_roots) <= nodes.keys():
            raise ValueError("unknown_retained_root")
        for c in self.retained:
            for i in c.inputs:
                if not set(i.producer_refs) <= nodes.keys():
                    raise ValueError("unknown_retained_producer")
                constraint = next(p.constraint for p in c.candidate.inputs if p.port == i.port)
                if any(nodes[p].candidate.output != constraint for p in i.producer_refs):
                    raise ValueError("retained_producer_type_mismatch")
        expected = (
            "truncated"
            if self.truncated
            else ("preparation_only" if self.retained_roots else "rejected")
        )
        if self.status != expected or (self.truncated and self.retained):
            raise ValueError("admission_status_mismatch")
        if (self.query_checksum is None) != (self.grounding_status is None):
            raise ValueError("grounding_status_binding_mismatch")
        if (self.purpose == "decisive_research") != (self.query_checksum is not None):
            raise ValueError("formal_research_isolation")
        if self.purpose != "decisive_research" and (
            self.upstream_findings or self.product_demands or self.capability_gaps
        ):
            raise ValueError("research_has_no_medical_attribution")
        return self
