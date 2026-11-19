"""05B transient Runtime metadata and strict deterministic results (not medical evidence)."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.contracts import CapabilityGap, CoordinateSpace, Laterality, QCStatus
from packages.retrieval.knowledge.models import RetrievalContext

from .locks import GroundingLock
from .models import (
    ArtifactConstraint,
    Candidate,
    Checksum,
    Code,
    PortRef,
    Ref,
    Term,
    Token,
    Value,
    Version,
    check_seal,
    digest,
    require_kind,
    unique,
)

Metric = Literal["information_gain", "directness", "cost", "latency", "failure_risk"]
Reason = Literal[
    "artifact_outside_catalog",
    "unknown_scope_term",
    "capability_outside_tool",
    "tool_outside_catalog",
    "workflow_outside_tool",
    "estimate_outside_catalog",
    "output_port_scope_mismatch",
    "output_type_mismatch",
    "complete_input_set_required",
    "tool_not_active",
    "tool_safety_revoked",
    "tool_validation_rejected",
    "intended_use_rejected",
    "artifact_type_mismatch",
    "modality_mismatch",
    "laterality_mismatch",
    "coordinate_mismatch",
    "unit_mismatch",
    "qc_rejected",
    "scope_mismatch",
    "time_unknown",
    "time_mismatch",
    "required_condition_unknown",
    "precondition_rejected",
    "environment_unknown",
    "environment_unavailable",
    "patient_data_missing",
    "capability_unregistered",
    "required_input_missing",
    "unmapped_in_locked_view",
    "unresolved_producer",
    "cycle_detected",
    "depth_exhausted",
    "budget_exhausted",
]
GapKind = Literal[
    "patient_data_missing",
    "artifact_undefined",
    "capability_unregistered",
    "implementation_missing",
    "validation_insufficient",
    "input_incompatible",
    "environment_unavailable",
    "safety_restriction",
    "search_incomplete",
    "mapping_missing",
    "unresolved_producer",
    "input_unsatisfied",
]


def category(reason: str) -> str:
    groups = {
        "patient_data_missing": {"patient_data_missing"},
        "input_unsatisfied": {"required_input_missing"},
        "mapping_missing": {"unmapped_in_locked_view"},
        "unresolved_producer": {"unresolved_producer"},
        "artifact_undefined": {"artifact_outside_catalog"},
        "capability_unregistered": {"capability_outside_tool", "capability_unregistered"},
        "implementation_missing": {"tool_outside_catalog", "workflow_outside_tool"},
        "validation_insufficient": {"tool_validation_rejected"},
        "environment_unavailable": {"environment_unknown", "environment_unavailable"},
        "safety_restriction": {
            "tool_not_active",
            "tool_safety_revoked",
            "intended_use_rejected",
            "precondition_rejected",
            "required_condition_unknown",
            "unknown_scope_term",
        },
        "search_incomplete": {"cycle_detected", "depth_exhausted", "budget_exhausted"},
    }
    return next((key for key, values in groups.items() if reason in values), "input_incompatible")


class CaseScope(Value):
    case_code: str = Field(pattern=r"^CASE:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
    patient_hash: Checksum
    encounter_code: str = Field(pattern=r"^ENCOUNTER:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
    laterality: Laterality


class ArtifactMetadata(Value):
    """Explicit ingestion projection. No URI, report, measurements or image bytes."""

    asset_code: str = Field(pattern=r"^ASSET:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
    checksum: Checksum
    case_code: str
    patient_hash: Checksum
    encounter_code: str
    run_id: str | None
    artifact_ref: Ref
    modality: Term
    laterality: Laterality
    coordinate_space: CoordinateSpace
    unit: Term
    scope_terms: tuple[Code, ...] = Field(max_length=32)
    qc_status: QCStatus
    acquired_at: datetime | None

    @model_validator(mode="after")
    def shape(self):
        require_kind(self.artifact_ref, "artifact")
        unique(self.scope_terms)
        if self.acquired_at is not None and self.acquired_at.utcoffset() is None:
            raise ValueError("timezone required")
        return self


class EnvironmentFact(Value):
    implementation_ref: Ref
    availability: Literal["available", "unavailable", "unknown"]
    preconditions: Literal["satisfied", "rejected", "unknown"]
    source_ref: Ref

    @model_validator(mode="after")
    def kinds(self):
        require_kind(self.implementation_ref, "implementation")
        require_kind(self.source_ref, "source_version")
        return self


class Component(Value):
    name: Metric
    state: Literal["known", "unknown", "disabled"]
    value: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    source_ref: Ref | None

    @model_validator(mode="after")
    def coherent(self):
        if (self.state == "known") != (self.value is not None and self.source_ref is not None):
            raise ValueError("known estimate requires value and source")
        if self.state != "known" and (self.value is not None or self.source_ref is not None):
            raise ValueError("unknown/disabled estimates cannot carry values")
        if (
            self.name in {"directness", "failure_risk"}
            and self.value is not None
            and self.value > 1
        ):
            raise ValueError("normalized component outside [0,1]")
        if self.source_ref:
            require_kind(self.source_ref, "source_version")
        return self


METRICS = ("information_gain", "directness", "cost", "latency", "failure_risk")


class RankingEstimate(Value):
    catalog_ref: Ref
    estimate_ref: Ref
    components: tuple[Component, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def shape(self):
        require_kind(self.catalog_ref, "catalog")
        require_kind(self.estimate_ref, "estimate")
        if tuple(c.name for c in self.components) != METRICS:
            raise ValueError("complete canonical component set required")
        if self.estimate_ref.checksum != digest(self.components):
            raise ValueError("estimate content mismatch")
        return self


class QueryConfig(Value):
    code: Literal["POLICY:GROUNDING:QUERY_BASELINE"] = "POLICY:GROUNDING:QUERY_BASELINE"
    version: Version
    checksum: Checksum
    baseline_only: Literal[True] = True
    max_depth: int = Field(ge=1, le=12)
    max_evaluations: int = Field(ge=1, le=512)
    max_operations: int = Field(default=10000, ge=1, le=100000)
    max_findings: int = Field(default=4096, ge=2, le=8192)
    top_k: int = Field(ge=1, le=128)
    information_gain_enabled: bool
    order: tuple[Metric, ...] = Field(min_length=1, max_length=5)
    missing_order: Literal["last"] = "last"
    tie_break: Literal["mapping_ref,candidate_code,dependency_checksum"] = (
        "mapping_ref,candidate_code,dependency_checksum"
    )
    score_scope: Literal["individual_tool_estimates_not_chain_totals"] = (
        "individual_tool_estimates_not_chain_totals"
    )

    @model_validator(mode="after")
    def shape(self):
        unique(self.order)
        check_seal(self)
        return self


class QueryRequest(Value):
    schema_version: Literal["grounding-query/1.0.0"] = "grounding-query/1.0.0"
    lock: GroundingLock
    requirement_ref: Ref
    target_node_code: Code
    scope: CaseScope
    artifacts: tuple[ArtifactMetadata, ...] = Field(max_length=256)
    availability_source_ref: Ref
    snapshot_at: datetime
    window_start: datetime
    window_end: datetime
    patient_data_types: tuple[Ref, ...] = Field(max_length=256)
    satisfied_terms: tuple[Code, ...] = Field(max_length=128)
    environment: tuple[EnvironmentFact, ...] = Field(max_length=128)
    context: RetrievalContext
    config: QueryConfig

    @model_validator(mode="after")
    def shape(self):
        require_kind(self.requirement_ref, "evidence_requirement")
        require_kind(self.availability_source_ref, "source_version")
        if self.requirement_ref not in self.lock.knowledge_refs:
            raise ValueError("requirement outside locked Knowledge closure")
        if any(
            t.utcoffset() is None for t in (self.snapshot_at, self.window_start, self.window_end)
        ):
            raise ValueError("timezone required")
        if not self.window_start <= self.window_end <= self.snapshot_at:
            raise ValueError("invalid availability/time window")
        unique(self.artifacts, lambda a: a.asset_code)
        unique(self.environment, lambda e: e.implementation_ref)
        unique(self.patient_data_types)
        for ref in self.patient_data_types:
            require_kind(ref, "artifact")
        unique(self.satisfied_terms)
        for a in self.artifacts:
            if (a.case_code, a.patient_hash, a.encounter_code) != (
                self.scope.case_code,
                self.scope.patient_hash,
                self.scope.encounter_code,
            ) or a.run_id not in (None, self.lock.run_id):
                raise ValueError("cross patient/encounter/run metadata forbidden")
        return self


Path = Annotated[tuple[str, ...], Field(min_length=1, max_length=26)]


class PathSource(Value):
    """Exact locked mapping candidate and its occurrence in the dependency search."""

    mapping_ref: Ref
    candidate: Candidate
    path: Path
    mapping_path: tuple[Ref, ...] = Field(min_length=1, max_length=13)

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.mapping_ref, "mapping")
        for ref in self.mapping_path:
            require_kind(ref, "mapping")
        if (
            self.path[-1] != f"{self.mapping_ref.code}/{self.candidate.code}"
            or len(self.path) % 2 != 1
            or len(self.mapping_path) != (len(self.path) + 1) // 2
            or self.mapping_path[-1] != self.mapping_ref
        ):
            raise ValueError("source path/candidate mismatch")
        return self


class CatalogProducer(Value):
    """Catalog output projection, not an admitted mapping or execution authorization.

    All declared capabilities are retained; multiple capabilities do not establish
    an unambiguous output-to-capability relationship and are never guessed.
    """

    tool_ref: Ref
    implementation_ref: Ref
    capability_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    output_port: PortRef
    output: ArtifactConstraint

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.tool_ref, "tool")
        require_kind(self.implementation_ref, "implementation")
        unique(self.capability_refs)
        for ref in self.capability_refs:
            require_kind(ref, "capability")
        if self.output_port.tool_ref != self.tool_ref or self.output_port.direction != "output":
            raise ValueError("catalog producer output scope mismatch")
        return self


class InputImpact(Value):
    source: PathSource
    input_port: PortRef

    @model_validator(mode="after")
    def coherent(self):
        if self.input_port not in {i.port for i in self.source.candidate.inputs if i.required}:
            raise ValueError("impact outside required consumer input")
        return self


class Finding(Value):
    reason: Reason
    category: GapKind
    path: Path
    scope: CaseScope
    constraint: ArtifactConstraint | None
    catalog_ref: Ref
    source: PathSource | None
    input_port: PortRef | None
    catalog_producers: tuple[CatalogProducer, ...] = Field(default=(), max_length=4096)
    asset_code: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.catalog_ref, "catalog")
        if self.category != category(self.reason):
            raise ValueError("reason/category mismatch")
        if self.source is None:
            if self.reason != "budget_exhausted" and self.path != ("existing_artifact",):
                raise ValueError("candidate finding requires exact source")
            if self.input_port is not None or self.catalog_producers:
                raise ValueError("input finding requires exact source")
        elif self.input_port is None:
            if self.path != self.source.path or self.constraint != self.source.candidate.output:
                raise ValueError("own failure must bind source output")
        else:
            inputs = {i.port: i.constraint for i in self.source.candidate.inputs}
            if (
                self.input_port not in inputs
                or self.constraint != inputs[self.input_port]
                or self.path != (*self.source.path, self.input_port.port_name)
            ):
                raise ValueError("input finding port/product/source mismatch")
        if (
            self.reason
            in {
                "patient_data_missing",
                "unmapped_in_locked_view",
                "unresolved_producer",
                "required_input_missing",
            }
            and self.input_port is None
        ):
            raise ValueError("missing product requires input port")
        unique(self.catalog_producers, digest)
        if self.catalog_producers and (
            self.reason != "unmapped_in_locked_view"
            or any(
                p.output.artifact_ref != self.constraint.artifact_ref
                for p in self.catalog_producers
            )
        ):
            raise ValueError("catalog producer discovery/product mismatch")
        if self.reason == "unmapped_in_locked_view" and not self.catalog_producers:
            raise ValueError("mapping gap requires catalog producer evidence")
        return self


class ProductDemand(Value):
    """Unmet product, separately from any resolved capability extension target."""

    binding_checksum: Checksum
    requirement_ref: Ref
    catalog_ref: Ref
    policy_ref: Ref
    constraint: ArtifactConstraint
    resolution: Literal[
        "patient_data_missing",
        "unmapped_in_locked_view",
        "unresolved_producer",
        "mapped_producer_blocked",
        "search_incomplete",
        "input_incompatible",
    ]
    catalog_producers: tuple[CatalogProducer, ...] = Field(max_length=4096)
    producer_paths: tuple[PathSource, ...] = Field(max_length=512)
    affected_inputs: tuple[InputImpact, ...] = Field(min_length=1, max_length=8192)
    finding_checksums: tuple[Checksum, ...] = Field(min_length=1, max_length=8192)
    capability_gap_status: Literal["not_constructible_from_product_requirement"] = (
        "not_constructible_from_product_requirement"
    )

    @model_validator(mode="after")
    def coherent(self):
        unique(self.affected_inputs, digest)
        unique(self.finding_checksums)
        unique(self.catalog_producers, digest)
        unique(self.producer_paths, digest)
        if any(p.candidate.output != self.constraint for p in self.producer_paths):
            raise ValueError("mapped producer/product mismatch")
        for impact in self.affected_inputs:
            if (
                next(
                    i.constraint
                    for i in impact.source.candidate.inputs
                    if i.port == impact.input_port
                )
                != self.constraint
            ):
                raise ValueError("different product demands cannot merge")
        return self


class Dependency(Value):
    port_name: Token
    required: bool
    cardinality_min: int = Field(ge=0, le=64)
    cardinality_max: int = Field(ge=1, le=64)
    constraint: ArtifactConstraint
    artifact_options: tuple[str, ...] = Field(max_length=256)
    producer_options: tuple["FeasibleCandidate", ...] = Field(max_length=512)
    optional_missing: bool

    @model_validator(mode="after")
    def coherent(self):
        unique(self.artifact_options)
        unique(self.producer_options, lambda c: c.identity)
        count = len(self.artifact_options) + int(bool(self.producer_options))
        if (
            self.required != (self.cardinality_min > 0)
            or self.cardinality_max < self.cardinality_min
        ):
            raise ValueError("dependency cardinality mismatch")
        if count < self.cardinality_min or self.optional_missing != (
            not self.required and count == 0
        ):
            raise ValueError("unsatisfied dependency on feasible candidate")
        if any(c.candidate.output != self.constraint for c in self.producer_options):
            raise ValueError("implicit conversion forbidden")
        return self


class FeasibleCandidate(Value):
    identity: Checksum
    mapping_ref: Ref
    candidate: Candidate
    dependencies: tuple[Dependency, ...] = Field(min_length=1, max_length=64)
    components: tuple[Component, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def coherent(self):
        require_kind(self.mapping_ref, "mapping")
        unique(self.dependencies, lambda d: d.port_name)
        expected = {i.port.port_name: i for i in self.candidate.inputs}
        if set(expected) != {d.port_name for d in self.dependencies}:
            raise ValueError("complete AND inputs required")
        for d in self.dependencies:
            i = expected[d.port_name]
            if (d.required, d.cardinality_min, d.cardinality_max, d.constraint) != (
                i.required,
                i.cardinality_min,
                i.cardinality_max,
                i.constraint,
            ):
                raise ValueError("dependency differs from frozen port")
        if tuple(c.name for c in self.components) != METRICS:
            raise ValueError("complete canonical components required")
        if self.identity != digest(
            dict(
                mapping_ref=self.mapping_ref,
                candidate=self.candidate,
                dependencies=self.dependencies,
                components=self.components,
            )
        ):
            raise ValueError("candidate content mismatch")
        return self


class RejectedCandidate(Value):
    source: PathSource
    mapping_ref: Ref
    capability_ref: Ref
    candidate_code: Code
    path: Path
    reasons: tuple[Finding, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def coherent(self):
        unique(self.reasons, lambda f: digest(f))
        if (self.mapping_ref, self.capability_ref, self.candidate_code, self.path) != (
            self.source.mapping_ref,
            self.source.candidate.capability_ref,
            self.source.candidate.code,
            self.source.path,
        ) or any(f.source is not None and f.source != self.source for f in self.reasons):
            raise ValueError("rejection/source mismatch")
        return self


class DeferredCandidate(Value):
    mapping_ref: Ref
    candidate_code: Code
    path: Path
    reason: Literal["depth_exhausted", "budget_exhausted"]


class ExtensionSuggestion(Value):
    capability_gap: CapabilityGap
    target_capability_ref: Ref
    target_tool_ref: Ref
    target_implementation_ref: Ref
    constraint: ArtifactConstraint
    source_paths: tuple[PathSource, ...] = Field(min_length=1, max_length=512)
    affected_inputs: tuple[InputImpact, ...] = Field(max_length=8192)
    finding_checksums: tuple[Checksum, ...] = Field(min_length=1)
    requirement_ref: Ref
    catalog_ref: Ref
    policy_ref: Ref
    task_creation_owner: Literal["application/Runtime/11"] = "application/Runtime/11"
    requirement_document_port: Literal["PostgresRuntimeRepository.save_tool_requirement"] = (
        "PostgresRuntimeRepository.save_tool_requirement"
    )
    gap_port: Literal["PostgresRuntimeRepository.save_capability_gap"] = (
        "PostgresRuntimeRepository.save_capability_gap"
    )
    action: Literal["suggest_only_for_future_catalog"] = "suggest_only_for_future_catalog"

    @model_validator(mode="after")
    def coherent(self):
        for ref, kind in (
            (self.target_capability_ref, "capability"),
            (self.target_tool_ref, "tool"),
            (self.target_implementation_ref, "implementation"),
            (self.requirement_ref, "evidence_requirement"),
            (self.catalog_ref, "catalog"),
            (self.policy_ref, "policy"),
        ):
            require_kind(ref, kind)
        unique(self.finding_checksums)
        unique(self.source_paths, digest)
        unique(self.affected_inputs, digest)
        if self.capability_gap.capability_code != self.target_capability_ref.code:
            raise ValueError("CapabilityGap target capability mismatch")
        for source in self.source_paths:
            c = source.candidate
            if (c.capability_ref, c.tool_ref, c.implementation_ref, c.output) != (
                self.target_capability_ref,
                self.target_tool_ref,
                self.target_implementation_ref,
                self.constraint,
            ):
                raise ValueError("extension target differs from actual failed producer")
        return self


class QueryResult(Value):
    schema_version: Literal["grounding-query-result/2.0.0"] = "grounding-query-result/2.0.0"
    request: QueryRequest
    binding_checksum: Checksum
    safety_checksum: Checksum
    checksum: Checksum
    status: Literal["existing_artifact", "candidates", "gaps", "truncated"]
    existing_artifacts: tuple[str, ...] = Field(max_length=256)
    candidates: tuple[FeasibleCandidate, ...] = Field(max_length=512)
    top_k: tuple[Checksum, ...] = Field(max_length=128)
    rejected: tuple[RejectedCandidate, ...] = Field(max_length=512)
    deferred: tuple[DeferredCandidate, ...] = Field(max_length=4096)
    gaps: tuple[Finding, ...] = Field(max_length=8192)
    product_demands: tuple[ProductDemand, ...] = Field(max_length=8192)
    suggestions: tuple[ExtensionSuggestion, ...] = Field(max_length=512)
    evaluations: int = Field(ge=0, le=512)
    operations: int = Field(ge=0, le=100000)
    enumeration: Literal[
        "complete", "budget_exhausted", "depth_exhausted", "existing_artifact_short_circuit"
    ]
    expert_review_recommended: bool
    clinical_effect: Literal["NONE"] = "NONE"
    execution_compiled: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        unique(self.existing_artifacts)
        unique(self.candidates, lambda c: (c.mapping_ref, c.candidate.code))
        unique(self.candidates, lambda c: c.identity)
        unique(self.rejected, lambda c: (c.mapping_ref, c.candidate_code, c.path))
        unique(self.gaps, lambda f: digest(f))
        unique(self.top_k)
        if self.binding_checksum != digest(self.request):
            raise ValueError("input binding mismatch")
        if self.top_k != tuple(c.identity for c in self.candidates[: self.request.config.top_k]):
            raise ValueError("Top-K is not ranked prefix")
        if any(g.scope != self.request.scope for g in self.gaps):
            raise ValueError("cross-scope gap")
        if any(
            digest(g) not in {digest(f) for f in self.gaps}
            for r in self.rejected
            for g in r.reasons
        ):
            raise ValueError("rejection reasons missing from gaps")
        valid_assets = {a.asset_code for a in self.request.artifacts}
        if not set(self.existing_artifacts) <= valid_assets:
            raise ValueError("invented existing artifact")
        truncated = self.enumeration in {"budget_exhausted", "depth_exhausted"}
        expected = (
            "truncated"
            if truncated
            else "existing_artifact"
            if self.existing_artifacts
            else "candidates"
            if self.candidates
            else "gaps"
        )
        if self.status != expected or bool(self.deferred) != truncated:
            raise ValueError("status/enumeration mismatch")
        if self.status == "gaps" and not self.gaps:
            raise ValueError("empty unexplained result")
        if self.existing_artifacts and (
            self.candidates
            or self.evaluations
            or self.enumeration != "existing_artifact_short_circuit"
        ):
            raise ValueError("existing evidence must short circuit candidate search")
        if (
            self.evaluations > self.request.config.max_evaluations
            or self.operations > self.request.config.max_operations
            or len(self.gaps) > self.request.config.max_findings
        ):
            raise ValueError("evaluation budget exceeded")
        rejected_roots = {
            (r.mapping_ref, r.candidate_code) for r in self.rejected if len(r.path) == 1
        }
        if rejected_roots & {(c.mapping_ref, c.candidate.code) for c in self.candidates}:
            raise ValueError("candidate both admitted and rejected")
        from .compatibility import artifact_compatibility
        from .query import normalized_request, rank_key

        if self.request != normalized_request(self.request):
            raise ValueError("request is not in canonical set order")
        if self.candidates != tuple(
            sorted(self.candidates, key=lambda c: rank_key(c, self.request.config))
        ):
            raise ValueError("candidate ordering differs from versioned ranking")
        if self.expert_review_recommended != (not bool(self.existing_artifacts or self.candidates)):
            raise ValueError("expert recommendation/status mismatch")
        assets = {a.asset_code: a for a in self.request.artifacts}

        def check_candidate(candidate, depth):
            if (
                depth > self.request.config.max_depth
                or candidate.mapping_ref not in self.request.lock.mapping_refs
            ):
                raise ValueError("candidate outside lock or depth bound")
            for component in candidate.components:
                disabled = (
                    component.name == "information_gain"
                    and not self.request.config.information_gain_enabled
                )
                if (component.state == "disabled") != disabled:
                    raise ValueError("component/config mismatch")
            for dep in candidate.dependencies:
                for code in dep.artifact_options:
                    if code not in assets or artifact_compatibility(
                        assets[code], dep.constraint, self.request
                    ):
                        raise ValueError("forged or incompatible artifact binding")
                for producer in dep.producer_options:
                    check_candidate(producer, depth + 1)

        for candidate in self.candidates:
            check_candidate(candidate, 1)
        for finding in self.gaps:
            if finding.catalog_ref != self.request.lock.catalog_ref or (
                finding.source is not None
                and not set(finding.source.mapping_path) <= set(self.request.lock.mapping_refs)
            ):
                raise ValueError("finding outside locked catalog/mapping")
        # Semantic derivation is shared with query construction, not just checksums.
        # Official validate_result also re-fetches current mappings/catalog and recomputes.
        from .attribution import derive_attribution

        demands, suggestions = derive_attribution(
            self.request, self.candidates, self.rejected, self.gaps
        )
        if self.product_demands != demands or self.suggestions != suggestions:
            raise ValueError("product demand/extension target attribution mismatch")
        check_seal(self)
        return self
