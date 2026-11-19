"""Pydantic v2 message contracts for schema version 1.0."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .enums import (
    ActorType,
    CapabilityAvailability,
    ClinicalEffect,
    ConclusionStatus,
    DataAvailability,
    EdgeType,
    EvidenceObservationStatus,
    EvidenceValidity,
    ExecutionPolicy,
    ExpertAssessmentDecision,
    ExpertAssessmentStatus,
    GraphNodeStatus,
    GraphUpdateSource,
    IntendedUse,
    NodeType,
    OfflineReviewStatus,
    PlanAction,
    ProposalKind,
    QCStatus,
    RunStatus,
    StepStatus,
    ToolCatalogStatus,
    ToolExecutionStatus,
    ToolExtensionJobStatus,
    ToolGapCode,
    ToolValidationStatus,
    ValidationScope,
)
from .identifiers import CodeNamespace, StableCode, validate_stable_code
from .semantics import (
    EvidenceSemantics,
    ToolImplementationIdentity,
    validate_actor_proposal,
    validate_evidence_semantics,
    validate_tool_use,
)
from .state_machines import require_transition

SCHEMA_VERSION = "1.0"

_STABLE_CODE_PATTERN = r"^[A-Z][A-Z0-9_]{1,31}(?::[A-Z0-9][A-Z0-9._-]{0,63})+$"
_OPAQUE_PATTERN = r"^[A-Z][A-Z0-9_]{1,31}:(?:ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$"
_CHECKSUM_PATTERN = r"^sha256:[0-9a-f]{64}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$"
_TOKEN_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_UNIT_PATTERN = r"^(?:1|[A-Za-z%][A-Za-z0-9%*/.^_-]{0,31})$"
_MEDIA_TYPE_PATTERN = r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$"
_OBJECT_URI_RE = re.compile(r"^(?:https|s3|gs|az|object)://[^\s]+$")


def _stable_code(value: str) -> str:
    return str(validate_stable_code(value))


def _code_in(namespace: CodeNamespace):  # type: ignore[no-untyped-def]
    def validate(value: str) -> str:
        return str(validate_stable_code(value, namespace))

    return validate


StrictText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1000)]
ShortText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
Token = Annotated[str, StringConstraints(strict=True, pattern=_TOKEN_PATTERN)]
VersionToken = Annotated[str, StringConstraints(strict=True, pattern=_VERSION_PATTERN)]
Checksum = Annotated[str, StringConstraints(strict=True, pattern=_CHECKSUM_PATTERN)]
UnitCode = Annotated[str, StringConstraints(strict=True, pattern=_UNIT_PATTERN)]
MediaType = Annotated[str, StringConstraints(strict=True, pattern=_MEDIA_TYPE_PATTERN)]
StableCodeValue = Annotated[
    str,
    StringConstraints(strict=True, pattern=_STABLE_CODE_PATTERN),
    AfterValidator(_stable_code),
]
OpaqueEntityCode = Annotated[
    str,
    StringConstraints(strict=True, pattern=_OPAQUE_PATTERN),
    AfterValidator(_stable_code),
]
CaseCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.CASE))]
EncounterCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.ENCOUNTER))]
GraphCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.GRAPH))]
MessageCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.MESSAGE))]
RunCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.RUN))]
StepCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.STEP))]
TaskCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.TASK))]
AssetCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.ASSET))]
ValidationCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.VALIDATION))]
EpisodeCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.EPISODE))]
CounterfactualCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.CFPAIR))]
RequirementCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.REQUIREMENT))]
ExpertAssessmentCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.EXPERT_ASSESSMENT))]
ImplementationCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.IMPLEMENTATION))]
PolicyCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.POLICY))]
ToolCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.TOOL))]
CapabilityCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.CAPABILITY))]
WorkflowCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.WORKFLOW))]
TemplateCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.TEMPLATE))]
GoalCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.GOAL))]
ArtifactTypeCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.ARTIFACT))]
RuleCode = Annotated[str, AfterValidator(_code_in(CodeNamespace.RULE))]


class Laterality(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BILATERAL = "BILATERAL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CoordinateSpace(StrEnum):
    PIXEL_2D = "PIXEL_2D"
    VOXEL_3D = "VOXEL_3D"
    PATIENT = "PATIENT"
    SCANNER = "SCANNER"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ScopeType(StrEnum):
    GLOBAL = "GLOBAL"
    CASE = "CASE"
    RUN = "RUN"
    GRAPH = "GRAPH"
    STEP = "STEP"
    TOOL_EXTENSION = "TOOL_EXTENSION"
    MEMORY = "MEMORY"


class ExpertTaskType(StrEnum):
    ACTION_REVIEW = "ACTION_REVIEW"
    EVIDENCE_REVIEW = "EVIDENCE_REVIEW"
    GRAPH_REVIEW = "GRAPH_REVIEW"
    STOP_REVIEW = "STOP_REVIEW"
    COUNTERFACTUAL_REVIEW = "COUNTERFACTUAL_REVIEW"


class RequirementFailureStage(StrEnum):
    ELIGIBILITY = "ELIGIBILITY"
    ADAPT = "ADAPT"
    SANDBOX = "SANDBOX"
    SMOKE = "SMOKE"
    BENCHMARK = "BENCHMARK"
    POLICY_GATE = "POLICY_GATE"


class CaseImpact(StrEnum):
    NONE = "NONE"
    REDUCED_COMPLETENESS = "REDUCED_COMPLETENESS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CounterfactualPairType(StrEnum):
    REMOVE_EVIDENCE = "REMOVE_EVIDENCE"
    REVERSE_EVIDENCE = "REVERSE_EVIDENCE"
    SELECT_ALTERNATIVE_CHECK = "SELECT_ALTERNATIVE_CHECK"
    REPLACE_TOOL = "REPLACE_TOOL"
    FAILURE_RECOVERY = "FAILURE_RECOVERY"


class CounterfactualOperation(StrEnum):
    REMOVE = "REMOVE"
    REVERSE = "REVERSE"
    REPLACE = "REPLACE"
    RETRY_WITH_ALTERNATIVE = "RETRY_WITH_ALTERNATIVE"


class ContractModel(BaseModel):
    """Strict immutable base for nested contract structures."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class VersionedReference(ContractModel):
    code: StableCodeValue
    version: VersionToken
    checksum: Checksum | None = None


class GraphSnapshotReference(ContractModel):
    graph_code: GraphCode
    graph_version: int = Field(ge=1)
    checksum: Checksum


class ImmutablePolicyReference(ContractModel):
    policy_code: PolicyCode
    version: VersionToken
    checksum: Checksum


class ScopeReference(ContractModel):
    scope_type: ScopeType
    case_code: CaseCode | None = None
    run_id: RunCode | None = None
    graph_code: GraphCode | None = None
    step_id: StepCode | None = None
    task_id: TaskCode | None = None
    episode_code: EpisodeCode | None = None

    @model_validator(mode="after")
    def references_match_scope_type(self) -> ScopeReference:
        reference_values = {
            "case_code": self.case_code,
            "run_id": self.run_id,
            "graph_code": self.graph_code,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "episode_code": self.episode_code,
        }
        required_and_allowed = {
            ScopeType.GLOBAL: (frozenset(), frozenset()),
            ScopeType.CASE: (frozenset({"case_code"}), frozenset({"case_code"})),
            ScopeType.RUN: (
                frozenset({"run_id"}),
                frozenset({"case_code", "run_id", "graph_code"}),
            ),
            ScopeType.GRAPH: (
                frozenset({"graph_code"}),
                frozenset({"case_code", "run_id", "graph_code"}),
            ),
            ScopeType.STEP: (
                frozenset({"run_id", "step_id"}),
                frozenset({"case_code", "run_id", "graph_code", "step_id", "episode_code"}),
            ),
            ScopeType.TOOL_EXTENSION: (
                frozenset({"task_id"}),
                frozenset({"case_code", "run_id", "task_id"}),
            ),
            ScopeType.MEMORY: (
                frozenset({"episode_code"}),
                frozenset({"case_code", "run_id", "graph_code", "episode_code"}),
            ),
        }
        required, allowed = required_and_allowed[self.scope_type]
        missing = sorted(name for name in required if reference_values[name] is None)
        forbidden = sorted(
            name
            for name, value in reference_values.items()
            if value is not None and name not in allowed
        )
        if missing:
            raise ValueError(f"{self.scope_type.value} scope requires {', '.join(missing)}")
        if forbidden:
            raise ValueError(
                f"{self.scope_type.value} scope forbids references: {', '.join(forbidden)}"
            )
        return self


class Provenance(ContractModel):
    actor: ActorType
    created_at: datetime
    generator_version: VersionToken
    source_codes: tuple[StableCodeValue, ...] = Field(min_length=1)
    rule_versions: tuple[VersionedReference, ...] = ()
    knowledge_versions: tuple[VersionedReference, ...] = ()
    tool_versions: tuple[VersionedReference, ...] = ()
    input_checksums: tuple[Checksum, ...] = ()

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include an explicit timezone")
        return value


_FORBIDDEN_EXTENSION_KEY_PARTS = frozenset(
    {
        "catalog",
        "clinical_effect",
        "decision",
        "extension_job",
        "graph_delta",
        "onboarding",
        "policy_gate",
        "qc",
        "reasoning",
        "state",
        "status",
        "thought",
        "validation",
    }
)


def _validate_extension_mapping(value: object, *, path: str = "extensions") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = key.lower().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_EXTENSION_KEY_PARTS):
                raise ValueError(f"{path}.{key} may not control state or hidden reasoning")
            _validate_extension_mapping(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_extension_mapping(nested, path=f"{path}[{index}]")


class BaseContract(ContractModel):
    allowed_scope_types: ClassVar[frozenset[ScopeType]] = frozenset()
    scope_identity_fields: ClassVar[tuple[tuple[str, str], ...]] = ()

    schema_version: Literal["1.0"]
    contract_type: str
    provenance: Provenance
    scope: ScopeReference
    laterality: Laterality
    coordinate_space: CoordinateSpace
    unit: UnitCode | None
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("extensions")
    @classmethod
    def extensions_are_non_controlling(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_extension_mapping(value)
        return value

    @model_validator(mode="after")
    def scope_type_is_allowed_for_contract(self) -> BaseContract:
        if self.scope.scope_type not in self.allowed_scope_types:
            raise ValueError(
                f"{self.contract_type} does not allow {self.scope.scope_type.value} scope"
            )
        for model_field, scope_field in self.scope_identity_fields:
            model_value = getattr(self, model_field)
            scope_value = getattr(self.scope, scope_field)
            if scope_value is None:
                raise ValueError(
                    f"{self.contract_type} scope.{scope_field} is required to bind {model_field}"
                )
            if scope_value != model_value:
                raise ValueError(
                    f"{self.contract_type} scope.{scope_field} must match {model_field}"
                )
        return self


class Measurement(ContractModel):
    name: Token
    value: float
    unit: UnitCode


class Artifact(BaseContract):
    allowed_scope_types = frozenset({ScopeType.CASE, ScopeType.RUN, ScopeType.STEP})
    scope_identity_fields = ()

    contract_type: Literal["Artifact"]
    asset_code: AssetCode
    artifact_type_code: ArtifactTypeCode
    uri: StrictText
    checksum: Checksum
    media_type: MediaType
    qc_status: QCStatus
    dimensions: tuple[int, ...] = Field(default=(), max_length=5)
    measurements: tuple[Measurement, ...] = ()
    derived_from: tuple[AssetCode, ...] = ()
    research_only: bool

    @field_validator("uri")
    @classmethod
    def require_nonlocal_object_uri(cls, value: str) -> str:
        if not _OBJECT_URI_RE.fullmatch(value):
            raise ValueError(
                "uri must be a non-local object URI; file and absolute paths forbidden"
            )
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("artifact dimensions must be positive")
        return value


class GoalRequest(ContractModel):
    goal_code: GoalCode
    goal_level: Token


class StructuredObservation(ContractModel):
    concept_code: StableCodeValue
    observation: EvidenceObservationStatus
    value: JsonValue | None
    unit: UnitCode | None
    observed_at: datetime | None = None
    source_code: StableCodeValue


class CaseBundle(BaseContract):
    allowed_scope_types = frozenset({ScopeType.CASE})
    scope_identity_fields = (("case_code", "case_code"),)

    contract_type: Literal["CaseBundle"]
    case_code: CaseCode
    patient_hash: Checksum
    encounter_code: EncounterCode
    goals: tuple[GoalRequest, ...] = Field(min_length=1)
    artifacts: tuple[Artifact, ...] = ()
    observations: tuple[StructuredObservation, ...] = ()
    report_uri: StrictText | None = None

    @field_validator("report_uri")
    @classmethod
    def validate_report_uri(cls, value: str | None) -> str | None:
        if value is not None and not _OBJECT_URI_RE.fullmatch(value):
            raise ValueError("report_uri must be a non-local object URI")
        return value

    @model_validator(mode="after")
    def case_scope_matches(self) -> CaseBundle:
        for artifact in self.artifacts:
            if artifact.scope.case_code != self.case_code:
                raise ValueError("CaseBundle artifact scope.case_code must match case_code")
        return self


class TemplateNode(ContractModel):
    node_code: StableCodeValue
    node_type: NodeType
    concept_code: StableCodeValue
    required: bool


class GraphEdge(ContractModel):
    source_node_code: StableCodeValue
    target_node_code: StableCodeValue
    edge_type: EdgeType
    rule_ref: VersionedReference | None = None


def _validate_graph_members(
    nodes: tuple[TemplateNode | GraphNodeInstance, ...],
    edges: tuple[GraphEdge, ...],
    goal_node_code: str,
) -> None:
    node_codes = tuple(node.node_code for node in nodes)
    if len(set(node_codes)) != len(node_codes):
        raise ValueError("node_code values must be unique")
    if goal_node_code not in node_codes:
        raise ValueError("goal_node_code must reference a declared node")
    known = set(node_codes)
    for edge in edges:
        if edge.source_node_code not in known or edge.target_node_code not in known:
            raise ValueError("every edge endpoint must reference a declared node")


class GECEGTemplate(BaseContract):
    allowed_scope_types = frozenset({ScopeType.GLOBAL})
    scope_identity_fields = ()

    contract_type: Literal["GECEGTemplate"]
    template_code: TemplateCode
    template_version: VersionToken
    goal_node_code: StableCodeValue
    nodes: tuple[TemplateNode, ...] = Field(min_length=1)
    edges: tuple[GraphEdge, ...] = ()
    citations: tuple[VersionedReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def graph_is_closed(self) -> GECEGTemplate:
        _validate_graph_members(self.nodes, self.edges, self.goal_node_code)
        return self


class GraphNodeInstance(ContractModel):
    node_code: StableCodeValue
    node_type: NodeType
    concept_code: StableCodeValue
    status: GraphNodeStatus
    availability: DataAvailability
    evidence_bindings: tuple[StableCodeValue, ...] = ()


class GECEGInstance(BaseContract):
    allowed_scope_types = frozenset({ScopeType.GRAPH})
    scope_identity_fields = (("graph_code", "graph_code"), ("case_code", "case_code"))

    contract_type: Literal["GECEGInstance"]
    graph_code: GraphCode
    case_code: CaseCode
    template_ref: VersionedReference
    graph_version: int = Field(ge=1)
    goal_node_code: StableCodeValue
    nodes: tuple[GraphNodeInstance, ...] = Field(min_length=1)
    edges: tuple[GraphEdge, ...] = ()

    @model_validator(mode="after")
    def graph_is_closed(self) -> GECEGInstance:
        _validate_graph_members(self.nodes, self.edges, self.goal_node_code)
        return self


class AddNodeOperation(ContractModel):
    op: Literal["ADD_NODE"]
    node: GraphNodeInstance


class AddEdgeOperation(ContractModel):
    op: Literal["ADD_EDGE"]
    edge: GraphEdge


class UpdateNodeStatusOperation(ContractModel):
    op: Literal["UPDATE_NODE_STATUS"]
    node_code: StableCodeValue
    from_status: GraphNodeStatus
    to_status: GraphNodeStatus
    source: GraphUpdateSource
    evidence_binding: StableCodeValue | None = None

    @model_validator(mode="after")
    def transition_is_legal(self) -> UpdateNodeStatusOperation:
        require_transition(self.from_status, self.to_status)
        from .semantics import validate_graph_status_source

        validate_graph_status_source(self.to_status, self.source)
        return self


GraphOperation = Annotated[
    AddNodeOperation | AddEdgeOperation | UpdateNodeStatusOperation,
    Field(discriminator="op"),
]


class GraphDelta(BaseContract):
    allowed_scope_types = frozenset({ScopeType.GRAPH})
    scope_identity_fields = (("graph_code", "graph_code"),)

    contract_type: Literal["GraphDelta"]
    graph_code: GraphCode
    base_version: int = Field(ge=1)
    idempotency_key: TaskCode
    proposed_by: ActorType
    proposal_kind: ProposalKind
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)
    operations: tuple[GraphOperation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def actor_owns_proposal(self) -> GraphDelta:
        if self.proposal_kind not in {
            ProposalKind.GRAPH_DELTA_CANDIDATE,
            ProposalKind.GRAPH_DELTA_VALIDATION,
            ProposalKind.GRAPH_DELTA_APPLICATION,
        }:
            raise ValueError("GraphDelta requires a graph-delta proposal kind")
        validate_actor_proposal(self.proposed_by, self.proposal_kind)
        if self.provenance.actor is not self.proposed_by:
            raise ValueError("GraphDelta proposed_by must match provenance.actor")
        return self


class ArtifactBinding(ContractModel):
    port_name: Token
    asset_code: AssetCode


class ToolExecutionRequest(BaseContract):
    allowed_scope_types = frozenset({ScopeType.RUN})
    scope_identity_fields = (("case_code", "case_code"), ("run_id", "run_id"))

    contract_type: Literal["ToolExecutionRequest"]
    request_id: TaskCode
    case_code: CaseCode
    run_id: RunCode
    target_node_code: StableCodeValue
    workflow_ref: VersionedReference
    implementation_code: ImplementationCode
    implementation_version: VersionToken
    implementation_checksum: Checksum
    inputs: tuple[ArtifactBinding, ...] = Field(min_length=1)
    execution_policy: ExecutionPolicy
    catalog_status: ToolCatalogStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse
    decisive_evidence: bool
    dry_run: bool
    idempotency_key: TaskCode

    @model_validator(mode="after")
    def tool_is_allowed(self) -> ToolExecutionRequest:
        if self.provenance.actor is not ActorType.TOOL_AGENT:
            raise ValueError("ToolExecutionRequest provenance actor must be TOOL_AGENT")
        if self.intended_use is not IntendedUse.RESEARCH_ONLY:
            raise ValueError("current research strategy requires research_only intended_use")
        validate_tool_use(
            self.catalog_status,
            self.validation_status,
            self.intended_use,
            self.execution_policy,
            decisive_evidence=self.decisive_evidence,
        )
        ToolImplementationIdentity(
            implementation_code=StableCode(self.implementation_code),
            version=self.implementation_version,
            checksum=self.implementation_checksum,
        )
        return self


class ErrorDetail(ContractModel):
    code: Token
    message: StrictText
    retryable: bool


class ToolExecutionResult(BaseContract):
    allowed_scope_types = frozenset({ScopeType.RUN})
    scope_identity_fields = (("workflow_run_id", "run_id"),)

    contract_type: Literal["ToolExecutionResult"]
    request_id: TaskCode
    workflow_run_id: RunCode
    status: ToolExecutionStatus
    execution_policy: ExecutionPolicy
    implementation_code: ImplementationCode
    implementation_version: VersionToken
    implementation_checksum: Checksum
    catalog_status: ToolCatalogStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse
    decisive_evidence: bool
    qc_status: QCStatus
    artifacts: tuple[Artifact, ...] = ()
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def execution_layers_are_consistent(self) -> ToolExecutionResult:
        if self.provenance.actor is not ActorType.TOOL_AGENT:
            raise ValueError("ToolExecutionResult provenance actor must be TOOL_AGENT")
        if self.intended_use is not IntendedUse.RESEARCH_ONLY:
            raise ValueError("current research strategy requires research_only intended_use")
        validate_tool_use(
            self.catalog_status,
            self.validation_status,
            self.intended_use,
            self.execution_policy,
            decisive_evidence=self.decisive_evidence,
        )
        if self.status is not ToolExecutionStatus.SUCCEEDED and self.qc_status is QCStatus.PASS:
            raise ValueError("non-SUCCEEDED tool execution cannot have QC PASS")
        if self.status is ToolExecutionStatus.SUCCEEDED:
            if self.error is not None or not self.artifacts:
                raise ValueError("SUCCEEDED requires artifacts and forbids error")
            artifact_qc = tuple(artifact.qc_status for artifact in self.artifacts)
            if QCStatus.FAIL in artifact_qc:
                expected_qc = QCStatus.FAIL
            elif all(status is QCStatus.PASS for status in artifact_qc):
                expected_qc = QCStatus.PASS
            elif all(status is QCStatus.UNASSESSED for status in artifact_qc):
                expected_qc = QCStatus.UNASSESSED
            else:
                expected_qc = QCStatus.PENDING
            if self.qc_status is not expected_qc:
                raise ValueError(
                    "overall qc_status must equal the conservative aggregate of artifact QC"
                )
        elif self.artifacts:
            raise ValueError("non-SUCCEEDED result cannot publish artifacts")
        if (
            self.status
            in {
                ToolExecutionStatus.FAILED,
                ToolExecutionStatus.TIMED_OUT,
                ToolExecutionStatus.BLOCKED,
            }
            and self.error is None
        ):
            raise ValueError("failed, timed-out, or blocked result requires error")
        for artifact in self.artifacts:
            if not artifact.research_only:
                raise ValueError("tool output artifacts must be marked research_only")
            if artifact.scope.run_id != self.workflow_run_id:
                raise ValueError("tool output artifact run_id must match workflow_run_id")
            if artifact.scope.case_code != self.scope.case_code:
                raise ValueError("tool output artifact case_code must match result scope")
        return self


class ValidationResult(BaseContract):
    allowed_scope_types = frozenset({ScopeType.GRAPH, ScopeType.STEP})
    scope_identity_fields = ()

    contract_type: Literal["ValidationResult"]
    validation_code: ValidationCode
    validation_scope: ValidationScope
    target_node_code: StableCodeValue
    observation: EvidenceObservationStatus
    capability: CapabilityAvailability
    tool_execution_status: ToolExecutionStatus | None = None
    qc_status: QCStatus
    validity: EvidenceValidity
    clinical_effect: ClinicalEffect
    support_score: float | None = Field(default=None, ge=0, le=1)
    refute_score: float | None = Field(default=None, ge=0, le=1)
    rule_refs: tuple[VersionedReference, ...] = Field(min_length=1)
    evidence_bindings: tuple[StableCodeValue, ...] = ()
    graph_delta_candidate: GraphDelta | None = None
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_layers_are_consistent(self) -> ValidationResult:
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=self.observation,
                validity=self.validity,
                clinical_effect=self.clinical_effect,
                capability=self.capability,
                tool_execution=self.tool_execution_status,
                qc=self.qc_status,
            )
        )
        if self.provenance.actor is not ActorType.VERIFICATION_AGENT:
            raise ValueError("ValidationResult provenance actor must be VERIFICATION_AGENT")
        if self.graph_delta_candidate is not None:
            if self.validity is not EvidenceValidity.VALID:
                raise ValueError("GraphDelta candidate requires VALID evidence")
            if (
                self.graph_delta_candidate.proposed_by is not ActorType.VERIFICATION_AGENT
                or self.graph_delta_candidate.proposal_kind
                is not ProposalKind.GRAPH_DELTA_CANDIDATE
            ):
                raise ValueError("ValidationResult may contain only a Verification candidate")
            delta = self.graph_delta_candidate
            if self.scope.graph_code != delta.graph_code:
                raise ValueError("validation and GraphDelta must target the same graph")
            for scope_field in ("case_code", "run_id", "graph_code"):
                if getattr(self.scope, scope_field) != getattr(delta.scope, scope_field):
                    raise ValueError(f"validation and GraphDelta scope {scope_field} must match")
            if not self.evidence_bindings:
                raise ValueError("GraphDelta validation requires evidence_bindings")
            if len(delta.operations) != 1 or not isinstance(
                delta.operations[0], UpdateNodeStatusOperation
            ):
                raise ValueError(
                    "validation GraphDelta must contain exactly one node status update"
                )
            operation = delta.operations[0]
            if operation.node_code != self.target_node_code:
                raise ValueError("GraphDelta target must match validation target node")
            if operation.source is not GraphUpdateSource.VALIDATED_EVIDENCE:
                raise ValueError("validation GraphDelta source must be validated_evidence")
            if operation.evidence_binding not in self.evidence_bindings:
                raise ValueError("GraphDelta evidence binding must be declared by ValidationResult")
        return self


class PlannerState(BaseContract):
    allowed_scope_types = frozenset({ScopeType.RUN})
    scope_identity_fields = (("run_id", "run_id"), ("graph_code", "graph_code"))

    contract_type: Literal["PlannerState"]
    run_id: RunCode
    run_status: RunStatus
    graph_code: GraphCode
    graph_version: int = Field(ge=1)
    unresolved_nodes: tuple[StableCodeValue, ...]
    available_artifacts: tuple[AssetCode, ...]
    legal_actions: tuple[PlanAction, ...]
    remaining_budget: dict[Token, int]

    @field_validator("remaining_budget")
    @classmethod
    def budget_is_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(amount < 0 for amount in value.values()):
            raise ValueError("remaining budgets must be nonnegative")
        return value

    @model_validator(mode="after")
    def legal_actions_match_run_lifecycle(self) -> PlannerState:
        terminal = {
            RunStatus.COMPLETED_SUPPORTED,
            RunStatus.COMPLETED_REFUTED,
            RunStatus.COMPLETED_UNCERTAIN,
            RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
            RunStatus.FAILED_TECHNICAL,
            RunStatus.STOPPED_UNSAFE,
        }
        if self.run_status in terminal and self.legal_actions:
            raise ValueError("terminal PlannerState requires empty legal_actions")
        if self.run_status not in terminal and not self.legal_actions:
            raise ValueError("non-terminal PlannerState requires at least one legal action")
        return self


class PlannerAction(BaseContract):
    allowed_scope_types = frozenset({ScopeType.RUN})
    scope_identity_fields = (("run_id", "run_id"), ("graph_code", "graph_code"))

    contract_type: Literal["PlannerAction"]
    run_id: RunCode
    graph_code: GraphCode
    graph_version: int = Field(ge=1)
    proposed_by: ActorType
    proposal_kind: ProposalKind
    selected_action: PlanAction
    target_code: StableCodeValue | None = None
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)
    idempotency_key: TaskCode

    @model_validator(mode="after")
    def actor_owns_action(self) -> PlannerAction:
        validate_actor_proposal(self.proposed_by, self.proposal_kind)
        if self.provenance.actor is not self.proposed_by:
            raise ValueError("PlannerAction proposed_by must match provenance.actor")
        return self


class PlannerObservation(BaseContract):
    allowed_scope_types = frozenset({ScopeType.STEP})
    scope_identity_fields = (
        ("run_id", "run_id"),
        ("graph_code", "graph_code"),
        ("step_id", "step_id"),
    )

    contract_type: Literal["PlannerObservation"]
    run_id: RunCode
    graph_code: GraphCode
    step_id: StepCode
    step_status: StepStatus
    action: PlanAction
    observed_codes: tuple[StableCodeValue, ...] = ()
    tool_result_ref: TaskCode | None = None
    capability_gap_refs: tuple[RequirementCode, ...] = ()


class PlannerFeedback(BaseContract):
    allowed_scope_types = frozenset({ScopeType.STEP})
    scope_identity_fields = (
        ("run_id", "run_id"),
        ("graph_code", "graph_code"),
        ("step_id", "step_id"),
    )

    contract_type: Literal["PlannerFeedback"]
    run_id: RunCode
    graph_code: GraphCode
    step_id: StepCode
    accepted: bool
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)
    allowed_actions: tuple[PlanAction, ...]
    expected_graph_version: int = Field(ge=1)


class PublishedKnowledgeReference(ContractModel):
    source_ref: VersionedReference
    release_status: Literal["PUBLISHED"]


class ExpertAssessment(BaseContract):
    allowed_scope_types = frozenset({ScopeType.STEP})
    scope_identity_fields = (("run_id", "run_id"), ("step_id", "step_id"))

    contract_type: Literal["ExpertAssessment"]
    expert_assessment_id: ExpertAssessmentCode
    run_id: RunCode
    step_id: StepCode
    status: ExpertAssessmentStatus
    task_type: ExpertTaskType
    input_versions: tuple[VersionedReference, ...] = Field(min_length=1)
    proposal_code: StableCodeValue
    decision: ExpertAssessmentDecision | None = None
    allowed_action_codes: tuple[PlanAction, ...]
    selected_action_code: PlanAction | None = None
    finding_codes: tuple[RuleCode, ...] = Field(min_length=1)
    knowledge_refs: tuple[PublishedKnowledgeReference, ...] = Field(min_length=1)
    unresolved_nodes: tuple[StableCodeValue, ...]
    scope_limitations: tuple[StrictText, ...]
    summary: StrictText

    @model_validator(mode="after")
    def expert_output_is_bounded(self) -> ExpertAssessment:
        if self.provenance.actor is not ActorType.EXPERT_AGENT:
            raise ValueError("ExpertAssessment provenance actor must be EXPERT_AGENT")
        validate_actor_proposal(ActorType.EXPERT_AGENT, ProposalKind.EXPERT_REVIEW)
        if self.status is ExpertAssessmentStatus.COMPLETED and self.decision is None:
            raise ValueError("COMPLETED assessment requires decision")
        if self.status is not ExpertAssessmentStatus.COMPLETED and self.decision is not None:
            raise ValueError("only COMPLETED assessment may carry a decision")
        if self.decision is ExpertAssessmentDecision.SELECT_ALLOWED_ALTERNATIVE:
            if self.selected_action_code not in self.allowed_action_codes:
                raise ValueError("selected action must be a member of allowed_action_codes")
        elif self.selected_action_code is not None:
            raise ValueError("selected_action_code is only valid for allowed alternative decision")
        return self


class CapabilityGap(BaseContract):
    allowed_scope_types = frozenset({ScopeType.RUN})
    scope_identity_fields = (("origin_run_id", "run_id"),)

    contract_type: Literal["CapabilityGap"]
    gap_id: RequirementCode
    origin_run_id: RunCode
    target_node_code: StableCodeValue
    capability_code: CapabilityCode
    gap_codes: tuple[ToolGapCode, ...] = Field(min_length=1)
    observation: Literal[EvidenceObservationStatus.UNOBSERVED_TOOL_GAP]
    capability: Literal[CapabilityAvailability.UNAVAILABLE_CAPABILITY]
    validity: EvidenceValidity
    clinical_effect: Literal[ClinicalEffect.NONE]
    alternative_paths: tuple[StableCodeValue, ...]
    blocks_required_path: bool

    @model_validator(mode="after")
    def gap_is_not_clinical_evidence(self) -> CapabilityGap:
        validate_evidence_semantics(
            EvidenceSemantics(
                observation=self.observation,
                validity=self.validity,
                clinical_effect=self.clinical_effect,
                capability=self.capability,
            )
        )
        return self


class ArtifactRequirement(ContractModel):
    artifact_type_code: ArtifactTypeCode
    required: bool
    laterality: Laterality
    coordinate_space: CoordinateSpace
    unit: UnitCode | None


class AttemptedCandidate(ContractModel):
    candidate_code: ToolCode
    failed_stage: RequirementFailureStage
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)


class MetricRequirement(ContractModel):
    metric_code: RuleCode
    unit: UnitCode


class ValidationPlan(ContractModel):
    frozen_datasets: tuple[VersionedReference, ...]
    metrics: tuple[MetricRequirement, ...]
    threshold_policy: ImmutablePolicyReference | None = None


class ToolRequirementDocument(BaseContract):
    allowed_scope_types = frozenset({ScopeType.TOOL_EXTENSION})
    scope_identity_fields = (
        ("origin_run_id", "run_id"),
        ("extension_job_id", "task_id"),
    )

    contract_type: Literal["ToolRequirementDocument"]
    requirement_id: RequirementCode
    origin_run_id: RunCode
    extension_job_id: TaskCode
    target_node_code: StableCodeValue
    capability_code: CapabilityCode
    clinical_purpose: StrictText
    available_inputs: tuple[AssetCode, ...]
    required_input_contract: tuple[ArtifactRequirement, ...] = Field(min_length=1)
    required_output_contract: tuple[ArtifactRequirement, ...] = Field(min_length=1)
    attempted_candidates: tuple[AttemptedCandidate, ...]
    validation_plan: ValidationPlan
    blocked_nodes: tuple[StableCodeValue, ...]
    conclusion_effect: CaseImpact
    available_alternatives: tuple[StableCodeValue, ...]
    replay_when_available: bool


class MachineEvaluation(ContractModel):
    evaluation_code: ValidationCode
    job_id: TaskCode
    implementation_code: ImplementationCode
    implementation_version: VersionToken
    implementation_checksum: Checksum
    evaluator_version: VersionToken
    result_checksum: Checksum
    policy_ref: ImmutablePolicyReference
    passed: bool
    gate_evidence: tuple[VersionedReference, ...] = Field(min_length=1)
    reason_codes: tuple[RuleCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def implementation_identity_is_complete(self) -> MachineEvaluation:
        ToolImplementationIdentity(
            implementation_code=StableCode(self.implementation_code),
            version=self.implementation_version,
            checksum=self.implementation_checksum,
        )
        return self


class ToolOnboardingJob(BaseContract):
    allowed_scope_types = frozenset({ScopeType.TOOL_EXTENSION})
    scope_identity_fields = (("job_id", "task_id"),)

    contract_type: Literal["ToolOnboardingJob"]
    job_id: TaskCode
    extension_job_status: ToolExtensionJobStatus
    catalog_status: ToolCatalogStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse
    tool_code: ToolCode
    capability_code: CapabilityCode
    implementation_code: ImplementationCode
    implementation_version: VersionToken
    implementation_checksum: Checksum
    policy_ref: ImmutablePolicyReference
    source_catalog_version: VersionToken
    target_catalog_version: VersionToken
    affects_current_run: Literal[False]

    @model_validator(mode="after")
    def job_domains_are_consistent(self) -> ToolOnboardingJob:
        if self.provenance.actor is not ActorType.TOOL_AGENT:
            raise ValueError("ToolOnboardingJob provenance actor must be TOOL_AGENT")
        if self.intended_use is not IntendedUse.RESEARCH_ONLY:
            raise ValueError("automatic onboarding is restricted to research_only")
        ToolImplementationIdentity(
            implementation_code=StableCode(self.implementation_code),
            version=self.implementation_version,
            checksum=self.implementation_checksum,
        )
        if self.extension_job_status is ToolExtensionJobStatus.ACTIVATED_NEXT_RUN:
            if (
                self.catalog_status is not ToolCatalogStatus.ACTIVE
                or self.validation_status is not ToolValidationStatus.AUTO_VALIDATED_RESEARCH
                or self.intended_use is not IntendedUse.RESEARCH_ONLY
            ):
                raise ValueError(
                    "ACTIVATED_NEXT_RUN requires active + auto_validated_research + research_only"
                )
        elif self.catalog_status is ToolCatalogStatus.ACTIVE:
            raise ValueError("active catalog status requires ACTIVATED_NEXT_RUN")
        return self


class ToolOnboardingDecision(BaseContract):
    allowed_scope_types = frozenset({ScopeType.TOOL_EXTENSION})
    scope_identity_fields = (("job_id", "task_id"),)

    contract_type: Literal["ToolOnboardingDecision"]
    decision_id: ValidationCode
    job_id: TaskCode
    extension_job_status: ToolExtensionJobStatus
    catalog_status: ToolCatalogStatus
    validation_status: ToolValidationStatus
    intended_use: IntendedUse
    implementation_code: ImplementationCode
    implementation_version: VersionToken
    implementation_checksum: Checksum
    policy_ref: ImmutablePolicyReference
    machine_evaluation: MachineEvaluation
    published_catalog_version: VersionToken | None = None
    visible_to_current_run: Literal[False]

    @model_validator(mode="after")
    def decision_is_machine_gated(self) -> ToolOnboardingDecision:
        if self.provenance.actor is not ActorType.RULE_ENGINE:
            raise ValueError("ToolOnboardingDecision provenance actor must be RULE_ENGINE")
        if self.machine_evaluation.policy_ref != self.policy_ref:
            raise ValueError("machine evaluation must use the referenced immutable policy")
        if self.machine_evaluation.job_id != self.job_id:
            raise ValueError("machine evaluation job_id must match decision job_id")
        for field_name in (
            "implementation_code",
            "implementation_version",
            "implementation_checksum",
        ):
            if getattr(self.machine_evaluation, field_name) != getattr(self, field_name):
                raise ValueError(
                    f"machine evaluation {field_name} must match decision {field_name}"
                )
        if self.intended_use is not IntendedUse.RESEARCH_ONLY:
            raise ValueError("automatic onboarding is restricted to research_only")
        ToolImplementationIdentity(
            implementation_code=StableCode(self.implementation_code),
            version=self.implementation_version,
            checksum=self.implementation_checksum,
        )
        terminal_jobs = {
            ToolExtensionJobStatus.ACTIVATED_NEXT_RUN,
            ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
            ToolExtensionJobStatus.FAILED,
        }
        if self.extension_job_status not in terminal_jobs:
            raise ValueError("ToolOnboardingDecision requires a terminal extension job status")
        if self.extension_job_status is ToolExtensionJobStatus.ACTIVATED_NEXT_RUN:
            if not self.machine_evaluation.passed:
                raise ValueError("activation requires a passing machine evaluation")
            if self.published_catalog_version is None:
                raise ValueError("activation requires published_catalog_version")
            if self.validation_status is not ToolValidationStatus.AUTO_VALIDATED_RESEARCH:
                raise ValueError("activation requires auto_validated_research")
            if self.catalog_status is not ToolCatalogStatus.ACTIVE:
                raise ValueError("activation requires active catalog status")
            validate_tool_use(
                self.catalog_status,
                self.validation_status,
                self.intended_use,
                ExecutionPolicy.RESEARCH,
                decisive_evidence=True,
            )
        else:
            if self.catalog_status is ToolCatalogStatus.ACTIVE:
                raise ValueError("failed or documented extension cannot activate catalog entry")
            if self.published_catalog_version is not None:
                raise ValueError("non-activation decision cannot publish a catalog version")
        return self


class PlannerStateSummary(ContractModel):
    graph_code: GraphCode
    graph_version: int = Field(ge=1)
    run_status: RunStatus
    unresolved_nodes: tuple[StableCodeValue, ...]


class TrajectoryStep(BaseContract):
    allowed_scope_types = frozenset({ScopeType.STEP})
    scope_identity_fields = (
        ("case_code", "case_code"),
        ("run_id", "run_id"),
        ("graph_code", "graph_code"),
        ("step_id", "step_id"),
        ("episode_code", "episode_code"),
    )

    contract_type: Literal["TrajectoryStep"]
    case_code: CaseCode
    run_id: RunCode
    graph_code: GraphCode
    step_id: StepCode
    episode_code: EpisodeCode
    step_index: int = Field(ge=0)
    state: PlannerState
    legal_actions: tuple[PlanAction, ...] = Field(min_length=1)
    selected_action: PlannerAction
    observation: PlannerObservation
    next_state: PlannerState
    feedback: PlannerFeedback | None = None

    @model_validator(mode="after")
    def trajectory_links_are_consistent(self) -> TrajectoryStep:
        nested_contracts = (
            self.state,
            self.selected_action,
            self.observation,
            self.next_state,
        )
        scoped_contracts = nested_contracts + ((self.feedback,) if self.feedback else ())
        if any(item.scope.case_code != self.case_code for item in scoped_contracts):
            raise ValueError("all trajectory contract scopes must share case_code")
        if any(item.run_id != self.run_id for item in nested_contracts):
            raise ValueError("all trajectory contracts must share run_id")
        if self.observation.step_id != self.step_id:
            raise ValueError("observation step_id must match trajectory step_id")
        if self.feedback is not None:
            if self.feedback.run_id != self.run_id:
                raise ValueError("feedback run_id must match trajectory run_id")
            if self.feedback.step_id != self.step_id:
                raise ValueError("feedback step_id must match trajectory step_id")
        if any(item.graph_code != self.graph_code for item in scoped_contracts):
            raise ValueError("all trajectory contracts must share graph_code")
        if self.legal_actions != self.state.legal_actions:
            raise ValueError("trajectory legal_actions must equal state legal_actions")
        if self.selected_action.selected_action not in self.legal_actions:
            raise ValueError("selected action must be a member of legal_actions")
        if self.selected_action.graph_version != self.state.graph_version:
            raise ValueError("selected action graph_version must match state")
        if self.next_state.graph_version not in {
            self.state.graph_version,
            self.state.graph_version + 1,
        }:
            raise ValueError("next graph version must stay unchanged or advance by one")
        require_transition(self.state.run_status, self.next_state.run_status)
        if self.observation.action is not self.selected_action.selected_action:
            raise ValueError("observation action must match selected action")
        if self.feedback is not None:
            if self.feedback.expected_graph_version != self.state.graph_version:
                raise ValueError("feedback expected graph version must match state")
            expected_actions = (
                self.legal_actions if self.feedback.accepted else self.next_state.legal_actions
            )
            if self.feedback.allowed_actions != expected_actions:
                raise ValueError(
                    "feedback allowed_actions must preserve current legal actions when accepted; "
                    "rejected feedback must return next_state legal actions"
                )
        return self


class Episode(BaseContract):
    allowed_scope_types = frozenset({ScopeType.MEMORY})
    scope_identity_fields = (
        ("episode_code", "episode_code"),
        ("case_code", "case_code"),
        ("run_id", "run_id"),
    )

    contract_type: Literal["Episode"]
    episode_code: EpisodeCode
    case_code: CaseCode
    run_id: RunCode
    initial_graph_snapshot_ref: GraphSnapshotReference
    final_graph_snapshot_ref: GraphSnapshotReference
    tool_catalog_ref: VersionedReference
    policy_refs: tuple[ImmutablePolicyReference, ...] = Field(min_length=1)
    tool_refs: tuple[VersionedReference, ...]
    rule_refs: tuple[VersionedReference, ...] = Field(min_length=1)
    expert_assessment_refs: tuple[ExpertAssessmentCode, ...]
    conclusion_status: ConclusionStatus
    review_status: OfflineReviewStatus
    trajectory: tuple[TrajectoryStep, ...] = Field(min_length=1)
    eligible_for_retrieval: bool

    @model_validator(mode="after")
    def retrieval_requires_approval(self) -> Episode:
        if self.eligible_for_retrieval and self.review_status is not OfflineReviewStatus.APPROVED:
            raise ValueError("only an APPROVED episode may be eligible for retrieval")
        indices = tuple(step.step_index for step in self.trajectory)
        if indices != tuple(range(len(indices))):
            raise ValueError("trajectory step_index values must be contiguous from zero")
        if self.initial_graph_snapshot_ref.graph_code != self.final_graph_snapshot_ref.graph_code:
            raise ValueError("episode initial and final graph snapshots must reference one graph")
        if self.scope.graph_code != self.final_graph_snapshot_ref.graph_code:
            raise ValueError("episode scope graph must match graph snapshots")
        if (
            self.initial_graph_snapshot_ref.graph_version
            > self.final_graph_snapshot_ref.graph_version
        ):
            raise ValueError("episode initial graph version cannot exceed final graph version")
        for step in self.trajectory:
            if step.episode_code != self.episode_code:
                raise ValueError("trajectory episode_code must match containing episode")
            if step.case_code != self.case_code:
                raise ValueError("trajectory case_code must match containing episode")
            if step.run_id != self.run_id:
                raise ValueError("trajectory run_id must match containing episode")
            if step.graph_code != self.final_graph_snapshot_ref.graph_code:
                raise ValueError("trajectory graph_code must match episode graph snapshot")
        step_ids = tuple(step.step_id for step in self.trajectory)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("trajectory step_id values must be unique")
        for previous, current in zip(self.trajectory, self.trajectory[1:], strict=False):
            if previous.next_state.graph_version != current.state.graph_version:
                raise ValueError("adjacent trajectory graph versions must be continuous")
            if previous.next_state.run_status is not current.state.run_status:
                raise ValueError("adjacent trajectory run statuses must be continuous")
        if self.trajectory[0].state.graph_version != self.initial_graph_snapshot_ref.graph_version:
            raise ValueError("initial trajectory graph version must match initial graph snapshot")
        final_state = self.trajectory[-1].next_state
        if final_state.graph_version != self.final_graph_snapshot_ref.graph_version:
            raise ValueError("final trajectory graph version must match graph snapshot")
        conclusion_run_status = {
            ConclusionStatus.SUPPORTED: RunStatus.COMPLETED_SUPPORTED,
            ConclusionStatus.REFUTED: RunStatus.COMPLETED_REFUTED,
            ConclusionStatus.UNCERTAIN_MISSING_EVIDENCE: (
                RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE
            ),
            ConclusionStatus.UNCERTAIN_CONFLICTING_EVIDENCE: RunStatus.COMPLETED_UNCERTAIN,
            ConclusionStatus.UNABLE_TO_EVALUATE_TECHNICAL: RunStatus.FAILED_TECHNICAL,
            ConclusionStatus.CAPABILITY_UNAVAILABLE: RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
            ConclusionStatus.STOPPED_UNSAFE: RunStatus.STOPPED_UNSAFE,
        }
        if final_state.run_status is not conclusion_run_status[self.conclusion_status]:
            raise ValueError("conclusion_status must match final next_state.run_status")
        return self


class CounterfactualIntervention(ContractModel):
    target_code: StableCodeValue
    operation: CounterfactualOperation
    replacement_code: StableCodeValue | None = None

    @model_validator(mode="after")
    def replacement_is_explicit(self) -> CounterfactualIntervention:
        if (
            self.operation
            in {
                CounterfactualOperation.REPLACE,
                CounterfactualOperation.RETRY_WITH_ALTERNATIVE,
            }
            and self.replacement_code is None
        ):
            raise ValueError("replacement operation requires replacement_code")
        if (
            self.operation
            in {
                CounterfactualOperation.REMOVE,
                CounterfactualOperation.REVERSE,
            }
            and self.replacement_code is not None
        ):
            raise ValueError("remove/reverse operation forbids replacement_code")
        return self


class CounterfactualOutcome(ContractModel):
    root_status_before: GraphNodeStatus
    root_status_after: GraphNodeStatus
    support_delta: float | None = Field(default=None, ge=-1, le=1)
    refute_delta: float | None = Field(default=None, ge=-1, le=1)
    completeness_delta: float | None = Field(default=None, ge=-1, le=1)


class CounterfactualPair(BaseContract):
    allowed_scope_types = frozenset({ScopeType.MEMORY})
    scope_identity_fields = (("episode_code", "episode_code"),)

    contract_type: Literal["CounterfactualPair"]
    pair_code: CounterfactualCode
    episode_code: EpisodeCode
    pair_type: CounterfactualPairType
    factual_graph_ref: GraphSnapshotReference
    intervention: CounterfactualIntervention
    outcome: CounterfactualOutcome
    applicability_codes: tuple[StableCodeValue, ...] = Field(min_length=1)
    validation_refs: tuple[ValidationCode, ...] = Field(min_length=1)
    review_status: OfflineReviewStatus
    eligible_for_planning: bool
    lesson_codes: tuple[RuleCode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def planning_requires_approval(self) -> CounterfactualPair:
        if self.eligible_for_planning and self.review_status is not OfflineReviewStatus.APPROVED:
            raise ValueError("only an APPROVED counterfactual pair may affect planning")
        if self.scope.graph_code != self.factual_graph_ref.graph_code:
            raise ValueError("counterfactual scope graph must match factual graph")
        allowed_operation = {
            CounterfactualPairType.REMOVE_EVIDENCE: CounterfactualOperation.REMOVE,
            CounterfactualPairType.REVERSE_EVIDENCE: CounterfactualOperation.REVERSE,
            CounterfactualPairType.SELECT_ALTERNATIVE_CHECK: CounterfactualOperation.REPLACE,
            CounterfactualPairType.REPLACE_TOOL: CounterfactualOperation.REPLACE,
            CounterfactualPairType.FAILURE_RECOVERY: (
                CounterfactualOperation.RETRY_WITH_ALTERNATIVE
            ),
        }[self.pair_type]
        if self.intervention.operation is not allowed_operation:
            raise ValueError("counterfactual pair_type and intervention operation conflict")
        return self


class MessageEnvelope(BaseContract):
    allowed_scope_types = frozenset(
        {ScopeType.RUN, ScopeType.GRAPH, ScopeType.STEP, ScopeType.TOOL_EXTENSION}
    )
    scope_identity_fields = (("correlation_id", "run_id"),)

    contract_type: Literal["MessageEnvelope"]
    message_id: MessageCode
    correlation_id: RunCode
    causation_id: MessageCode | None = None
    actor: ActorType
    proposal_kind: ProposalKind
    payload_contract_type: Token
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def actor_owns_message(self) -> MessageEnvelope:
        validate_actor_proposal(self.actor, self.proposal_kind)
        if self.provenance.actor is not self.actor:
            raise ValueError("message actor must match provenance.actor")
        payload_type = self.payload.get("contract_type")
        if payload_type != self.payload_contract_type:
            raise ValueError("payload contract_type must match payload_contract_type")
        try:
            payload_model = CONTRACT_MODEL_BY_TYPE[self.payload_contract_type]
        except KeyError as exc:
            raise ValueError("payload_contract_type must name a registered contract") from exc
        parsed_payload = payload_model.model_validate(self.payload)
        for scope_field in (
            "case_code",
            "run_id",
            "graph_code",
            "step_id",
            "task_id",
            "episode_code",
        ):
            payload_value = getattr(parsed_payload.scope, scope_field)
            if payload_value is not None and getattr(self.scope, scope_field) != payload_value:
                raise ValueError(
                    f"message scope.{scope_field} must match payload scope.{scope_field}"
                )
        return self


CONTRACT_MODELS: tuple[type[BaseContract], ...] = (
    Artifact,
    CapabilityGap,
    CaseBundle,
    CounterfactualPair,
    Episode,
    ExpertAssessment,
    GECEGInstance,
    GECEGTemplate,
    GraphDelta,
    MessageEnvelope,
    PlannerAction,
    PlannerFeedback,
    PlannerObservation,
    PlannerState,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolOnboardingDecision,
    ToolOnboardingJob,
    ToolRequirementDocument,
    TrajectoryStep,
    ValidationResult,
)


CONTRACT_MODEL_BY_TYPE: dict[str, type[BaseContract]] = {
    model.model_fields["contract_type"].annotation.__args__[0]: model for model in CONTRACT_MODELS
}
