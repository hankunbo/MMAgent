"""Pure cross-field rules that 01B models must bind to their fields."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .enums import (
    ActorType,
    CapabilityAvailability,
    ClinicalEffect,
    EvidenceObservationStatus,
    EvidenceValidity,
    ExecutionPolicy,
    GraphNodeStatus,
    GraphUpdateSource,
    IntendedUse,
    ProposalKind,
    QCStatus,
    ToolCatalogStatus,
    ToolExecutionStatus,
    ToolValidationStatus,
)
from .identifiers import CodeNamespace, OpaqueEntityId


class SemanticConstraintError(ValueError):
    """Raised when independently owned states form an unsafe combination."""


@dataclass(frozen=True, slots=True)
class EvidenceSemantics:
    observation: EvidenceObservationStatus
    validity: EvidenceValidity
    clinical_effect: ClinicalEffect
    capability: CapabilityAvailability = CapabilityAvailability.AVAILABLE
    tool_execution: ToolExecutionStatus | None = None
    qc: QCStatus = QCStatus.UNASSESSED


_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ToolImplementationIdentity:
    """Immutable identity of one concrete Tool implementation version."""

    implementation_code: OpaqueEntityId
    version: str
    checksum: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_code, OpaqueEntityId):
            raise TypeError("implementation_code must be OpaqueEntityId")
        if self.implementation_code.namespace is not CodeNamespace.IMPLEMENTATION:
            raise SemanticConstraintError("implementation_code must use IMPLEMENTATION namespace")
        if not isinstance(self.version, str) or not _VERSION_RE.fullmatch(self.version):
            raise SemanticConstraintError("version must be a non-path canonical version token")
        if not isinstance(self.checksum, str) or not _SHA256_RE.fullmatch(self.checksum):
            raise SemanticConstraintError("checksum must be canonical sha256 with 64 lowercase hex")


_NON_SUCCEEDED_TOOL_STATES = frozenset(set(ToolExecutionStatus) - {ToolExecutionStatus.SUCCEEDED})

_OBSERVATIONS_WITH_CLINICAL_DIRECTION = frozenset(
    {EvidenceObservationStatus.POSITIVE, EvidenceObservationStatus.NEGATIVE}
)


def validate_evidence_semantics(value: EvidenceSemantics) -> EvidenceSemantics:
    """Reject technical failure/QC/gap states masquerading as clinical evidence."""

    if not isinstance(value, EvidenceSemantics):
        raise TypeError("value must be EvidenceSemantics")
    expected_types = (
        (value.observation, EvidenceObservationStatus, "observation"),
        (value.validity, EvidenceValidity, "validity"),
        (value.clinical_effect, ClinicalEffect, "clinical_effect"),
        (value.capability, CapabilityAvailability, "capability"),
        (value.qc, QCStatus, "qc"),
    )
    for field_value, field_type, field_name in expected_types:
        if not isinstance(field_value, field_type):
            raise TypeError(f"{field_name} must be {field_type.__name__}")
    if value.tool_execution is not None and not isinstance(
        value.tool_execution, ToolExecutionStatus
    ):
        raise TypeError("tool_execution must be ToolExecutionStatus or None")

    if value.tool_execution in _NON_SUCCEEDED_TOOL_STATES:
        if value.qc is QCStatus.PASS:
            raise SemanticConstraintError("non-SUCCEEDED tool execution cannot have QC PASS")
        if value.observation in _OBSERVATIONS_WITH_CLINICAL_DIRECTION:
            raise SemanticConstraintError("unfinished or failed tool execution has no observation")
        if value.validity is EvidenceValidity.VALID:
            raise SemanticConstraintError(
                "unfinished or failed tool execution cannot be VALID evidence"
            )
        if value.clinical_effect is not ClinicalEffect.NONE:
            raise SemanticConstraintError(
                "unfinished or failed tool execution has no clinical effect"
            )

    if (
        value.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY
        and value.observation is not EvidenceObservationStatus.UNOBSERVED_TOOL_GAP
    ):
        raise SemanticConstraintError(
            "UNAVAILABLE_CAPABILITY requires UNOBSERVED_TOOL_GAP observation"
        )
    if (
        value.observation is EvidenceObservationStatus.UNOBSERVED_TOOL_GAP
        and value.capability is not CapabilityAvailability.UNAVAILABLE_CAPABILITY
    ):
        raise SemanticConstraintError("UNOBSERVED_TOOL_GAP requires UNAVAILABLE_CAPABILITY")

    gap = (
        value.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY
        or value.observation is EvidenceObservationStatus.UNOBSERVED_TOOL_GAP
    )
    if gap:
        if value.observation is EvidenceObservationStatus.NEGATIVE:
            raise SemanticConstraintError("a tool gap cannot be a NEGATIVE observation")
        if value.validity is EvidenceValidity.VALID:
            raise SemanticConstraintError("a tool gap cannot be VALID evidence")
        if value.clinical_effect is not ClinicalEffect.NONE:
            raise SemanticConstraintError("a tool gap cannot support or refute a clinical claim")

    if value.qc is QCStatus.FAIL:
        if value.validity is not EvidenceValidity.INVALID:
            raise SemanticConstraintError("QC FAIL requires INVALID evidence validity")
        if value.clinical_effect is not ClinicalEffect.NONE:
            raise SemanticConstraintError("QC FAIL cannot have a clinical effect")

    if (
        value.tool_execution is ToolExecutionStatus.SUCCEEDED
        and value.validity is EvidenceValidity.VALID
        and value.qc is not QCStatus.PASS
    ):
        raise SemanticConstraintError("tool evidence requires QC PASS before it can be VALID")

    if (
        value.validity is not EvidenceValidity.VALID
        and value.clinical_effect is not ClinicalEffect.NONE
    ):
        raise SemanticConstraintError("only VALID evidence can have a clinical effect")
    if (
        value.clinical_effect is not ClinicalEffect.NONE
        and value.observation not in _OBSERVATIONS_WITH_CLINICAL_DIRECTION
    ):
        raise SemanticConstraintError("clinical effect requires a POSITIVE or NEGATIVE observation")

    return value


def validate_tool_use(
    catalog_status: ToolCatalogStatus,
    validation: ToolValidationStatus,
    intended_use: IntendedUse,
    policy: ExecutionPolicy,
    *,
    decisive_evidence: bool,
) -> None:
    """Enforce catalog, validation, and intended-use boundaries."""

    expected_types = (
        (catalog_status, ToolCatalogStatus, "catalog_status"),
        (validation, ToolValidationStatus, "validation"),
        (intended_use, IntendedUse, "intended_use"),
        (policy, ExecutionPolicy, "policy"),
    )
    for field_value, field_type, field_name in expected_types:
        if not isinstance(field_value, field_type):
            raise TypeError(f"{field_name} must be {field_type.__name__}")

    if policy is ExecutionPolicy.CLINICAL_DEPLOYMENT:
        raise SemanticConstraintError(
            "clinical_deployment is forbidden because no clinical approval status exists"
        )
    if catalog_status is not ToolCatalogStatus.ACTIVE:
        raise SemanticConstraintError(f"{catalog_status.value} tools cannot execute")
    if validation is ToolValidationStatus.UNVALIDATED:
        raise SemanticConstraintError("unvalidated tools cannot execute")
    if decisive_evidence and validation is not ToolValidationStatus.AUTO_VALIDATED_RESEARCH:
        raise SemanticConstraintError(
            "decisive research evidence requires auto_validated_research; "
            "smoke/benchmark alone is insufficient"
        )


_TERMINAL_CATALOG_STATUSES = frozenset(
    {
        ToolCatalogStatus.QUARANTINED,
        ToolCatalogStatus.FAILED,
        ToolCatalogStatus.DEPRECATED,
    }
)


def validate_new_implementation_after_terminal(
    previous_status: ToolCatalogStatus,
    previous_identity: ToolImplementationIdentity,
    new_status: ToolCatalogStatus,
    new_identity: ToolImplementationIdentity,
) -> ToolImplementationIdentity:
    """Require a new implementation record instead of reviving a terminal version."""

    if not isinstance(previous_status, ToolCatalogStatus) or not isinstance(
        new_status, ToolCatalogStatus
    ):
        raise TypeError("previous_status and new_status must be ToolCatalogStatus")
    if not isinstance(previous_identity, ToolImplementationIdentity) or not isinstance(
        new_identity, ToolImplementationIdentity
    ):
        raise TypeError("implementation identities must be ToolImplementationIdentity")
    if previous_status not in _TERMINAL_CATALOG_STATUSES:
        raise SemanticConstraintError("replacement is only defined for a terminal implementation")
    if new_status is not ToolCatalogStatus.CANDIDATE:
        raise SemanticConstraintError("a replacement implementation must start from candidate")

    unchanged_fields = tuple(
        field_name
        for field_name in ("implementation_code", "version", "checksum")
        if getattr(previous_identity, field_name) == getattr(new_identity, field_name)
    )
    if unchanged_fields:
        raise SemanticConstraintError(
            "replacement implementation requires new " + ", ".join(unchanged_fields)
        )
    return new_identity


_ALLOWED_PROPOSALS: dict[ActorType, frozenset[ProposalKind]] = {
    ActorType.PLANNER: frozenset({ProposalKind.PLAN_DECISION, ProposalKind.STOP_RECOMMENDATION}),
    ActorType.TRACE_AGENT: frozenset(
        {ProposalKind.ACTION_RECOMMENDATION, ProposalKind.STOP_RECOMMENDATION}
    ),
    ActorType.TOOL_AGENT: frozenset({ProposalKind.TOOL_EXECUTION_PROPOSAL}),
    ActorType.VERIFICATION_AGENT: frozenset(
        {
            ProposalKind.VALIDATION_RESULT,
            ProposalKind.GRAPH_DELTA_CANDIDATE,
            ProposalKind.STOP_RECOMMENDATION,
        }
    ),
    ActorType.EXPERT_AGENT: frozenset(
        {ProposalKind.EXPERT_REVIEW, ProposalKind.STOP_RECOMMENDATION}
    ),
    ActorType.RULE_ENGINE: frozenset({ProposalKind.GRAPH_DELTA_VALIDATION}),
    ActorType.GRAPH_ENGINE: frozenset(
        {ProposalKind.GRAPH_DELTA_VALIDATION, ProposalKind.GRAPH_DELTA_APPLICATION}
    ),
}


def allowed_proposals_for(actor: ActorType) -> frozenset[ProposalKind]:
    """Return the complete proposal allowlist for one actor."""

    if not isinstance(actor, ActorType):
        raise TypeError("actor must be ActorType")
    return _ALLOWED_PROPOSALS[actor]


def validate_actor_proposal(actor: ActorType, proposal: ProposalKind) -> None:
    """Enforce the complete Actor-to-Proposal allowlist."""

    if not isinstance(proposal, ProposalKind):
        raise TypeError("proposal must be ProposalKind")
    if proposal not in allowed_proposals_for(actor):
        raise SemanticConstraintError(f"{actor.value} cannot submit {proposal.value}")


def validate_graph_status_source(status: GraphNodeStatus, source: GraphUpdateSource) -> None:
    """Allow graph mutation only from validated evidence or deterministic rules."""

    if not isinstance(status, GraphNodeStatus) or not isinstance(source, GraphUpdateSource):
        raise TypeError("status and source must be canonical graph update enums")
    if source not in {
        GraphUpdateSource.VALIDATED_EVIDENCE,
        GraphUpdateSource.DETERMINISTIC_RULE,
    }:
        raise SemanticConstraintError(
            f"{source.value} cannot directly derive graph node status {status.value}"
        )
