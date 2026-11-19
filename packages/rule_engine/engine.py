"""Pure deterministic, disease-independent stage-03C propagation rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import ValidationError

from packages.contracts import (
    ActorType,
    CapabilityAvailability,
    CapabilityGap,
    ClinicalEffect,
    ConclusionStatus,
    DataAvailability,
    EdgeType,
    EvidenceObservationStatus,
    EvidenceValidity,
    ExpertAssessment,
    ExpertAssessmentDecision,
    ExpertAssessmentStatus,
    GECEGInstance,
    GraphDelta,
    GraphNodeStatus,
    GraphUpdateSource,
    PlanAction,
    ProposalKind,
    Provenance,
    QCStatus,
    StopReason,
    ToolExecutionStatus,
    UpdateNodeStatusOperation,
    ValidationResult,
    VersionedReference,
    can_transition,
)
from packages.graph_engine import (
    CLINICAL_EDGE_TYPES,
    DATA_FLOW_EDGE_TYPES,
    TRANSITION_RESET_MARKER,
    TRANSITION_RESET_OPERATION_PREFIX,
    TRANSITION_RESET_REASON,
    TRANSITION_STRATEGY_VERSION,
    AcceptedDeltaEvent,
    GraphEngineError,
    TransitionResetAudit,
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    canonical_checksum,
    transition_reset_metadata_from_delta,
    transition_reset_reference,
    validate_instance_structure,
)
from packages.graph_engine import (
    TransitionResetAuditEntry as TransitionResetAuditEntry,
)

RULE_ENGINE_VERSION = "03C.4.0"
RULE_PACKAGE_PAYLOAD_VERSION = "03C_RULE_PACKAGE_V2"
PROPAGATION_TRACE_VERSION = "03C_PROPAGATION_TRACE_V3"
BOUND_INPUT_PAYLOAD_VERSION = "03C_BOUND_INPUT_V1"
_TRANSITION_RESET_OPERATION_PREFIX = TRANSITION_RESET_OPERATION_PREFIX

GateOperator = Literal["AND", "OR"]
ReleaseStatus = Literal["PUBLISHED", "DRAFT"]
BlockedChannel = Literal[
    "UNOBSERVED_TOOL_GAP",
    "CAPABILITY_GAP",
    "TOOL_EXECUTION_FAILED",
    "TOOL_EXECUTION_TIMEOUT",
    "QC_FAILED",
    "DATA_MISSING",
    "MEDICAL_NEGATIVE",
    "EVIDENCE_REFUTATION",
    "NOT_APPLICABLE",
    "UNRESOLVED_EVIDENCE",
]


class RuleEngineErrorCode(StrEnum):
    PACKAGE_MISSING = "GRA03C_RULE_PACKAGE_MISSING"
    PACKAGE_NOT_PUBLISHED = "GRA03C_RULE_PACKAGE_NOT_PUBLISHED"
    PACKAGE_CHECKSUM_MISMATCH = "GRA03C_RULE_PACKAGE_CHECKSUM_MISMATCH"
    PACKAGE_VERSION_MISMATCH = "GRA03C_RULE_PACKAGE_VERSION_MISMATCH"
    PACKAGE_SCOPE_MISMATCH = "GRA03C_RULE_PACKAGE_SCOPE_MISMATCH"
    RULE_REFERENCE_INVALID = "GRA03C_RULE_REFERENCE_INVALID"
    GRAPH_CONTRACT_INVALID = "GRA03C_GRAPH_CONTRACT_INVALID"
    UNKNOWN_NODE = "GRA03C_UNKNOWN_NODE"
    DUPLICATE_EDGE = "GRA03C_DUPLICATE_EDGE"
    DUPLICATE_GATE = "GRA03C_DUPLICATE_GATE"
    GATE_MIXED_OPERATOR = "GRA03C_GATE_MIXED_OPERATOR"
    GATE_EDGE_MISMATCH = "GRA03C_GATE_EDGE_MISMATCH"
    DEPENDENCY_CYCLE = "GRA03C_DEPENDENCY_CYCLE"
    CONTEXT_INVALID = "GRA03C_CONTEXT_INVALID"
    SNAPSHOT_BINDING_MISMATCH = "GRA03C_SNAPSHOT_BINDING_MISMATCH"
    TRANSITION_BLOCKED = "GRA03C_TRANSITION_BLOCKED"
    INPUT_INVALID = "GRA03C_INPUT_INVALID"
    EXPERT_ACTION_INVALID = "GRA03C_EXPERT_ACTION_INVALID"
    TRANSITION_AUDIT_INVALID = "GRA03C_TRANSITION_AUDIT_INVALID"


class RuleEngineError(ValueError):
    def __init__(self, code: RuleEngineErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


@dataclass(frozen=True, slots=True)
class GateDefinition:
    gate_code: str
    parent_node_code: str
    operator: GateOperator
    child_node_codes: tuple[str, ...]
    rule_ref: VersionedReference


@dataclass(frozen=True, slots=True)
class RulePackage:
    payload_version: str
    package_code: str
    package_version: str
    package_checksum: str
    release_status: ReleaseStatus
    graph_code: str
    template_ref: VersionedReference
    rule_refs: tuple[VersionedReference, ...]
    gates: tuple[GateDefinition, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ValidatedRulePackage:
    package: RulePackage
    canonical_checksum: str
    snapshot_binding: SnapshotBinding
    validation_provenance: Provenance


@dataclass(frozen=True, slots=True)
class PropagationBudget:
    max_rounds: int
    round_index: int
    remaining_actions: int
    baseline_only: bool


@dataclass(frozen=True, slots=True)
class PropagationHistory:
    prior_graph_checksums: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SnapshotBinding:
    graph_code: str
    graph_version: int
    graph_checksum: str


@dataclass(frozen=True, slots=True)
class BoundValidation:
    payload_version: str
    item: ValidationResult
    source_snapshot: SnapshotBinding
    item_checksum: str
    binding_checksum: str


@dataclass(frozen=True, slots=True)
class BoundCapabilityGap:
    payload_version: str
    item: CapabilityGap
    source_snapshot: SnapshotBinding
    item_checksum: str
    binding_checksum: str


@dataclass(frozen=True, slots=True)
class PropagationContext:
    expected_package_version: str
    expected_package_checksum: str
    snapshot_binding: SnapshotBinding
    validations: tuple[BoundValidation, ...] = ()
    capability_gaps: tuple[BoundCapabilityGap, ...] = ()
    budget: PropagationBudget = PropagationBudget(1, 0, 1, True)
    history: PropagationHistory = PropagationHistory()


@dataclass(frozen=True, slots=True)
class KnowledgeGap:
    code: str
    message: str
    rule_refs: tuple[VersionedReference, ...]


@dataclass(frozen=True, slots=True)
class BlockedEvidence:
    node_code: str
    channel: BlockedChannel
    reason_code: str
    required: bool


@dataclass(frozen=True, slots=True)
class EvidenceComponent:
    contributing_nodes: tuple[str, ...]
    rule_refs: tuple[VersionedReference, ...]
    required_count: int
    resolved_count: int
    blocked_count: int
    explanations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NodePropagation:
    node_code: str
    current_status: GraphNodeStatus
    derived_status: GraphNodeStatus | None
    supporting_nodes: tuple[str, ...]
    refuting_nodes: tuple[str, ...]
    excluding_nodes: tuple[str, ...]
    conflicting_nodes: tuple[str, ...]
    unresolved_nodes: tuple[str, ...]
    blocked_nodes: tuple[str, ...]
    rule_refs: tuple[VersionedReference, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class TransitionBlocked:
    node_code: str
    current_status: GraphNodeStatus
    derived_status: GraphNodeStatus
    next_status: GraphNodeStatus | None
    reason_code: str
    strategy_version: str


@dataclass(frozen=True, slots=True)
class LegalAction:
    action: PlanAction
    target_node_code: str | None
    graph_version: int
    rule_package_checksum: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class StopSuggestion:
    reason: StopReason
    conclusion_status: ConclusionStatus
    reason_codes: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class PropagationTrace:
    payload_version: str
    engine_version: str
    graph_checksum: str
    rule_package_checksum: str
    node_results: tuple[NodePropagation, ...]
    blocked_evidence: tuple[BlockedEvidence, ...]
    transition_blocks: tuple[TransitionBlocked, ...]
    data_flow_edges: tuple[tuple[str, str, EdgeType], ...]
    trace_checksum: str


@dataclass(frozen=True, slots=True)
class PropagationEvaluation:
    support: EvidenceComponent
    refutation: EvidenceComponent
    completeness: EvidenceComponent
    conflict: EvidenceComponent
    node_results: tuple[NodePropagation, ...]
    blocked_evidence: tuple[BlockedEvidence, ...]
    transition_blocks: tuple[TransitionBlocked, ...]
    data_flow_edges: tuple[tuple[str, str, EdgeType], ...]


@dataclass(frozen=True, slots=True)
class PropagationResult:
    evaluation: PropagationEvaluation
    candidate_delta: GraphDelta | None
    legal_actions: tuple[LegalAction, ...]
    stop_suggestion: StopSuggestion | None
    knowledge_gaps: tuple[KnowledgeGap, ...]
    trace: PropagationTrace


def _ref_key(reference: VersionedReference) -> tuple[str, str, str]:
    return (str(reference.code), reference.version, reference.checksum or "")


def _normalized_refs(references: tuple[VersionedReference, ...]) -> tuple[VersionedReference, ...]:
    if type(references) is not tuple or any(
        type(reference) is not VersionedReference for reference in references
    ):
        _raise(RuleEngineErrorCode.INPUT_INVALID, "rule references must be a tuple of contracts")
    by_key = {_ref_key(reference): reference for reference in references}
    return tuple(by_key[key] for key in sorted(by_key))


def _normalized_gate(gate: GateDefinition) -> GateDefinition:
    if type(gate) is not GateDefinition or type(gate.child_node_codes) is not tuple:
        _raise(RuleEngineErrorCode.INPUT_INVALID, "gate is not a canonical GateDefinition")
    if any(type(code) is not str for code in gate.child_node_codes):
        _raise(RuleEngineErrorCode.INPUT_INVALID, "gate child codes must be strings")
    if type(gate.rule_ref) is not VersionedReference:
        _raise(RuleEngineErrorCode.INPUT_INVALID, "gate rule_ref must be a contract reference")
    return replace(gate, child_node_codes=tuple(sorted(gate.child_node_codes)))


def _normalized_provenance(provenance: Provenance) -> Provenance:
    try:
        parsed = Provenance.model_validate(provenance.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.INPUT_INVALID,
            "rule package provenance is not a canonical contract",
        ) from error
    return parsed.model_copy(
        update={
            "source_codes": tuple(sorted(set(parsed.source_codes))),
            "rule_versions": _normalized_refs(parsed.rule_versions),
            "knowledge_versions": _normalized_refs(parsed.knowledge_versions),
            "tool_versions": _normalized_refs(parsed.tool_versions),
        }
    )


def rule_package_checksum_payload(package: RulePackage) -> dict[str, object]:
    if type(package) is not RulePackage:
        _raise(RuleEngineErrorCode.PACKAGE_MISSING, "rule package has the wrong type")
    if type(package.rule_refs) is not tuple or type(package.gates) is not tuple:
        _raise(RuleEngineErrorCode.INPUT_INVALID, "rule package set fields must be tuples")
    gates = tuple(
        sorted(
            (_normalized_gate(gate) for gate in package.gates),
            key=lambda item: (item.parent_node_code, item.operator, item.gate_code),
        )
    )
    return {
        "payload_version": package.payload_version,
        "package_code": package.package_code,
        "package_version": package.package_version,
        "release_status": package.release_status,
        "graph_code": package.graph_code,
        "template_ref": package.template_ref,
        "rule_refs": _normalized_refs(package.rule_refs),
        "gates": gates,
        "provenance": _normalized_provenance(package.provenance),
    }


def compute_rule_package_checksum(package: RulePackage) -> str:
    try:
        return canonical_checksum(rule_package_checksum_payload(package))
    except RuleEngineError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.INPUT_INVALID,
            "rule package cannot be encoded as strict canonical JSON",
        ) from error


def snapshot_binding_for(graph: GECEGInstance) -> SnapshotBinding:
    """Bind all read-only propagation context to one exact immutable snapshot."""

    _validate_graph(graph)
    return SnapshotBinding(
        graph_code=str(graph.graph_code),
        graph_version=graph.graph_version,
        graph_checksum=canonical_checksum(graph),
    )


def _bound_input_payload(
    *,
    kind: str,
    item: ValidationResult | CapabilityGap,
    source_snapshot: SnapshotBinding,
    item_checksum: str,
) -> dict[str, object]:
    return {
        "payload_version": BOUND_INPUT_PAYLOAD_VERSION,
        "kind": kind,
        "source_snapshot": source_snapshot,
        "item_checksum": item_checksum,
        "item": item,
    }


def _bind_item_provenance(
    item: ValidationResult | CapabilityGap, binding: SnapshotBinding
) -> ValidationResult | CapabilityGap:
    input_checksums = tuple(sorted(set((*item.provenance.input_checksums, binding.graph_checksum))))
    return item.model_copy(
        update={
            "provenance": item.provenance.model_copy(update={"input_checksums": input_checksums})
        }
    )


def bind_validation(graph: GECEGInstance, item: ValidationResult) -> BoundValidation:
    """Bind one validation to the exact graph snapshot that produced it."""

    binding = snapshot_binding_for(graph)
    try:
        if type(item) is not ValidationResult:
            raise TypeError("validation has the wrong type")
        parsed = ValidationResult.model_validate(item.model_dump(mode="python"))
        if (
            parsed.scope.case_code != graph.case_code
            or parsed.scope.run_id != graph.scope.run_id
            or parsed.scope.graph_code != graph.graph_code
        ):
            raise ValueError("validation scope differs from source graph")
        parsed = ValidationResult.model_validate(
            _bind_item_provenance(parsed, binding).model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise RuleEngineError(RuleEngineErrorCode.CONTEXT_INVALID, "invalid validation") from error
    item_checksum = canonical_checksum(parsed)
    payload = _bound_input_payload(
        kind="VALIDATION_RESULT",
        item=parsed,
        source_snapshot=binding,
        item_checksum=item_checksum,
    )
    return BoundValidation(
        BOUND_INPUT_PAYLOAD_VERSION,
        parsed,
        binding,
        item_checksum,
        canonical_checksum(payload),
    )


def bind_capability_gap(graph: GECEGInstance, item: CapabilityGap) -> BoundCapabilityGap:
    """Bind one capability gap to the exact graph snapshot that produced it."""

    binding = snapshot_binding_for(graph)
    try:
        if type(item) is not CapabilityGap:
            raise TypeError("capability gap has the wrong type")
        parsed = CapabilityGap.model_validate(item.model_dump(mode="python"))
        if (
            parsed.scope.case_code != graph.case_code
            or parsed.scope.run_id != graph.scope.run_id
            or parsed.scope.graph_code != graph.graph_code
            or parsed.origin_run_id != graph.scope.run_id
        ):
            raise ValueError("capability gap scope differs from source graph")
        parsed = CapabilityGap.model_validate(
            _bind_item_provenance(parsed, binding).model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.CONTEXT_INVALID, "invalid capability gap"
        ) from error
    item_checksum = canonical_checksum(parsed)
    payload = _bound_input_payload(
        kind="CAPABILITY_GAP",
        item=parsed,
        source_snapshot=binding,
        item_checksum=item_checksum,
    )
    return BoundCapabilityGap(
        BOUND_INPUT_PAYLOAD_VERSION,
        parsed,
        binding,
        item_checksum,
        canonical_checksum(payload),
    )


def _raise(code: RuleEngineErrorCode, message: str) -> None:
    raise RuleEngineError(code, message)


def _validate_graph(graph: GECEGInstance) -> dict[str, object]:
    try:
        parsed = validate_instance_structure(graph)
    except (AttributeError, GraphEngineError, ValidationError, TypeError, ValueError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.GRAPH_CONTRACT_INVALID,
            "graph violates the public graph-engine structure boundary",
        ) from error
    return {str(node.node_code): node for node in parsed.nodes}


def _check_cycles(graph: GECEGInstance) -> None:
    adjacency: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.edge_type in CLINICAL_EDGE_TYPES:
            adjacency.setdefault(str(edge.source_node_code), set()).add(str(edge.target_node_code))
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            _raise(RuleEngineErrorCode.DEPENDENCY_CYCLE, f"clinical dependency cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for target in sorted(adjacency.get(node, ())):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(adjacency):
        visit(node)


def validate_rule_package(
    graph: GECEGInstance,
    package: RulePackage,
    *,
    expected_version: str,
    expected_checksum: str,
) -> ValidatedRulePackage:
    nodes = _validate_graph(graph)
    if type(package) is not RulePackage:
        _raise(RuleEngineErrorCode.PACKAGE_MISSING, "rule package is missing")
    scalar_fields = (
        package.payload_version,
        package.package_code,
        package.package_version,
        package.package_checksum,
        package.release_status,
        package.graph_code,
    )
    if any(type(value) is not str for value in scalar_fields):
        _raise(RuleEngineErrorCode.INPUT_INVALID, "rule package scalar fields must be strings")
    if type(expected_version) is not str or type(expected_checksum) is not str:
        _raise(RuleEngineErrorCode.INPUT_INVALID, "expected rule identity must be strings")
    if package.payload_version != RULE_PACKAGE_PAYLOAD_VERSION:
        _raise(RuleEngineErrorCode.PACKAGE_VERSION_MISMATCH, "payload version mismatch")
    if package.release_status != "PUBLISHED":
        _raise(RuleEngineErrorCode.PACKAGE_NOT_PUBLISHED, "only PUBLISHED rules may run")
    if package.package_version != expected_version:
        _raise(RuleEngineErrorCode.PACKAGE_VERSION_MISMATCH, "package version mismatch")
    checksum = compute_rule_package_checksum(package)
    if package.package_checksum != checksum or expected_checksum != checksum:
        _raise(RuleEngineErrorCode.PACKAGE_CHECKSUM_MISMATCH, "package checksum mismatch")
    if package.graph_code != str(graph.graph_code) or package.template_ref != graph.template_ref:
        _raise(RuleEngineErrorCode.PACKAGE_SCOPE_MISMATCH, "package graph/template scope mismatch")
    normalized_provenance = _normalized_provenance(package.provenance)
    if normalized_provenance.actor is not ActorType.RULE_ENGINE:
        _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, "package provenance actor mismatch")
    refs = _normalized_refs(package.rule_refs)
    if not refs or len({_ref_key(ref) for ref in refs}) != len(refs):
        _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, "rule refs are missing or duplicated")
    if any(ref.checksum is None for ref in refs):
        _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, "every rule ref requires checksum")
    ref_keys = {_ref_key(ref) for ref in refs}
    if not ref_keys.issubset({_ref_key(ref) for ref in normalized_provenance.rule_versions}):
        _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, "rule refs lack provenance binding")

    gates = tuple(
        sorted(
            (_normalized_gate(gate) for gate in package.gates),
            key=lambda gate: (gate.parent_node_code, gate.operator, gate.gate_code),
        )
    )
    parents: set[str] = set()
    membership: dict[tuple[str, str], GateOperator] = {}
    edge_keys = {
        (str(edge.source_node_code), str(edge.target_node_code), edge.edge_type)
        for edge in graph.edges
    }
    pair_operators: dict[tuple[str, str], set[GateOperator]] = {}
    for gate in gates:
        for child in gate.child_node_codes:
            pair_operators.setdefault((child, gate.parent_node_code), set()).add(gate.operator)
    mixed_pairs = tuple(
        sorted(pair for pair, operators in pair_operators.items() if len(operators) > 1)
    )
    if mixed_pairs:
        _raise(RuleEngineErrorCode.GATE_MIXED_OPERATOR, f"mixed gates for {mixed_pairs}")

    for gate in gates:
        if gate.parent_node_code in parents:
            _raise(RuleEngineErrorCode.DUPLICATE_GATE, gate.parent_node_code)
        parents.add(gate.parent_node_code)
        if gate.operator not in {"AND", "OR"} or not gate.child_node_codes:
            _raise(RuleEngineErrorCode.GATE_EDGE_MISMATCH, gate.gate_code)
        if len(set(gate.child_node_codes)) != len(gate.child_node_codes):
            _raise(RuleEngineErrorCode.DUPLICATE_GATE, gate.gate_code)
        if gate.parent_node_code not in nodes or any(
            child not in nodes for child in gate.child_node_codes
        ):
            _raise(RuleEngineErrorCode.UNKNOWN_NODE, gate.gate_code)
        if _ref_key(gate.rule_ref) not in ref_keys:
            _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, gate.gate_code)
        edge_type = EdgeType.AND_CHILD if gate.operator == "AND" else EdgeType.OR_CHILD
        for child in gate.child_node_codes:
            pair = (child, gate.parent_node_code)
            prior = membership.get(pair)
            if prior is not None and prior != gate.operator:
                _raise(RuleEngineErrorCode.GATE_MIXED_OPERATOR, f"mixed gate for {pair}")
            membership[pair] = gate.operator
            if (child, gate.parent_node_code, edge_type) not in edge_keys:
                _raise(RuleEngineErrorCode.GATE_EDGE_MISMATCH, f"missing {edge_type} for {pair}")
            gate_edge = next(
                item
                for item in graph.edges
                if str(item.source_node_code) == child
                and str(item.target_node_code) == gate.parent_node_code
                and item.edge_type is edge_type
            )
            if gate_edge.rule_ref != gate.rule_ref:
                _raise(
                    RuleEngineErrorCode.RULE_REFERENCE_INVALID,
                    f"gate edge {child}->{gate.parent_node_code}",
                )
    for source, target, edge_type in edge_keys:
        if edge_type in {EdgeType.AND_CHILD, EdgeType.OR_CHILD}:
            expected = "AND" if edge_type is EdgeType.AND_CHILD else "OR"
            if membership.get((source, target)) != expected:
                _raise(RuleEngineErrorCode.GATE_EDGE_MISMATCH, "unregistered gate edge")
        elif edge_type in CLINICAL_EDGE_TYPES:
            edge = next(
                item
                for item in graph.edges
                if str(item.source_node_code) == source
                and str(item.target_node_code) == target
                and item.edge_type is edge_type
            )
            if edge_type in {EdgeType.CAUSES, EdgeType.RISK_MODIFIES}:
                _raise(RuleEngineErrorCode.GATE_EDGE_MISMATCH, f"no 03C handler for {edge_type}")
            if edge.rule_ref is None or _ref_key(edge.rule_ref) not in ref_keys:
                _raise(RuleEngineErrorCode.RULE_REFERENCE_INVALID, f"edge {source}->{target}")
        elif edge_type not in DATA_FLOW_EDGE_TYPES:
            _raise(RuleEngineErrorCode.GATE_EDGE_MISMATCH, f"unknown edge class {edge_type}")
    _check_cycles(graph)
    normalized = replace(
        package,
        rule_refs=refs,
        gates=gates,
        provenance=normalized_provenance,
    )
    binding = snapshot_binding_for(graph)
    validation_provenance = normalized.provenance.model_copy(
        update={
            "generator_version": RULE_ENGINE_VERSION,
            "input_checksums": (binding.graph_checksum, checksum),
        }
    )
    return ValidatedRulePackage(normalized, checksum, binding, validation_provenance)


def _require_snapshot_binding(binding: object, expected: SnapshotBinding) -> None:
    if (
        type(binding) is not SnapshotBinding
        or type(binding.graph_code) is not str
        or type(binding.graph_version) is not int
        or type(binding.graph_checksum) is not str
        or binding != expected
    ):
        _raise(
            RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH,
            "snapshot binding is non-canonical or targets a different graph snapshot",
        )


def _validate_context(
    graph: GECEGInstance, validated: ValidatedRulePackage, context: PropagationContext
) -> tuple[tuple[ValidationResult, ...], tuple[CapabilityGap, ...]]:
    if type(context) is not PropagationContext:
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "context has the wrong type")
    expected_binding = snapshot_binding_for(graph)
    _require_snapshot_binding(context.snapshot_binding, expected_binding)
    budget = context.budget
    if (
        type(budget) is not PropagationBudget
        or type(budget.max_rounds) is not int
        or type(budget.round_index) is not int
        or type(budget.remaining_actions) is not int
        or budget.max_rounds < 1
        or budget.round_index < 0
        or budget.remaining_actions < 0
        or type(budget.baseline_only) is not bool
    ):
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "invalid explicit propagation budget")
    if (
        type(context.history) is not PropagationHistory
        or type(context.history.prior_graph_checksums) is not tuple
    ):
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "invalid propagation history")
    if any(
        type(checksum) is not str or not checksum.startswith("sha256:")
        for checksum in context.history.prior_graph_checksums
    ):
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "history checksums are not canonical")
    if type(context.validations) is not tuple or type(context.capability_gaps) is not tuple:
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "context evidence fields must be tuples")
    nodes = {str(node.node_code) for node in graph.nodes}
    validations: dict[str, ValidationResult] = {}
    for bound in context.validations:
        try:
            if type(bound) is not BoundValidation:
                raise TypeError("validation lacks an immutable source binding")
            if bound.payload_version != BOUND_INPUT_PAYLOAD_VERSION:
                raise ValueError("validation binding payload version mismatch")
            _require_snapshot_binding(bound.source_snapshot, expected_binding)
            if type(bound.item) is not ValidationResult:
                raise TypeError("bound validation item has the wrong type")
            parsed = ValidationResult.model_validate(bound.item.model_dump(mode="python"))
            expected_item_checksum = canonical_checksum(parsed)
            expected_binding_checksum = canonical_checksum(
                _bound_input_payload(
                    kind="VALIDATION_RESULT",
                    item=parsed,
                    source_snapshot=bound.source_snapshot,
                    item_checksum=expected_item_checksum,
                )
            )
            if (
                parsed != bound.item
                or bound.item_checksum != expected_item_checksum
                or bound.binding_checksum != expected_binding_checksum
                or expected_binding.graph_checksum not in parsed.provenance.input_checksums
            ):
                raise ValueError("validation binding identity mismatch")
        except RuleEngineError:
            raise
        except (ValidationError, TypeError, ValueError) as error:
            raise RuleEngineError(
                RuleEngineErrorCode.CONTEXT_INVALID, "invalid validation"
            ) from error
        if str(parsed.target_node_code) not in nodes:
            _raise(RuleEngineErrorCode.UNKNOWN_NODE, str(parsed.target_node_code))
        if (
            parsed.scope.case_code != graph.case_code
            or parsed.scope.run_id != graph.scope.run_id
            or parsed.scope.graph_code != graph.graph_code
        ):
            _raise(RuleEngineErrorCode.CONTEXT_INVALID, "validation snapshot scope mismatch")
        validations[canonical_checksum(parsed)] = parsed
    gaps: dict[str, CapabilityGap] = {}
    for bound in context.capability_gaps:
        try:
            if type(bound) is not BoundCapabilityGap:
                raise TypeError("capability gap lacks an immutable source binding")
            if bound.payload_version != BOUND_INPUT_PAYLOAD_VERSION:
                raise ValueError("capability binding payload version mismatch")
            _require_snapshot_binding(bound.source_snapshot, expected_binding)
            if type(bound.item) is not CapabilityGap:
                raise TypeError("bound capability item has the wrong type")
            parsed_gap = CapabilityGap.model_validate(bound.item.model_dump(mode="python"))
            expected_item_checksum = canonical_checksum(parsed_gap)
            expected_binding_checksum = canonical_checksum(
                _bound_input_payload(
                    kind="CAPABILITY_GAP",
                    item=parsed_gap,
                    source_snapshot=bound.source_snapshot,
                    item_checksum=expected_item_checksum,
                )
            )
            if (
                parsed_gap != bound.item
                or bound.item_checksum != expected_item_checksum
                or bound.binding_checksum != expected_binding_checksum
                or expected_binding.graph_checksum not in parsed_gap.provenance.input_checksums
            ):
                raise ValueError("capability binding identity mismatch")
        except RuleEngineError:
            raise
        except (ValidationError, TypeError, ValueError) as error:
            raise RuleEngineError(
                RuleEngineErrorCode.CONTEXT_INVALID, "invalid capability gap"
            ) from error
        if str(parsed_gap.target_node_code) not in nodes:
            _raise(RuleEngineErrorCode.UNKNOWN_NODE, str(parsed_gap.target_node_code))
        if (
            parsed_gap.scope.case_code != graph.case_code
            or parsed_gap.scope.run_id != graph.scope.run_id
            or parsed_gap.scope.graph_code != graph.graph_code
            or parsed_gap.origin_run_id != graph.scope.run_id
        ):
            _raise(RuleEngineErrorCode.CONTEXT_INVALID, "capability gap snapshot scope mismatch")
        gaps[canonical_checksum(parsed_gap)] = parsed_gap
    return (
        tuple(validations[key] for key in sorted(validations)),
        tuple(gaps[key] for key in sorted(gaps)),
    )


def _blocked_evidence(
    graph: GECEGInstance,
    validations: tuple[ValidationResult, ...],
    gaps: tuple[CapabilityGap, ...],
    required_nodes: set[str],
) -> tuple[BlockedEvidence, ...]:
    blocked: dict[tuple[str, str], BlockedEvidence] = {}

    def add(node: str, channel: BlockedChannel, reason: str) -> None:
        blocked[(node, channel)] = BlockedEvidence(node, channel, reason, node in required_nodes)

    for gap in gaps:
        add(str(gap.target_node_code), "CAPABILITY_GAP", "RULE:CAPABILITY_GAP")
    for validation in validations:
        node = str(validation.target_node_code)
        if validation.observation is EvidenceObservationStatus.UNOBSERVED_TOOL_GAP:
            add(node, "UNOBSERVED_TOOL_GAP", "RULE:UNOBSERVED_TOOL_GAP")
        if validation.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY:
            add(node, "CAPABILITY_GAP", "RULE:CAPABILITY_GAP")
        if validation.tool_execution_status is ToolExecutionStatus.FAILED:
            add(node, "TOOL_EXECUTION_FAILED", "RULE:TOOL_EXECUTION_FAILED")
        if validation.tool_execution_status is ToolExecutionStatus.TIMED_OUT:
            add(node, "TOOL_EXECUTION_TIMEOUT", "RULE:TOOL_EXECUTION_TIMEOUT")
        if validation.qc_status is QCStatus.FAIL:
            add(node, "QC_FAILED", "RULE:QC_FAILED")
        technically_blocked = (
            validation.observation is EvidenceObservationStatus.UNOBSERVED_TOOL_GAP
            or validation.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY
            or validation.tool_execution_status
            in {ToolExecutionStatus.FAILED, ToolExecutionStatus.TIMED_OUT}
            or validation.qc_status is QCStatus.FAIL
        )
        if not technically_blocked and (
            validation.observation is EvidenceObservationStatus.NEGATIVE
            and validation.validity is EvidenceValidity.VALID
            and validation.clinical_effect is ClinicalEffect.REFUTES
        ):
            add(node, "EVIDENCE_REFUTATION", "RULE:VALID_EVIDENCE_REFUTATION")
        elif (
            not technically_blocked and validation.observation is EvidenceObservationStatus.NEGATIVE
        ):
            add(node, "MEDICAL_NEGATIVE", "RULE:MEDICAL_NEGATIVE")
        elif not technically_blocked and validation.observation in {
            EvidenceObservationStatus.UNOBSERVED,
            EvidenceObservationStatus.INCONCLUSIVE,
        }:
            add(node, "DATA_MISSING", "RULE:DATA_MISSING")
    for node in graph.nodes:
        code = str(node.node_code)
        if code == str(graph.goal_node_code):
            continue
        if node.availability is DataAvailability.NOT_APPLICABLE:
            add(code, "NOT_APPLICABLE", "RULE:NOT_APPLICABLE")
        elif node.availability is DataAvailability.UNAVAILABLE:
            add(code, "DATA_MISSING", "RULE:DATA_MISSING")
        elif node.status in {GraphNodeStatus.PENDING, GraphNodeStatus.UNKNOWN}:
            if not any(item.node_code == code for item in blocked.values()):
                add(code, "UNRESOLVED_EVIDENCE", "RULE:UNRESOLVED_EVIDENCE")
    return tuple(sorted(blocked.values(), key=lambda item: (item.node_code, item.channel)))


_TECHNICAL_BLOCKING_CHANNELS = frozenset(
    {
        "UNOBSERVED_TOOL_GAP",
        "CAPABILITY_GAP",
        "TOOL_EXECUTION_FAILED",
        "TOOL_EXECUTION_TIMEOUT",
        "QC_FAILED",
        "DATA_MISSING",
        "NOT_APPLICABLE",
    }
)


def _trusted_status(node: object, blocked: tuple[BlockedEvidence, ...]) -> GraphNodeStatus | None:
    """Single authority for every clinical contribution path."""

    status = getattr(node, "status", None)
    availability = getattr(node, "availability", None)
    if availability is not DataAvailability.AVAILABLE:
        return None
    if any(item.channel in _TECHNICAL_BLOCKING_CHANNELS for item in blocked):
        return None
    if status in {GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED}:
        return status
    return None


def _clinical_goal_path_nodes(graph: GECEGInstance) -> set[str]:
    incoming: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.edge_type in CLINICAL_EDGE_TYPES:
            incoming.setdefault(str(edge.target_node_code), set()).add(str(edge.source_node_code))
    relevant = {str(graph.goal_node_code)}
    pending = [str(graph.goal_node_code)]
    while pending:
        target = pending.pop()
        for source in sorted(incoming.get(target, ())):
            if source not in relevant:
                relevant.add(source)
                pending.append(source)
    return relevant


def _transition_block(result: NodePropagation) -> TransitionBlocked | None:
    desired = result.derived_status
    if desired is None or desired is result.current_status:
        return None
    if can_transition(result.current_status, desired, allow_idempotent=False):
        return None
    next_status = (
        GraphNodeStatus.PENDING
        if can_transition(
            result.current_status,
            GraphNodeStatus.PENDING,
            allow_idempotent=False,
        )
        else None
    )
    return TransitionBlocked(
        node_code=result.node_code,
        current_status=result.current_status,
        derived_status=desired,
        next_status=next_status,
        reason_code="RULE:TRANSITION_RESET_REQUIRED",
        strategy_version=TRANSITION_STRATEGY_VERSION,
    )


def _component(
    contributions: set[str],
    refs: set[VersionedReference],
    *,
    required: int,
    resolved: int,
    blocked: int,
    explanations: set[str],
) -> EvidenceComponent:
    return EvidenceComponent(
        tuple(sorted(contributions)),
        tuple(sorted(refs, key=_ref_key)),
        required,
        resolved,
        blocked,
        tuple(sorted(explanations)),
    )


def evaluate_propagation(
    graph: GECEGInstance,
    validated: ValidatedRulePackage,
    context: PropagationContext,
) -> PropagationEvaluation:
    if type(context) is not PropagationContext:
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "context has the wrong type")
    if type(validated) is not ValidatedRulePackage:
        _raise(RuleEngineErrorCode.CONTEXT_INVALID, "validated package has the wrong type")
    _require_snapshot_binding(validated.snapshot_binding, snapshot_binding_for(graph))
    try:
        rebuilt_validated = validate_rule_package(
            graph,
            validated.package,
            expected_version=context.expected_package_version,
            expected_checksum=context.expected_package_checksum,
        )
    except RuleEngineError as error:
        raise RuleEngineError(
            RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH,
            "validated rule package cannot be rebuilt against the current snapshot",
        ) from error
    if rebuilt_validated != validated:
        _raise(
            RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH,
            "validated rule package differs from current-snapshot validation",
        )
    nodes = _validate_graph(graph)
    validations, gaps = _validate_context(graph, validated, context)
    gates = {gate.parent_node_code: gate for gate in validated.package.gates}
    clinical_path_nodes = _clinical_goal_path_nodes(graph)
    required_nodes = clinical_path_nodes - {str(graph.goal_node_code)}
    blocked = _blocked_evidence(graph, validations, gaps, required_nodes)
    blocked_by_node: dict[str, tuple[BlockedEvidence, ...]] = {
        code: tuple(item for item in blocked if item.node_code == code) for code in nodes
    }
    relation_edges = tuple(
        edge
        for edge in graph.edges
        if edge.edge_type
        in {
            EdgeType.SUPPORTS,
            EdgeType.REFUTES,
            EdgeType.EXCLUDES,
            EdgeType.FALSIFIES,
            EdgeType.CONFLICTS_WITH,
        }
    )
    targets = sorted(set(gates) | {str(edge.target_node_code) for edge in relation_edges})
    node_results: list[NodePropagation] = []
    all_support: set[str] = set()
    all_refute: set[str] = set()
    all_conflict: set[str] = set()
    all_refs: set[VersionedReference] = set()
    support_explanations: set[str] = set()
    refute_explanations: set[str] = set()
    conflict_explanations: set[str] = set()

    for target in targets:
        support: set[str] = set()
        refute: set[str] = set()
        exclude: set[str] = set()
        conflicts: set[str] = set()
        unresolved: set[str] = set()
        target_refs: set[VersionedReference] = set()
        gate_desired: GraphNodeStatus | None = None
        gate = gates.get(target)
        if gate is not None:
            children = [nodes[code] for code in gate.child_node_codes]
            supported = {
                str(child.node_code)
                for child in children
                if _trusted_status(child, blocked_by_node[str(child.node_code)])
                is GraphNodeStatus.SUPPORTED
            }
            refuted = {
                str(child.node_code)
                for child in children
                if _trusted_status(child, blocked_by_node[str(child.node_code)])
                is GraphNodeStatus.REFUTED
            }
            unresolved.update(
                str(child.node_code)
                for child in children
                if _trusted_status(child, blocked_by_node[str(child.node_code)]) is None
            )
            if gate.operator == "AND":
                if len(supported) == len(children):
                    support.update(supported)
                    gate_desired = GraphNodeStatus.SUPPORTED
                    support_explanations.add(
                        f"AND {gate.gate_code} all required children supported"
                    )
                elif refuted:
                    refute.update(refuted)
                    gate_desired = GraphNodeStatus.REFUTED
                    refute_explanations.add(f"AND {gate.gate_code} has a refuted necessary child")
            else:
                if supported:
                    support.update(supported)
                    refute.update(refuted)
                    gate_desired = GraphNodeStatus.SUPPORTED
                    support_explanations.add(f"OR {gate.gate_code} has a supported branch")
                    if refuted:
                        refute_explanations.add(f"OR {gate.gate_code} also has a refuted branch")
                elif len(refuted) == len(children):
                    refute.update(refuted)
                    gate_desired = GraphNodeStatus.REFUTED
                    refute_explanations.add(f"OR {gate.gate_code} has no non-refuted branch")
            target_refs.add(gate.rule_ref)

        explicit_support: set[str] = set()
        explicit_refute: set[str] = set()
        explicit_exclude: set[str] = set()
        explicit_conflict: set[str] = set()
        for edge in relation_edges:
            if str(edge.target_node_code) != target:
                continue
            source = nodes[str(edge.source_node_code)]
            if edge.rule_ref is not None:
                target_refs.add(edge.rule_ref)
            trusted = _trusted_status(source, blocked_by_node[str(source.node_code)])
            if trusted is not GraphNodeStatus.SUPPORTED:
                if trusted is None:
                    unresolved.add(str(source.node_code))
                continue
            if edge.edge_type is EdgeType.SUPPORTS:
                explicit_support.add(str(source.node_code))
                support_explanations.add(f"{source.node_code} SUPPORTS {target}")
            elif edge.edge_type in {EdgeType.REFUTES, EdgeType.FALSIFIES}:
                explicit_refute.add(str(source.node_code))
                refute_explanations.add(f"{source.node_code} {edge.edge_type.value} {target}")
            elif edge.edge_type is EdgeType.EXCLUDES:
                explicit_exclude.add(str(source.node_code))
                refute_explanations.add(f"{source.node_code} EXCLUDES {target}")
            elif edge.edge_type is EdgeType.CONFLICTS_WITH:
                explicit_conflict.add(str(source.node_code))
                conflict_explanations.add(f"{source.node_code} conflicts with {target}")

        support.update(explicit_support)
        refute.update(explicit_refute)
        exclude.update(explicit_exclude)
        conflicts.update(explicit_conflict)
        positive_signal = gate_desired is GraphNodeStatus.SUPPORTED or bool(explicit_support)
        negative_signal = gate_desired is GraphNodeStatus.REFUTED or bool(
            explicit_refute | explicit_exclude
        )
        desired: GraphNodeStatus | None
        if explicit_conflict or (positive_signal and negative_signal):
            desired = GraphNodeStatus.CONFLICTED
        elif explicit_exclude:
            desired = GraphNodeStatus.EXCLUDED
        elif negative_signal:
            desired = GraphNodeStatus.REFUTED
        elif positive_signal:
            desired = GraphNodeStatus.SUPPORTED
        else:
            desired = None
        target_blocked = tuple(
            sorted(
                {item.node_code for code in unresolved for item in blocked_by_node.get(code, ())}
            )
        )
        all_support.update(support)
        all_refute.update(refute | exclude)
        if desired is GraphNodeStatus.CONFLICTED:
            all_conflict.update(
                explicit_conflict
                | explicit_support
                | explicit_refute
                | explicit_exclude
                | (support if positive_signal else set())
                | (refute if negative_signal else set())
            )
        all_refs.update(target_refs)
        node_results.append(
            NodePropagation(
                target,
                nodes[target].status,
                desired,
                tuple(sorted(support)),
                tuple(sorted(refute)),
                tuple(sorted(exclude)),
                tuple(sorted(conflicts)),
                tuple(sorted(unresolved)),
                target_blocked,
                tuple(sorted(target_refs, key=_ref_key)),
                "conflict dominates support/refutation"
                if desired is GraphNodeStatus.CONFLICTED
                else "deterministic child/source to parent/target evaluation",
            )
        )

    node_results_tuple = tuple(node_results)
    transition_blocks = tuple(
        block for result in node_results_tuple if (block := _transition_block(result)) is not None
    )
    completeness_blocked_nodes = {
        item.node_code
        for item in blocked
        if item.node_code in required_nodes
        and item.channel in _TECHNICAL_BLOCKING_CHANNELS | {"UNRESOLVED_EVIDENCE"}
    }
    resolved_nodes = {
        code
        for code in required_nodes
        if (
            _trusted_status(nodes[code], blocked_by_node[code]) is not None
            or (
                nodes[code].status is GraphNodeStatus.EXCLUDED
                and nodes[code].availability is DataAvailability.AVAILABLE
                and not any(
                    item.channel in _TECHNICAL_BLOCKING_CHANNELS for item in blocked_by_node[code]
                )
            )
        )
    }
    required_count = len(required_nodes)
    resolved_count = len(resolved_nodes)
    blocked_count = len(completeness_blocked_nodes)
    return PropagationEvaluation(
        support=_component(
            all_support,
            all_refs,
            required=required_count,
            resolved=resolved_count,
            blocked=blocked_count,
            explanations=support_explanations,
        ),
        refutation=_component(
            all_refute,
            all_refs,
            required=required_count,
            resolved=resolved_count,
            blocked=blocked_count,
            explanations=refute_explanations,
        ),
        completeness=_component(
            set(),
            all_refs,
            required=required_count,
            resolved=resolved_count,
            blocked=blocked_count,
            explanations={"counts only; no hidden score or probability"},
        ),
        conflict=_component(
            all_conflict,
            all_refs,
            required=required_count,
            resolved=resolved_count,
            blocked=blocked_count,
            explanations=conflict_explanations,
        ),
        node_results=node_results_tuple,
        blocked_evidence=blocked,
        transition_blocks=transition_blocks,
        data_flow_edges=tuple(
            sorted(
                (
                    str(edge.source_node_code),
                    str(edge.target_node_code),
                    edge.edge_type,
                )
                for edge in graph.edges
                if edge.edge_type in DATA_FLOW_EDGE_TYPES
            )
        ),
    )


def _transition_reset_ref(code: str) -> VersionedReference:
    return transition_reset_reference(code)


def transition_reset_audit_from_delta(delta: GraphDelta) -> TransitionResetAudit:
    """Recover reset intent with the same policy enforced at graph write boundaries."""
    try:
        return transition_reset_metadata_from_delta(delta)
    except GraphEngineError as error:
        raise RuleEngineError(
            RuleEngineErrorCode.TRANSITION_AUDIT_INVALID,
            "transition reset metadata is missing, forged, or inconsistent",
        ) from error


def transition_reset_audit_from_event(
    event: AcceptedDeltaEvent,
) -> TransitionResetAudit:
    """Validate an accepted event and recover reset intent without in-memory trace state."""

    try:
        canonical_event = accepted_delta_event_from_payload(accepted_delta_event_to_payload(event))
        audit = transition_reset_audit_from_delta(canonical_event.delta)
        if audit.delta_checksum != canonical_event.delta_checksum:
            raise ValueError("reset Delta checksum differs from accepted event")
        return replace(audit, event_checksum=canonical_event.event_checksum)
    except RuleEngineError:
        raise
    except (AttributeError, GraphEngineError, TypeError, ValueError, ValidationError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.TRANSITION_AUDIT_INVALID,
            "accepted event does not contain a valid transition reset",
        ) from error


def transition_reset_audit_from_event_payload(payload: object) -> TransitionResetAudit:
    """Recover reset intent directly from the strict persisted accepted-event JSON payload."""

    try:
        event = accepted_delta_event_from_payload(payload)
    except (GraphEngineError, TypeError, ValueError, ValidationError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.TRANSITION_AUDIT_INVALID,
            "persisted accepted event is not strict canonical JSON",
        ) from error
    return transition_reset_audit_from_event(event)


def _delta(
    graph: GECEGInstance,
    validated: ValidatedRulePackage,
    evaluation: PropagationEvaluation,
) -> GraphDelta | None:
    nodes = {str(node.node_code): node for node in graph.nodes}
    transition_by_node = {item.node_code: item for item in evaluation.transition_blocks}
    changes: list[
        tuple[str, GraphNodeStatus, tuple[VersionedReference, ...], TransitionBlocked | None]
    ] = []
    for result in evaluation.node_results:
        desired = result.derived_status
        if desired is None or desired is result.current_status:
            continue
        transition = transition_by_node.get(result.node_code)
        planned = transition.next_status if transition is not None else desired
        if planned is None:
            continue
        changes.append((result.node_code, planned, result.rule_refs, transition))
    if not changes:
        return None
    changes.sort(key=lambda item: item[0])
    reset_changes = [item for item in changes if item[3] is not None]
    if reset_changes:
        changes = reset_changes
    refs = tuple(
        sorted(
            {ref for _, _, result_refs, _ in changes for ref in result_refs},
            key=_ref_key,
        )
    )
    reason_code_set = {str(ref.code) for ref in refs}
    if reset_changes:
        reset_reason_codes = {
            TRANSITION_RESET_MARKER,
            TRANSITION_RESET_REASON,
            *(
                f"{_TRANSITION_RESET_OPERATION_PREFIX}{index:03d}_DERIVED_{transition.derived_status.value}"
                for index, (_, _, _, transition) in enumerate(changes)
                if transition is not None
            ),
        }
        refs = tuple(
            sorted(
                {*refs, *(_transition_reset_ref(code) for code in reset_reason_codes)},
                key=_ref_key,
            )
        )
        reason_code_set.update(reset_reason_codes)
    reason_codes = tuple(sorted(reason_code_set))
    identity_payload = {
        "engine_version": RULE_ENGINE_VERSION,
        "graph_checksum": canonical_checksum(graph),
        "package_checksum": validated.canonical_checksum,
        "changes": tuple(
            (
                code,
                status.value,
                transition.derived_status.value if transition is not None else None,
            )
            for code, status, _, transition in changes
        ),
    }
    opaque = canonical_checksum(identity_payload).split(":", 1)[1][:20].upper()
    provenance = Provenance(
        actor=ActorType.RULE_ENGINE,
        created_at=validated.package.provenance.created_at,
        generator_version=RULE_ENGINE_VERSION,
        source_codes=validated.package.provenance.source_codes,
        rule_versions=refs,
        knowledge_versions=validated.package.provenance.knowledge_versions,
        tool_versions=validated.package.provenance.tool_versions,
        input_checksums=(canonical_checksum(graph), validated.canonical_checksum),
    )
    return GraphDelta(
        schema_version=graph.schema_version,
        contract_type="GraphDelta",
        provenance=provenance,
        scope=graph.scope,
        laterality=graph.laterality,
        coordinate_space=graph.coordinate_space,
        unit=graph.unit,
        extensions={},
        graph_code=graph.graph_code,
        base_version=graph.graph_version,
        idempotency_key=f"TASK:OID_R03C{opaque}",
        proposed_by=ActorType.RULE_ENGINE,
        proposal_kind=ProposalKind.GRAPH_DELTA_VALIDATION,
        reason_codes=reason_codes,
        operations=tuple(
            UpdateNodeStatusOperation(
                op="UPDATE_NODE_STATUS",
                node_code=code,
                from_status=nodes[code].status,
                to_status=status,
                source=GraphUpdateSource.DETERMINISTIC_RULE,
            )
            for code, status, _, _ in changes
        ),
    )


def _legal_actions(
    graph: GECEGInstance,
    validated: ValidatedRulePackage,
    evaluation: PropagationEvaluation,
    candidate: GraphDelta | None,
    context: PropagationContext,
) -> tuple[LegalAction, ...]:
    if (
        context.budget.remaining_actions == 0
        or context.budget.round_index >= context.budget.max_rounds
    ):
        return ()
    actions: set[tuple[PlanAction, str | None, str]] = set()
    if candidate is not None:
        actions.add((PlanAction.VERIFY, str(graph.goal_node_code), "RULE:APPLY_CANDIDATE_DELTA"))
    blocked = {item.node_code: item for item in evaluation.blocked_evidence}
    gates = {gate.parent_node_code: gate for gate in validated.package.gates}
    nodes = {str(node.node_code): node for node in graph.nodes}
    for result in evaluation.node_results:
        gate = gates.get(result.node_code)
        gate_children = set(gate.child_node_codes) if gate is not None else set()
        gate_is_decisive = gate is not None and (
            (gate.operator == "OR" and bool(gate_children & set(result.supporting_nodes)))
            or (
                gate.operator == "AND"
                and any(
                    nodes[code].status is GraphNodeStatus.REFUTED
                    for code in gate_children & set(result.refuting_nodes)
                )
            )
        )
        explicit_sources = {
            str(edge.source_node_code)
            for edge in graph.edges
            if str(edge.target_node_code) == result.node_code
            and edge.edge_type in CLINICAL_EDGE_TYPES - {EdgeType.AND_CHILD, EdgeType.OR_CHILD}
        }
        has_blocked_or_sibling = (
            gate is not None and gate.operator == "OR" and bool(result.blocked_nodes)
        )
        for node_code in result.unresolved_nodes:
            if gate_is_decisive and node_code in gate_children - explicit_sources:
                continue
            node = nodes[node_code]
            item = blocked.get(node_code)
            if item is not None and item.channel in {
                "CAPABILITY_GAP",
                "UNOBSERVED_TOOL_GAP",
                "TOOL_EXECUTION_FAILED",
                "TOOL_EXECUTION_TIMEOUT",
                "QC_FAILED",
                "DATA_MISSING",
                "MEDICAL_NEGATIVE",
                "EVIDENCE_REFUTATION",
                "NOT_APPLICABLE",
            }:
                continue
            if has_blocked_or_sibling:
                action = PlanAction.SWITCH_PATH
            elif node.availability is DataAvailability.ACQUIRABLE:
                action = PlanAction.REQUEST_DATA
            elif node.availability is DataAvailability.DERIVABLE:
                action = PlanAction.SELECT_WORKFLOW
            else:
                action = PlanAction.SELECT_EVIDENCE_TASK
            actions.add((action, node_code, "RULE:UNRESOLVED_LEGAL_PATH"))
    return tuple(
        LegalAction(action, target, graph.graph_version, validated.canonical_checksum, reason)
        for action, target, reason in sorted(
            actions, key=lambda item: (item[0].value, item[1] or "", item[2])
        )
    )


def _stop(
    graph: GECEGInstance,
    validated: ValidatedRulePackage,
    evaluation: PropagationEvaluation,
    candidate: GraphDelta | None,
    actions: tuple[LegalAction, ...],
    context: PropagationContext,
) -> StopSuggestion | None:
    provenance = validated.package.provenance.model_copy(
        update={
            "input_checksums": (
                canonical_checksum(graph),
                validated.canonical_checksum,
            )
        }
    )
    if candidate is not None or actions:
        return None
    goal = next(node for node in graph.nodes if node.node_code == graph.goal_node_code)
    goal_result = next(
        (item for item in evaluation.node_results if item.node_code == str(graph.goal_node_code)),
        None,
    )
    pending_correction = (
        goal_result is not None
        and goal_result.derived_status is not None
        and goal_result.derived_status is not goal.status
    )
    unresolvable_transition = tuple(
        item for item in evaluation.transition_blocks if item.next_status is None
    )
    if unresolvable_transition:
        return StopSuggestion(
            StopReason.UNSAFE_TO_CONTINUE,
            ConclusionStatus.STOPPED_UNSAFE,
            (RuleEngineErrorCode.TRANSITION_BLOCKED.value,),
            provenance,
        )
    if pending_correction:
        if (
            context.budget.remaining_actions == 0
            or context.budget.round_index >= context.budget.max_rounds
        ):
            return StopSuggestion(
                StopReason.BUDGET_EXHAUSTED,
                ConclusionStatus.UNCERTAIN_MISSING_EVIDENCE,
                ("RULE:BUDGET_EXHAUSTED",),
                provenance,
            )
        return StopSuggestion(
            StopReason.INSUFFICIENT_EVIDENCE,
            ConclusionStatus.UNCERTAIN_MISSING_EVIDENCE,
            ("RULE:NO_PROGRESS",),
            provenance,
        )
    goal_is_consistent = goal_result is None or goal_result.derived_status is goal.status
    if goal_is_consistent and goal.status is GraphNodeStatus.SUPPORTED:
        return StopSuggestion(
            StopReason.GOAL_SUPPORTED,
            ConclusionStatus.SUPPORTED,
            ("RULE:GOAL_SUPPORTED",),
            provenance,
        )
    if goal_is_consistent and goal.status is GraphNodeStatus.REFUTED:
        return StopSuggestion(
            StopReason.GOAL_REFUTED,
            ConclusionStatus.REFUTED,
            ("RULE:GOAL_REFUTED",),
            provenance,
        )
    if goal_is_consistent and goal.status is GraphNodeStatus.CONFLICTED:
        return StopSuggestion(
            StopReason.CONFLICTING_EVIDENCE,
            ConclusionStatus.UNCERTAIN_CONFLICTING_EVIDENCE,
            ("RULE:CONFLICT_UNRESOLVED",),
            provenance,
        )
    if (
        context.budget.remaining_actions == 0
        or context.budget.round_index >= context.budget.max_rounds
    ):
        return StopSuggestion(
            StopReason.BUDGET_EXHAUSTED,
            ConclusionStatus.UNCERTAIN_MISSING_EVIDENCE,
            ("RULE:BUDGET_EXHAUSTED",),
            provenance,
        )
    channels = {item.channel for item in evaluation.blocked_evidence if item.required}
    if channels & {"CAPABILITY_GAP", "UNOBSERVED_TOOL_GAP"}:
        return StopSuggestion(
            StopReason.CAPABILITY_UNAVAILABLE,
            ConclusionStatus.CAPABILITY_UNAVAILABLE,
            ("RULE:REQUIRED_CAPABILITY_UNAVAILABLE",),
            provenance,
        )
    if channels & {"TOOL_EXECUTION_FAILED", "TOOL_EXECUTION_TIMEOUT", "QC_FAILED"}:
        return StopSuggestion(
            StopReason.TECHNICAL_FAILURE,
            ConclusionStatus.UNABLE_TO_EVALUATE_TECHNICAL,
            ("RULE:REQUIRED_TECHNICAL_FAILURE",),
            provenance,
        )
    current_checksum = canonical_checksum(graph)
    fixed_point = current_checksum in context.history.prior_graph_checksums
    return StopSuggestion(
        StopReason.INSUFFICIENT_EVIDENCE,
        ConclusionStatus.UNCERTAIN_MISSING_EVIDENCE,
        ("RULE:NO_PROGRESS" if fixed_point else "RULE:PATHS_EXHAUSTED",),
        provenance,
    )


def _empty_component(message: str) -> EvidenceComponent:
    return EvidenceComponent((), (), 0, 0, 1, (message,))


def _safe_invalid_result(
    graph: object, package: object, error: RuleEngineError
) -> PropagationResult:
    zero_checksum = "sha256:" + "0" * 64
    checksum = zero_checksum
    refs: tuple[VersionedReference, ...] = ()
    provenance: Provenance | None = None
    if type(package) is RulePackage:
        if type(package.package_checksum) is str:
            checksum = package.package_checksum
        if type(package.rule_refs) is tuple and all(
            type(item) is VersionedReference for item in package.rule_refs
        ):
            refs = package.rule_refs
        if type(package.provenance) is Provenance:
            try:
                provenance = _normalized_provenance(package.provenance)
            except RuleEngineError:
                provenance = None
    gap = KnowledgeGap(error.code.value, str(error), refs)
    component = _empty_component(str(error))
    evaluation = PropagationEvaluation(component, component, component, component, (), (), (), ())
    graph_checksum = zero_checksum
    if type(graph) is GECEGInstance:
        try:
            graph_checksum = canonical_checksum(graph)
            if provenance is None:
                provenance = graph.provenance.model_copy(update={"actor": ActorType.RULE_ENGINE})
        except (AttributeError, TypeError, ValueError):
            graph_checksum = zero_checksum
    if provenance is None:
        provenance = Provenance(
            actor=ActorType.RULE_ENGINE,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            generator_version=RULE_ENGINE_VERSION,
            source_codes=("RULE:03C_SAFE_INVALID",),
        )
    stop = StopSuggestion(
        StopReason.UNSAFE_TO_CONTINUE,
        ConclusionStatus.STOPPED_UNSAFE,
        (error.code.value,),
        provenance,
    )
    unsigned = {
        "payload_version": PROPAGATION_TRACE_VERSION,
        "engine_version": RULE_ENGINE_VERSION,
        "graph_checksum": graph_checksum,
        "rule_package_checksum": checksum,
        "node_results": (),
        "blocked_evidence": (),
        "transition_blocks": (),
        "data_flow_edges": (),
    }
    trace = PropagationTrace(
        **unsigned,
        trace_checksum=canonical_checksum(unsigned),
    )
    return PropagationResult(evaluation, None, (), stop, (gap,), trace)


def propagate(
    graph: GECEGInstance,
    package: RulePackage | None,
    context: PropagationContext,
) -> PropagationResult:
    """Evaluate one immutable graph version and propose at most one candidate Delta."""

    try:
        if type(package) is not RulePackage:
            _raise(RuleEngineErrorCode.PACKAGE_MISSING, "rule package is missing")
        if type(context) is not PropagationContext:
            _raise(RuleEngineErrorCode.CONTEXT_INVALID, "context has the wrong type")
        validated = validate_rule_package(
            graph,
            package,
            expected_version=context.expected_package_version,
            expected_checksum=context.expected_package_checksum,
        )
        evaluation = evaluate_propagation(graph, validated, context)
    except RuleEngineError as error:
        return _safe_invalid_result(graph, package, error)
    except (AttributeError, GraphEngineError, TypeError, ValueError, ValidationError) as error:
        wrapped = RuleEngineError(
            RuleEngineErrorCode.INPUT_INVALID,
            f"input cannot be validated safely: {type(error).__name__}",
        )
        return _safe_invalid_result(graph, package, wrapped)
    halted = (
        context.budget.remaining_actions == 0
        or context.budget.round_index >= context.budget.max_rounds
        or canonical_checksum(graph) in context.history.prior_graph_checksums
    )
    candidate = None if halted else _delta(graph, validated, evaluation)
    actions = () if halted else _legal_actions(graph, validated, evaluation, candidate, context)
    stop = _stop(graph, validated, evaluation, candidate, actions, context)
    unsigned = {
        "payload_version": PROPAGATION_TRACE_VERSION,
        "engine_version": RULE_ENGINE_VERSION,
        "graph_checksum": canonical_checksum(graph),
        "rule_package_checksum": validated.canonical_checksum,
        "node_results": evaluation.node_results,
        "blocked_evidence": evaluation.blocked_evidence,
        "transition_blocks": evaluation.transition_blocks,
        "data_flow_edges": evaluation.data_flow_edges,
    }
    trace = PropagationTrace(**unsigned, trace_checksum=canonical_checksum(unsigned))
    return PropagationResult(evaluation, candidate, actions, stop, (), trace)


def validate_expert_action(
    assessment: ExpertAssessment,
    legal_actions: tuple[LegalAction, ...],
) -> LegalAction | None:
    """Accept only a completed expert selection already present in the deterministic set."""

    try:
        parsed = ExpertAssessment.model_validate(assessment.model_dump(mode="python"))
    except (ValidationError, TypeError, ValueError) as error:
        raise RuleEngineError(
            RuleEngineErrorCode.EXPERT_ACTION_INVALID,
            "expert assessment violates packages.contracts v1",
        ) from error
    allowed = tuple(sorted({item.action for item in legal_actions}, key=lambda item: item.value))
    if tuple(sorted(set(parsed.allowed_action_codes), key=lambda item: item.value)) != allowed:
        _raise(RuleEngineErrorCode.EXPERT_ACTION_INVALID, "expert allowed set is not canonical")
    if parsed.status is not ExpertAssessmentStatus.COMPLETED:
        _raise(RuleEngineErrorCode.EXPERT_ACTION_INVALID, "expert assessment is not completed")
    if parsed.decision is ExpertAssessmentDecision.SELECT_ALLOWED_ALTERNATIVE:
        matches = tuple(
            item for item in legal_actions if item.action is parsed.selected_action_code
        )
        if not matches:
            _raise(RuleEngineErrorCode.EXPERT_ACTION_INVALID, "expert selected an illegal action")
        return matches[0]
    if parsed.selected_action_code is not None:
        _raise(RuleEngineErrorCode.EXPERT_ACTION_INVALID, "unexpected selected action")
    return None
