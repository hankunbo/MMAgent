"""Pure deterministic validation and stage-03A template instantiation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from packages.contracts import (
    ActorType,
    CapabilityAvailability,
    CaseBundle,
    DataAvailability,
    EdgeType,
    GECEGInstance,
    GECEGTemplate,
    GraphEdge,
    GraphNodeInstance,
    GraphNodeStatus,
    NodeType,
    Provenance,
    ScopeReference,
    ScopeType,
    VersionedReference,
)


class GraphEngineErrorCode(StrEnum):
    TEMPLATE_CONTRACT_INVALID = "GRA03A_TEMPLATE_CONTRACT_INVALID"
    TEMPLATE_CHECKSUM_MISMATCH = "GRA03A_TEMPLATE_CHECKSUM_MISMATCH"
    GOAL_COUNT = "GRA03A_GOAL_COUNT"
    GOAL_REFERENCE = "GRA03A_GOAL_REFERENCE"
    DUPLICATE_NODE = "GRA03A_DUPLICATE_NODE"
    UNKNOWN_EDGE_ENDPOINT = "GRA03A_UNKNOWN_EDGE_ENDPOINT"
    DUPLICATE_EDGE = "GRA03A_DUPLICATE_EDGE"
    SELF_REFERENCE = "GRA03A_SELF_REFERENCE"
    EDGE_NODE_TYPE_MISMATCH = "GRA03A_EDGE_NODE_TYPE_MISMATCH"
    UNREACHABLE_NODE = "GRA03A_UNREACHABLE_NODE"
    UNKNOWN_RESOLUTION_NODE = "GRA03A_UNKNOWN_RESOLUTION_NODE"
    DUPLICATE_RESOLUTION = "GRA03A_DUPLICATE_RESOLUTION"
    INVALID_RESOLUTION = "GRA03A_INVALID_RESOLUTION"
    UNKNOWN_ARTIFACT = "GRA03A_UNKNOWN_ARTIFACT"
    GOAL_NOT_APPLICABLE = "GRA03A_GOAL_NOT_APPLICABLE"
    PRUNING_DISCONNECTS_GRAPH = "GRA03A_PRUNING_DISCONNECTS_GRAPH"
    DELTA_CONTRACT_INVALID = "GRA03B_DELTA_CONTRACT_INVALID"
    DELTA_BASE_GRAPH_INVALID = "GRA03B_BASE_GRAPH_INVALID"
    DELTA_GRAPH_CODE_MISMATCH = "GRA03B_GRAPH_CODE_MISMATCH"
    DELTA_BASE_VERSION_MISMATCH = "GRA03B_BASE_VERSION_MISMATCH"
    DELTA_SCOPE_MISMATCH = "GRA03B_SCOPE_MISMATCH"
    DELTA_RULE_PROVENANCE_MISSING = "GRA03B_RULE_PROVENANCE_MISSING"
    DELTA_DUPLICATE_OPERATION = "GRA03B_DUPLICATE_OPERATION"
    DELTA_UNKNOWN_NODE = "GRA03B_UNKNOWN_NODE"
    DELTA_UNKNOWN_EDGE_ENDPOINT = "GRA03B_UNKNOWN_EDGE_ENDPOINT"
    DELTA_SELF_REFERENCE = "GRA03B_SELF_REFERENCE"
    DELTA_EDGE_NODE_TYPE_MISMATCH = "GRA03B_EDGE_NODE_TYPE_MISMATCH"
    DELTA_FACT_OVERWRITE_FORBIDDEN = "GRA03B_FACT_OVERWRITE_FORBIDDEN"
    DELTA_STATUS_FROM_MISMATCH = "GRA03B_STATUS_FROM_MISMATCH"
    DELTA_STATUS_NOOP = "GRA03B_STATUS_NOOP"
    DELTA_STATUS_TRANSITION_INVALID = "GRA03B_STATUS_TRANSITION_INVALID"
    DELTA_UPDATE_SOURCE_INVALID = "GRA03B_UPDATE_SOURCE_INVALID"
    DELTA_VALIDATION_REQUIRED = "GRA03B_VALIDATION_REQUIRED"
    DELTA_VALIDATION_MISMATCH = "GRA03B_VALIDATION_MISMATCH"
    DELTA_EVIDENCE_BINDING_REQUIRED = "GRA03B_EVIDENCE_BINDING_REQUIRED"
    DELTA_EVIDENCE_BINDING_UNKNOWN = "GRA03B_EVIDENCE_BINDING_UNKNOWN"
    DELTA_ORPHANED_GRAPH = "GRA03B_ORPHANED_GRAPH"
    DELTA_IDEMPOTENCY_CONFLICT = "GRA03B_IDEMPOTENCY_CONFLICT"
    DELTA_PLAN_BASE_MISMATCH = "GRA03B_PLAN_BASE_MISMATCH"
    DELTA_PLAN_INTEGRITY_INVALID = "GRA03B_PLAN_INTEGRITY_INVALID"
    DELTA_EVENT_PAYLOAD_INVALID = "GRA03B_EVENT_PAYLOAD_INVALID"
    DELTA_REPLAY_MISMATCH = "GRA03B_REPLAY_MISMATCH"
    TRANSITION_AUDIT_INVALID = "GRA03B_TRANSITION_AUDIT_INVALID"


class GraphEngineError(ValueError):
    """A stable, machine-readable stage-03 graph-engine error."""

    def __init__(self, code: GraphEngineErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


class Applicability(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class ResolutionKind(StrEnum):
    EXPLICIT = "EXPLICIT"
    MAPPING_NOT_PROVIDED = "MAPPING_NOT_PROVIDED"
    STRUCTURAL_DEFAULT = "STRUCTURAL_DEFAULT"


@dataclass(frozen=True, slots=True)
class ArtifactNodeBinding:
    """An explicit Grounding-supplied Artifact-to-node association."""

    asset_code: str
    binding_code: str


@dataclass(frozen=True, slots=True)
class NodeResolution:
    """Parsed applicability/availability input; this is a port, not a v1 contract clone."""

    node_code: str
    applicability: Applicability
    availability: DataAvailability
    capability: CapabilityAvailability = CapabilityAvailability.AVAILABLE
    artifact_bindings: tuple[ArtifactNodeBinding, ...] = ()
    observation_concept_codes: tuple[str, ...] = ()
    reason_code: str = "RULE:INSTANTIATION:EXPLICIT"
    rule_ref: VersionedReference | None = None


@dataclass(frozen=True, slots=True)
class InstantiationContext:
    graph_code: str
    graph_version: int
    template_checksum: str
    created_at: datetime
    generator_version: str
    source_codes: tuple[str, ...]
    resolutions: tuple[NodeResolution, ...]
    rule_versions: tuple[VersionedReference, ...] = ()
    run_id: str | None = None


INSTANTIATION_INPUT_PAYLOAD_VERSION = "03A_INPUT_V1"


@dataclass(frozen=True, slots=True)
class InstantiationInputPayload:
    """Canonical, versioned identity of every input that can affect 03A output."""

    payload_version: str
    template_checksum: str
    case_checksum: str
    graph_code: str
    graph_version: int
    run_id: str | None
    created_at: datetime
    generator_version: str
    source_codes: tuple[str, ...]
    rule_versions: tuple[VersionedReference, ...]
    resolutions: tuple[NodeResolution, ...]


@dataclass(frozen=True, slots=True)
class NodeInstantiationTrace:
    node_code: str
    resolution_kind: ResolutionKind
    applicability: Applicability
    availability: DataAvailability
    capability: CapabilityAvailability
    reason_code: str
    rule_ref: VersionedReference | None
    mapped_artifact_codes: tuple[str, ...]
    evidence_binding_codes: tuple[str, ...]
    observation_concept_codes: tuple[str, ...]
    input_checksum: str


@dataclass(frozen=True, slots=True)
class PrunedNodeTrace:
    node_code: str
    reason_code: str
    rule_ref: VersionedReference | None
    input_checksum: str


@dataclass(frozen=True, slots=True)
class InstantiationTrace:
    engine_version: str
    input_payload_version: str
    template_ref: VersionedReference
    case_checksum: str
    input_checksum: str
    node_results: tuple[NodeInstantiationTrace, ...]
    pruned_nodes: tuple[PrunedNodeTrace, ...]


@dataclass(frozen=True, slots=True)
class InstantiationResult:
    instance: GECEGInstance
    trace: InstantiationTrace


CLINICAL_EDGE_TYPES = frozenset(
    {
        EdgeType.AND_CHILD,
        EdgeType.OR_CHILD,
        EdgeType.SUPPORTS,
        EdgeType.REFUTES,
        EdgeType.EXCLUDES,
        EdgeType.FALSIFIES,
        EdgeType.CAUSES,
        EdgeType.RISK_MODIFIES,
        EdgeType.CONFLICTS_WITH,
    }
)
DATA_FLOW_EDGE_TYPES = frozenset(
    {EdgeType.REQUIRES_INPUT, EdgeType.PRODUCES, EdgeType.VALIDATED_BY}
)

_LOGICAL_TARGETS = frozenset({NodeType.GOAL, NodeType.CLAIM, NodeType.LOGIC_GATE})
_PROPOSITION_TARGETS = frozenset({NodeType.GOAL, NodeType.CLAIM, NodeType.EVIDENCE})
_CLINICAL_SOURCES = frozenset(
    {NodeType.CLAIM, NodeType.EVIDENCE, NodeType.DATA, NodeType.QC, NodeType.EXCLUSION}
)
_CLINICAL_REFUTATION_SOURCES = frozenset(
    {NodeType.CLAIM, NodeType.EVIDENCE, NodeType.QC, NodeType.EXCLUSION}
)
_EDGE_ENDPOINTS: dict[EdgeType, tuple[frozenset[NodeType], frozenset[NodeType]]] = {
    EdgeType.AND_CHILD: (
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE, NodeType.EXCLUSION, NodeType.LOGIC_GATE}),
        _LOGICAL_TARGETS,
    ),
    EdgeType.OR_CHILD: (
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE, NodeType.EXCLUSION, NodeType.LOGIC_GATE}),
        _LOGICAL_TARGETS,
    ),
    EdgeType.SUPPORTS: (_CLINICAL_SOURCES, _PROPOSITION_TARGETS),
    EdgeType.REFUTES: (_CLINICAL_REFUTATION_SOURCES, _PROPOSITION_TARGETS),
    EdgeType.EXCLUDES: (frozenset({NodeType.EXCLUSION, NodeType.EVIDENCE}), _LOGICAL_TARGETS),
    EdgeType.FALSIFIES: (_CLINICAL_REFUTATION_SOURCES, _LOGICAL_TARGETS),
    EdgeType.CAUSES: (
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE}),
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE}),
    ),
    EdgeType.RISK_MODIFIES: (_CLINICAL_SOURCES, _LOGICAL_TARGETS),
    EdgeType.CONFLICTS_WITH: (
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE, NodeType.EXCLUSION}),
        frozenset({NodeType.CLAIM, NodeType.EVIDENCE, NodeType.EXCLUSION}),
    ),
    EdgeType.REQUIRES_INPUT: (frozenset({NodeType.DATA}), frozenset({NodeType.TOOL})),
    EdgeType.PRODUCES: (
        frozenset({NodeType.TOOL, NodeType.ACTION}),
        frozenset({NodeType.DATA}),
    ),
    EdgeType.VALIDATED_BY: (
        frozenset({NodeType.DATA, NodeType.EVIDENCE}),
        frozenset({NodeType.QC}),
    ),
}
_STRUCTURAL_NODE_TYPES = frozenset(
    {NodeType.GOAL, NodeType.CLAIM, NodeType.EXCLUSION, NodeType.LOGIC_GATE}
)


def _canonical_json_value(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, BaseModel):
        return _canonical_json_value(value.model_dump(mode="json"), path=path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_json_value(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in fields(value)
        }
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TypeError(f"{path} datetime must be timezone-aware canonical JSON input")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} non-finite float is not canonical JSON")
        return value
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} canonical JSON object keys must be strings")
        return {
            key: _canonical_json_value(item, path=f"{path}.{key}") for key, item in value.items()
        }
    raise TypeError(f"{path} value of type {type(value).__name__} is not canonical JSON")


def canonical_checksum(value: Any) -> str:
    """Return strict canonical JSON SHA-256; unknown values are rejected, never stringified."""

    payload = _canonical_json_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reference_key(reference: VersionedReference) -> tuple[str, str, str]:
    return (str(reference.code), reference.version, reference.checksum or "")


def _normalize_resolution(resolution: NodeResolution) -> NodeResolution:
    if (
        type(resolution) is not NodeResolution
        or not isinstance(resolution.applicability, Applicability)
        or not isinstance(resolution.availability, DataAvailability)
        or not isinstance(resolution.capability, CapabilityAvailability)
    ):
        _raise(
            GraphEngineErrorCode.INVALID_RESOLUTION,
            "resolution requires canonical applicability, availability, and capability enums",
        )
    return replace(
        resolution,
        artifact_bindings=tuple(
            sorted(
                resolution.artifact_bindings,
                key=lambda binding: (binding.asset_code, binding.binding_code),
            )
        ),
        observation_concept_codes=tuple(sorted(set(resolution.observation_concept_codes))),
    )


def build_instantiation_input_payload(
    *,
    template_checksum: str,
    case_checksum: str,
    context: InstantiationContext,
) -> InstantiationInputPayload:
    """Normalize set-like context fields into the versioned checksum payload."""

    normalized_created_at = context.created_at
    if normalized_created_at.tzinfo is not None and normalized_created_at.utcoffset() is not None:
        normalized_created_at = normalized_created_at.astimezone(UTC)
    rule_versions = tuple(
        sorted(
            {_reference_key(reference): reference for reference in context.rule_versions}.values(),
            key=_reference_key,
        )
    )
    resolutions = tuple(
        sorted(
            (_normalize_resolution(resolution) for resolution in context.resolutions),
            key=lambda resolution: resolution.node_code,
        )
    )
    return InstantiationInputPayload(
        payload_version=INSTANTIATION_INPUT_PAYLOAD_VERSION,
        template_checksum=template_checksum,
        case_checksum=case_checksum,
        graph_code=context.graph_code,
        graph_version=context.graph_version,
        run_id=context.run_id,
        created_at=normalized_created_at,
        generator_version=context.generator_version,
        source_codes=tuple(sorted(set(context.source_codes))),
        rule_versions=rule_versions,
        resolutions=resolutions,
    )


def _raise(code: GraphEngineErrorCode, message: str) -> None:
    raise GraphEngineError(code, message)


def validate_template(
    template: GECEGTemplate, *, expected_checksum: str | None = None
) -> GECEGTemplate:
    """Validate the 03A semantics missing from the frozen generic v1 graph contract."""

    if expected_checksum is not None and canonical_checksum(template) != expected_checksum:
        _raise(GraphEngineErrorCode.TEMPLATE_CHECKSUM_MISMATCH, "template checksum differs")

    node_codes = tuple(str(node.node_code) for node in template.nodes)
    if len(node_codes) != len(set(node_codes)):
        _raise(GraphEngineErrorCode.DUPLICATE_NODE, "node_code values must be unique")
    node_by_code = {str(node.node_code): node for node in template.nodes}
    goals = tuple(node for node in template.nodes if node.node_type is NodeType.GOAL)
    if len(goals) != 1:
        _raise(GraphEngineErrorCode.GOAL_COUNT, "template must contain exactly one GOAL node")
    if str(goals[0].node_code) != str(template.goal_node_code):
        _raise(GraphEngineErrorCode.GOAL_REFERENCE, "goal_node_code must identify the GOAL node")

    edge_keys: set[tuple[str, str, EdgeType]] = set()
    adjacency = {code: set() for code in node_codes}
    for edge in template.edges:
        source = str(edge.source_node_code)
        target = str(edge.target_node_code)
        if source not in node_by_code or target not in node_by_code:
            _raise(
                GraphEngineErrorCode.UNKNOWN_EDGE_ENDPOINT,
                f"unknown endpoint in {source}->{target}",
            )
        if source == target:
            _raise(GraphEngineErrorCode.SELF_REFERENCE, f"self edge at {source}")
        key = (source, target, edge.edge_type)
        if key in edge_keys:
            _raise(GraphEngineErrorCode.DUPLICATE_EDGE, f"duplicate edge {key}")
        edge_keys.add(key)
        source_types, target_types = _EDGE_ENDPOINTS[edge.edge_type]
        source_type = node_by_code[source].node_type
        target_type = node_by_code[target].node_type
        if source_type not in source_types or target_type not in target_types:
            _raise(
                GraphEngineErrorCode.EDGE_NODE_TYPE_MISMATCH,
                f"{edge.edge_type.value} forbids {source_type.value}->{target_type.value}",
            )
        adjacency[source].add(target)
        adjacency[target].add(source)

    if len(node_codes) > 1:
        visited: set[str] = set()
        pending = [str(template.goal_node_code)]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(sorted(adjacency[current] - visited, reverse=True))
        unreachable = sorted(set(node_codes) - visited)
        if unreachable:
            _raise(
                GraphEngineErrorCode.UNREACHABLE_NODE,
                "nodes are not connected to the goal: " + ", ".join(unreachable),
            )
    try:
        GECEGTemplate.model_validate(template.model_dump(mode="python"))
    except ValidationError as error:
        raise GraphEngineError(
            GraphEngineErrorCode.TEMPLATE_CONTRACT_INVALID,
            "template violates packages.contracts v1",
        ) from error
    return template


def validate_instance_structure(instance: GECEGInstance) -> GECEGInstance:
    """Validate one instance against the canonical 03A edge/endpoint matrix.

    This is the public structural boundary shared by later deterministic stages. It
    does not interpret propagation state and has no persistence side effects.
    """

    try:
        parsed = GECEGInstance.model_validate(instance.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.TEMPLATE_CONTRACT_INVALID,
            "instance violates packages.contracts v1",
        ) from error
    node_codes = tuple(str(node.node_code) for node in parsed.nodes)
    if len(node_codes) != len(set(node_codes)):
        _raise(GraphEngineErrorCode.DUPLICATE_NODE, "node_code values must be unique")
    node_by_code = {str(node.node_code): node for node in parsed.nodes}
    goals = tuple(node for node in parsed.nodes if node.node_type is NodeType.GOAL)
    if len(goals) != 1:
        _raise(GraphEngineErrorCode.GOAL_COUNT, "instance must contain exactly one GOAL node")
    if str(goals[0].node_code) != str(parsed.goal_node_code):
        _raise(GraphEngineErrorCode.GOAL_REFERENCE, "goal_node_code must identify the GOAL node")

    edge_keys: set[tuple[str, str, EdgeType]] = set()
    for edge in parsed.edges:
        source = str(edge.source_node_code)
        target = str(edge.target_node_code)
        if source not in node_by_code or target not in node_by_code:
            _raise(
                GraphEngineErrorCode.UNKNOWN_EDGE_ENDPOINT,
                f"unknown endpoint in {source}->{target}",
            )
        if source == target:
            _raise(GraphEngineErrorCode.SELF_REFERENCE, f"self edge at {source}")
        key = (source, target, edge.edge_type)
        if key in edge_keys:
            _raise(GraphEngineErrorCode.DUPLICATE_EDGE, f"duplicate edge {key}")
        edge_keys.add(key)
        allowed_sources, allowed_targets = _EDGE_ENDPOINTS[edge.edge_type]
        source_type = node_by_code[source].node_type
        target_type = node_by_code[target].node_type
        if source_type not in allowed_sources or target_type not in allowed_targets:
            _raise(
                GraphEngineErrorCode.EDGE_NODE_TYPE_MISMATCH,
                f"{edge.edge_type.value} forbids {source_type.value}->{target_type.value}",
            )

    unreachable = _unreachable_from_goal(set(node_codes), parsed.edges, str(parsed.goal_node_code))
    if unreachable:
        _raise(
            GraphEngineErrorCode.UNREACHABLE_NODE,
            "nodes are not connected to the goal: " + ", ".join(unreachable),
        )
    return parsed


def _validate_resolution(
    resolution: NodeResolution, node_type: NodeType, artifact_codes: set[str]
) -> None:
    if resolution.applicability is Applicability.UNKNOWN and (
        resolution.availability is not DataAvailability.UNAVAILABLE
        or resolution.artifact_bindings
        or resolution.observation_concept_codes
        or resolution.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY
    ):
        _raise(
            GraphEngineErrorCode.INVALID_RESOLUTION,
            "unknown applicability retains unavailable evidence without capability assertions",
        )
    if resolution.applicability is Applicability.NOT_APPLICABLE:
        if resolution.availability is not DataAvailability.NOT_APPLICABLE:
            _raise(
                GraphEngineErrorCode.INVALID_RESOLUTION,
                "NOT_APPLICABLE requires DataAvailability.NOT_APPLICABLE",
            )
        if resolution.artifact_bindings or resolution.observation_concept_codes:
            _raise(GraphEngineErrorCode.INVALID_RESOLUTION, "pruned nodes cannot bind evidence")
    elif resolution.availability is DataAvailability.NOT_APPLICABLE:
        _raise(
            GraphEngineErrorCode.INVALID_RESOLUTION,
            "NOT_APPLICABLE availability requires not-applicable resolution",
        )
    if resolution.capability is CapabilityAvailability.UNAVAILABLE_CAPABILITY:
        if resolution.availability is not DataAvailability.UNAVAILABLE:
            _raise(
                GraphEngineErrorCode.INVALID_RESOLUTION,
                "unavailable capability requires UNAVAILABLE data availability",
            )
        if resolution.artifact_bindings:
            _raise(GraphEngineErrorCode.INVALID_RESOLUTION, "capability gap cannot bind evidence")
    if node_type in _STRUCTURAL_NODE_TYPES and resolution.artifact_bindings:
        _raise(GraphEngineErrorCode.INVALID_RESOLUTION, "structural nodes cannot bind artifacts")
    if (
        resolution.availability is DataAvailability.AVAILABLE
        and not resolution.artifact_bindings
        and not resolution.observation_concept_codes
    ):
        _raise(
            GraphEngineErrorCode.INVALID_RESOLUTION,
            "AVAILABLE requires an explicit Artifact or StructuredObservation source",
        )
    for binding in resolution.artifact_bindings:
        if binding.asset_code not in artifact_codes:
            _raise(GraphEngineErrorCode.UNKNOWN_ARTIFACT, f"unknown asset {binding.asset_code}")


def _unreachable_from_goal(
    node_codes: set[str], edges: tuple[GraphEdge, ...], goal_node_code: str
) -> tuple[str, ...]:
    adjacency = {code: set() for code in node_codes}
    for edge in edges:
        source = str(edge.source_node_code)
        target = str(edge.target_node_code)
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited: set[str] = set()
    pending = [goal_node_code]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(sorted(adjacency[current] - visited, reverse=True))
    return tuple(sorted(node_codes - visited))


def instantiate_template(
    template: GECEGTemplate, case_bundle: CaseBundle, context: InstantiationContext
) -> InstantiationResult:
    """Instantiate without mutation, inference by keywords, persistence, or propagation."""

    validate_template(template, expected_checksum=context.template_checksum)
    resolution_codes = tuple(item.node_code for item in context.resolutions)
    if len(resolution_codes) != len(set(resolution_codes)):
        _raise(GraphEngineErrorCode.DUPLICATE_RESOLUTION, "one resolution per node is required")
    template_codes = {str(node.node_code) for node in template.nodes}
    unknown = sorted(set(resolution_codes) - template_codes)
    if unknown:
        _raise(
            GraphEngineErrorCode.UNKNOWN_RESOLUTION_NODE,
            "resolution references unknown nodes: " + ", ".join(unknown),
        )

    case_checksum = canonical_checksum(case_bundle)
    input_payload = build_instantiation_input_payload(
        template_checksum=context.template_checksum,
        case_checksum=case_checksum,
        context=context,
    )
    input_checksum = canonical_checksum(input_payload)
    resolution_by_code = {item.node_code: item for item in input_payload.resolutions}
    artifact_codes = {str(artifact.asset_code) for artifact in case_bundle.artifacts}
    observation_codes = {str(item.concept_code) for item in case_bundle.observations}
    nodes: list[GraphNodeInstance] = []
    node_traces: list[NodeInstantiationTrace] = []
    pruned: list[PrunedNodeTrace] = []

    for template_node in template.nodes:
        node_code = str(template_node.node_code)
        resolution = resolution_by_code.get(node_code)
        kind = ResolutionKind.EXPLICIT
        if resolution is None:
            if template_node.node_type in _STRUCTURAL_NODE_TYPES:
                kind = ResolutionKind.STRUCTURAL_DEFAULT
                resolution = NodeResolution(
                    node_code=node_code,
                    applicability=Applicability.APPLICABLE,
                    availability=DataAvailability.DERIVABLE,
                    reason_code="RULE:INSTANTIATION:STRUCTURAL_DEFAULT",
                )
            else:
                kind = ResolutionKind.MAPPING_NOT_PROVIDED
                resolution = NodeResolution(
                    node_code=node_code,
                    applicability=Applicability.APPLICABLE,
                    availability=DataAvailability.UNAVAILABLE,
                    reason_code="RULE:INSTANTIATION:MAPPING_NOT_PROVIDED",
                )
        _validate_resolution(resolution, template_node.node_type, artifact_codes)
        missing_observations = sorted(set(resolution.observation_concept_codes) - observation_codes)
        if missing_observations:
            _raise(
                GraphEngineErrorCode.INVALID_RESOLUTION,
                "explicit observations are absent from CaseBundle: "
                + ", ".join(missing_observations),
            )
        binding_codes = tuple(binding.binding_code for binding in resolution.artifact_bindings)
        if len(binding_codes) != len(set(binding_codes)):
            _raise(GraphEngineErrorCode.INVALID_RESOLUTION, "binding codes must be unique")
        mapped_assets = tuple(binding.asset_code for binding in resolution.artifact_bindings)
        node_trace = NodeInstantiationTrace(
            node_code=node_code,
            resolution_kind=kind,
            applicability=resolution.applicability,
            availability=resolution.availability,
            capability=resolution.capability,
            reason_code=resolution.reason_code,
            rule_ref=resolution.rule_ref,
            mapped_artifact_codes=mapped_assets,
            evidence_binding_codes=binding_codes,
            observation_concept_codes=resolution.observation_concept_codes,
            input_checksum=input_checksum,
        )
        node_traces.append(node_trace)
        if resolution.applicability is Applicability.NOT_APPLICABLE:
            if node_code == str(template.goal_node_code):
                _raise(GraphEngineErrorCode.GOAL_NOT_APPLICABLE, "the sole goal cannot be pruned")
            pruned.append(
                PrunedNodeTrace(
                    node_code=node_code,
                    reason_code=resolution.reason_code,
                    rule_ref=resolution.rule_ref,
                    input_checksum=input_checksum,
                )
            )
            continue
        nodes.append(
            GraphNodeInstance(
                node_code=template_node.node_code,
                node_type=template_node.node_type,
                concept_code=template_node.concept_code,
                status=GraphNodeStatus.PENDING,
                availability=resolution.availability,
                evidence_bindings=binding_codes,
            )
        )

    retained_codes = {str(node.node_code) for node in nodes}
    edges = tuple(
        edge
        for edge in template.edges
        if str(edge.source_node_code) in retained_codes
        and str(edge.target_node_code) in retained_codes
    )
    unreachable = _unreachable_from_goal(
        retained_codes,
        edges,
        str(template.goal_node_code),
    )
    if unreachable:
        _raise(
            GraphEngineErrorCode.PRUNING_DISCONNECTS_GRAPH,
            "pruning disconnects retained nodes from the goal: " + ", ".join(unreachable),
        )
    template_ref = VersionedReference(
        code=template.template_code,
        version=template.template_version,
        checksum=input_payload.template_checksum,
    )
    provenance = Provenance(
        actor=ActorType.GRAPH_ENGINE,
        created_at=input_payload.created_at,
        generator_version=input_payload.generator_version,
        source_codes=input_payload.source_codes,
        rule_versions=input_payload.rule_versions,
        knowledge_versions=template.citations,
        input_checksums=(input_checksum,),
    )
    instance = GECEGInstance(
        schema_version="1.0",
        contract_type="GECEGInstance",
        provenance=provenance,
        scope=ScopeReference(
            scope_type=ScopeType.GRAPH,
            case_code=case_bundle.case_code,
            run_id=input_payload.run_id,
            graph_code=input_payload.graph_code,
        ),
        laterality=case_bundle.laterality,
        coordinate_space=case_bundle.coordinate_space,
        unit=case_bundle.unit,
        graph_code=input_payload.graph_code,
        case_code=case_bundle.case_code,
        template_ref=template_ref,
        graph_version=input_payload.graph_version,
        goal_node_code=template.goal_node_code,
        nodes=tuple(nodes),
        edges=edges,
    )
    return InstantiationResult(
        instance=instance,
        trace=InstantiationTrace(
            engine_version=input_payload.generator_version,
            input_payload_version=input_payload.payload_version,
            template_ref=template_ref,
            case_checksum=case_checksum,
            input_checksum=input_checksum,
            node_results=tuple(node_traces),
            pruned_nodes=tuple(pruned),
        ),
    )
