"""Pure deterministic stage-03B GraphDelta validation, planning, apply, and replay."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace

from pydantic import JsonValue, ValidationError

from packages.contracts import (
    ActorType,
    AddEdgeOperation,
    AddNodeOperation,
    GECEGInstance,
    GraphDelta,
    GraphEdge,
    GraphNodeInstance,
    GraphNodeStatus,
    GraphUpdateSource,
    NodeType,
    Provenance,
    UpdateNodeStatusOperation,
    ValidationResult,
    VersionedReference,
    can_transition,
)

from .engine import (
    _EDGE_ENDPOINTS,
    GraphEngineError,
    GraphEngineErrorCode,
    _unreachable_from_goal,
    canonical_checksum,
    validate_instance_structure,
)
from .transition_reset import validate_transition_reset_metadata

DELTA_ENGINE_VERSION = "03B.2.0"
GRAPH_STRUCTURE_PAYLOAD_VERSION = "03B_GRAPH_STRUCTURE_V1"
DELTA_PLAN_PAYLOAD_VERSION = "03B_DELTA_PLAN_V2"
ACCEPTED_DELTA_EVENT_VERSION = "03B_ACCEPTED_DELTA_V2"
_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ValidatedGraphDelta:
    """A contract Delta bound to one exact immutable base and validation proof."""

    delta: GraphDelta
    base_checksum: str
    base_structure_hash: str
    delta_checksum: str
    validation_checksum: str | None


@dataclass(frozen=True, slots=True)
class AcceptedDeltaEvent:
    """Replayable accepted event; persistence remains a caller responsibility."""

    event_version: str
    graph_code: str
    base_version: int
    result_version: int
    idempotency_key: str
    base_checksum: str
    base_structure_hash: str
    delta_checksum: str
    validation_checksum: str | None
    result_checksum: str
    result_structure_hash: str
    plan_checksum: str
    delta: GraphDelta
    validation_result: ValidationResult | None
    event_checksum: str


@dataclass(frozen=True, slots=True)
class GraphDeltaPlan:
    """Complete atomic plan; no externally visible partial graph can be produced."""

    payload_version: str
    validated: ValidatedGraphDelta
    validation_result: ValidationResult | None
    result_instance: GECEGInstance
    result_checksum: str
    result_structure_hash: str
    plan_checksum: str
    accepted_event: AcceptedDeltaEvent
    idempotent_retry: bool = False


@dataclass(frozen=True, slots=True)
class GraphDeltaApplicationResult:
    instance: GECEGInstance
    result_checksum: str
    structure_hash: str
    delta_checksum: str
    accepted_event: AcceptedDeltaEvent
    idempotent_retry: bool


def _fail(code: GraphEngineErrorCode, message: str) -> None:
    raise GraphEngineError(code, message)


def _reference_key(reference: VersionedReference) -> tuple[str, str, str]:
    return (str(reference.code), reference.version, reference.checksum or "")


def _normalized_references(
    *groups: tuple[VersionedReference, ...],
) -> tuple[VersionedReference, ...]:
    unique = {_reference_key(reference): reference for group in groups for reference in group}
    return tuple(sorted(unique.values(), key=_reference_key))


def graph_structure_hash(graph: GECEGInstance) -> str:
    """Hash the complete business snapshot apart from generation provenance."""

    return canonical_checksum(
        {
            "payload_version": GRAPH_STRUCTURE_PAYLOAD_VERSION,
            "schema_version": graph.schema_version,
            "contract_type": graph.contract_type,
            "graph_code": graph.graph_code,
            "case_code": graph.case_code,
            "template_ref": graph.template_ref,
            "graph_version": graph.graph_version,
            "goal_node_code": graph.goal_node_code,
            "scope": graph.scope,
            "laterality": graph.laterality,
            "coordinate_space": graph.coordinate_space,
            "unit": graph.unit,
            "extensions": graph.extensions,
            "nodes": graph.nodes,
            "edges": graph.edges,
        }
    )


def _validate_base(base: GECEGInstance) -> None:
    try:
        validate_instance_structure(base)
    except (AttributeError, TypeError, ValueError, ValidationError, GraphEngineError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_BASE_GRAPH_INVALID,
            "base snapshot violates the canonical graph structure boundary",
        ) from error


def _validate_scope(base: GECEGInstance, delta: GraphDelta) -> None:
    if str(delta.graph_code) != str(base.graph_code):
        _fail(GraphEngineErrorCode.DELTA_GRAPH_CODE_MISMATCH, "Delta targets another graph")
    if delta.base_version != base.graph_version:
        _fail(
            GraphEngineErrorCode.DELTA_BASE_VERSION_MISMATCH,
            "Delta base_version is not the current snapshot version",
        )
    for name in ("case_code", "graph_code"):
        if getattr(delta.scope, name) != getattr(base.scope, name):
            _fail(
                GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
                f"Delta scope.{name} differs from the base snapshot",
            )
    if base.scope.run_id is not None and delta.scope.run_id != base.scope.run_id:
        _fail(
            GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
            "Delta scope.run_id differs from the bound base snapshot",
        )
    if base.scope.run_id is None and delta.scope.run_id is not None and base.graph_version != 1:
        _fail(
            GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
            "only an unbound version-1 snapshot may acquire its explicit run scope",
        )
    for name in ("laterality", "coordinate_space", "unit"):
        if getattr(delta, name) != getattr(base, name):
            _fail(
                GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
                f"Delta {name} differs from the base snapshot",
            )


def _validate_rule_provenance(delta: GraphDelta) -> None:
    by_code = {str(reference.code): reference for reference in delta.provenance.rule_versions}
    missing = tuple(
        str(code)
        for code in delta.reason_codes
        if str(code) not in by_code or by_code[str(code)].checksum is None
    )
    if missing:
        _fail(
            GraphEngineErrorCode.DELTA_RULE_PROVENANCE_MISSING,
            "reason_codes lack checksum-bound rule provenance: " + ", ".join(missing),
        )


def _edge_key(edge: GraphEdge) -> tuple[str, str, object]:
    return (str(edge.source_node_code), str(edge.target_node_code), edge.edge_type)


def _validate_update_source(delta: GraphDelta, operation: UpdateNodeStatusOperation) -> None:
    if operation.source not in {
        GraphUpdateSource.VALIDATED_EVIDENCE,
        GraphUpdateSource.DETERMINISTIC_RULE,
    }:
        _fail(
            GraphEngineErrorCode.DELTA_UPDATE_SOURCE_INVALID,
            "only validated evidence or deterministic rules can update status",
        )
    if operation.source is GraphUpdateSource.DETERMINISTIC_RULE and delta.proposed_by not in {
        ActorType.RULE_ENGINE,
        ActorType.GRAPH_ENGINE,
    }:
        _fail(
            GraphEngineErrorCode.DELTA_UPDATE_SOURCE_INVALID,
            "only deterministic engine actors may claim a deterministic rule update",
        )


def _validate_operations(
    base: GECEGInstance, delta: GraphDelta
) -> tuple[tuple[GraphNodeInstance, ...], tuple[GraphEdge, ...]]:
    nodes = list(base.nodes)
    edges = list(base.edges)
    node_by_code = {str(node.node_code): node for node in nodes}
    edge_keys = {_edge_key(edge) for edge in edges}
    mutation_keys: set[tuple[str, object]] = set()
    allowed_rule_refs = {_reference_key(reference) for reference in delta.provenance.rule_versions}

    for operation in delta.operations:
        if isinstance(operation, AddNodeOperation):
            code = str(operation.node.node_code)
            mutation_key = ("NODE", code)
            if mutation_key in mutation_keys:
                _fail(GraphEngineErrorCode.DELTA_DUPLICATE_OPERATION, f"node {code} repeated")
            mutation_keys.add(mutation_key)
            if code in node_by_code:
                _fail(
                    GraphEngineErrorCode.DELTA_FACT_OVERWRITE_FORBIDDEN,
                    f"ADD_NODE cannot overwrite {code}",
                )
            if (
                operation.node.node_type is NodeType.GOAL
                or operation.node.status not in {GraphNodeStatus.UNKNOWN, GraphNodeStatus.PENDING}
                or operation.node.evidence_bindings
            ):
                _fail(
                    GraphEngineErrorCode.DELTA_FACT_OVERWRITE_FORBIDDEN,
                    "ADD_NODE may not publish a goal, derived fact, or evidence binding",
                )
            nodes.append(operation.node)
            node_by_code[code] = operation.node
            continue

        if isinstance(operation, AddEdgeOperation):
            edge = operation.edge
            key = _edge_key(edge)
            mutation_key = ("EDGE", key)
            if mutation_key in mutation_keys:
                _fail(GraphEngineErrorCode.DELTA_DUPLICATE_OPERATION, f"edge {key} repeated")
            mutation_keys.add(mutation_key)
            source, target, _ = key
            if source not in node_by_code or target not in node_by_code:
                _fail(
                    GraphEngineErrorCode.DELTA_UNKNOWN_EDGE_ENDPOINT,
                    f"edge references an unknown endpoint: {source}->{target}",
                )
            if source == target:
                _fail(GraphEngineErrorCode.DELTA_SELF_REFERENCE, f"self edge at {source}")
            if key in edge_keys:
                _fail(
                    GraphEngineErrorCode.DELTA_FACT_OVERWRITE_FORBIDDEN,
                    f"ADD_EDGE cannot overwrite {key}",
                )
            allowed_sources, allowed_targets = _EDGE_ENDPOINTS[edge.edge_type]
            if (
                node_by_code[source].node_type not in allowed_sources
                or node_by_code[target].node_type not in allowed_targets
            ):
                _fail(
                    GraphEngineErrorCode.DELTA_EDGE_NODE_TYPE_MISMATCH,
                    f"{edge.edge_type.value} forbids the proposed endpoint types",
                )
            if edge.rule_ref is None or _reference_key(edge.rule_ref) not in allowed_rule_refs:
                _fail(
                    GraphEngineErrorCode.DELTA_RULE_PROVENANCE_MISSING,
                    "ADD_EDGE requires a checksum-bound rule_ref in Delta provenance",
                )
            edges.append(edge)
            edge_keys.add(key)
            continue

        if isinstance(operation, UpdateNodeStatusOperation):
            code = str(operation.node_code)
            mutation_key = ("NODE", code)
            if mutation_key in mutation_keys:
                _fail(GraphEngineErrorCode.DELTA_DUPLICATE_OPERATION, f"node {code} repeated")
            mutation_keys.add(mutation_key)
            _validate_update_source(delta, operation)
            current = node_by_code.get(code)
            if current is None:
                _fail(GraphEngineErrorCode.DELTA_UNKNOWN_NODE, f"unknown node {code}")
            if current.status is not operation.from_status:
                _fail(
                    GraphEngineErrorCode.DELTA_STATUS_FROM_MISMATCH,
                    f"{code} current status differs from from_status",
                )
            if operation.from_status is operation.to_status:
                _fail(GraphEngineErrorCode.DELTA_STATUS_NOOP, f"{code} status update is a no-op")
            if not can_transition(
                operation.from_status, operation.to_status, allow_idempotent=False
            ):
                _fail(
                    GraphEngineErrorCode.DELTA_STATUS_TRANSITION_INVALID,
                    f"illegal transition for {code}",
                )
            if operation.source is GraphUpdateSource.VALIDATED_EVIDENCE:
                if operation.evidence_binding is None:
                    _fail(
                        GraphEngineErrorCode.DELTA_EVIDENCE_BINDING_REQUIRED,
                        f"validated evidence update for {code} requires a binding",
                    )
                if operation.evidence_binding not in current.evidence_bindings:
                    _fail(
                        GraphEngineErrorCode.DELTA_EVIDENCE_BINDING_UNKNOWN,
                        f"binding is not declared on node {code}",
                    )
            elif operation.evidence_binding is not None:
                _fail(
                    GraphEngineErrorCode.DELTA_EVIDENCE_BINDING_UNKNOWN,
                    "deterministic rule updates cannot claim an evidence binding",
                )
            updated = current.model_copy(update={"status": operation.to_status})
            nodes[nodes.index(current)] = updated
            node_by_code[code] = updated
            continue

        _fail(GraphEngineErrorCode.DELTA_CONTRACT_INVALID, "operation is not whitelisted")

    unreachable = _unreachable_from_goal(set(node_by_code), tuple(edges), str(base.goal_node_code))
    if unreachable:
        _fail(
            GraphEngineErrorCode.DELTA_ORPHANED_GRAPH,
            "result has nodes disconnected from the goal: " + ", ".join(unreachable),
        )
    return tuple(nodes), tuple(edges)


def _validate_evidence_proof(
    delta: GraphDelta, validation_result: ValidationResult | None
) -> tuple[ValidationResult | None, str | None]:
    evidence_operations = tuple(
        operation
        for operation in delta.operations
        if isinstance(operation, UpdateNodeStatusOperation)
        and operation.source is GraphUpdateSource.VALIDATED_EVIDENCE
    )
    if not evidence_operations:
        if validation_result is not None:
            _fail(
                GraphEngineErrorCode.DELTA_VALIDATION_MISMATCH,
                "ValidationResult was supplied for a Delta without validated evidence",
            )
        return None, None
    if validation_result is None:
        _fail(
            GraphEngineErrorCode.DELTA_VALIDATION_REQUIRED,
            "validated evidence Delta requires its ValidationResult",
        )
    try:
        validated = ValidationResult.model_validate(validation_result.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_VALIDATION_MISMATCH,
            "ValidationResult violates packages.contracts v1",
        ) from error
    if validated.graph_delta_candidate != delta:
        _fail(
            GraphEngineErrorCode.DELTA_VALIDATION_MISMATCH,
            "ValidationResult does not bind the exact candidate Delta",
        )
    return validated, canonical_checksum(validated)


def validate_graph_delta(
    base: GECEGInstance,
    delta: GraphDelta,
    *,
    validation_result: ValidationResult | None = None,
) -> ValidatedGraphDelta:
    """Validate one candidate without mutating the base or creating a version."""

    _validate_base(base)
    try:
        if type(delta) is not GraphDelta:
            raise TypeError("Delta must be GraphDelta")
        _validate_scope(base, delta)
        for operation in delta.operations:
            if isinstance(operation, UpdateNodeStatusOperation):
                _validate_update_source(delta, operation)
                if operation.from_status is operation.to_status:
                    _fail(
                        GraphEngineErrorCode.DELTA_STATUS_NOOP,
                        f"{operation.node_code} status update is a no-op",
                    )
                if not can_transition(
                    operation.from_status, operation.to_status, allow_idempotent=False
                ):
                    _fail(
                        GraphEngineErrorCode.DELTA_STATUS_TRANSITION_INVALID,
                        f"illegal transition for {operation.node_code}",
                    )
        delta = GraphDelta.model_validate(delta.model_dump(mode="python"))
    except GraphEngineError:
        raise
    except (AttributeError, ValidationError, TypeError, ValueError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_CONTRACT_INVALID,
            "Delta violates packages.contracts v1",
        ) from error
    validate_transition_reset_metadata(delta)
    _validate_rule_provenance(delta)
    _validate_operations(base, delta)
    _, validation_checksum = _validate_evidence_proof(delta, validation_result)
    return ValidatedGraphDelta(
        delta=delta,
        base_checksum=canonical_checksum(base),
        base_structure_hash=graph_structure_hash(base),
        delta_checksum=canonical_checksum(delta),
        validation_checksum=validation_checksum,
    )


def _result_provenance(validated: ValidatedGraphDelta, base: GECEGInstance) -> Provenance:
    delta = validated.delta
    input_checksums = [validated.base_checksum, validated.delta_checksum]
    if validated.validation_checksum is not None:
        input_checksums.append(validated.validation_checksum)
    return Provenance(
        actor=ActorType.GRAPH_ENGINE,
        created_at=delta.provenance.created_at,
        generator_version=DELTA_ENGINE_VERSION,
        source_codes=tuple(
            sorted(set(base.provenance.source_codes) | set(delta.provenance.source_codes))
        ),
        rule_versions=_normalized_references(
            base.provenance.rule_versions, delta.provenance.rule_versions
        ),
        knowledge_versions=_normalized_references(
            base.provenance.knowledge_versions, delta.provenance.knowledge_versions
        ),
        tool_versions=_normalized_references(
            base.provenance.tool_versions, delta.provenance.tool_versions
        ),
        input_checksums=tuple(input_checksums),
    )


def _build_result(base: GECEGInstance, validated: ValidatedGraphDelta) -> GECEGInstance:
    nodes, edges = _validate_operations(base, validated.delta)
    data = base.model_dump(mode="python")
    data.update(
        graph_version=base.graph_version + 1,
        scope=validated.delta.scope,
        nodes=nodes,
        edges=edges,
        provenance=_result_provenance(validated, base),
    )
    return GECEGInstance.model_validate(data)


def build_delta_plan_checksum_payload(
    validated: ValidatedGraphDelta,
    *,
    result_version: int,
    result_checksum: str,
    result_structure_hash: str,
) -> dict[str, JsonValue]:
    """Return the sole canonical field list used to identify a 03B plan."""

    delta = validated.delta
    return {
        "payload_version": DELTA_PLAN_PAYLOAD_VERSION,
        "engine_version": DELTA_ENGINE_VERSION,
        "graph_code": str(delta.graph_code),
        "base_version": delta.base_version,
        "result_version": result_version,
        "idempotency_key": str(delta.idempotency_key),
        "base_checksum": validated.base_checksum,
        "base_structure_hash": validated.base_structure_hash,
        "delta_checksum": validated.delta_checksum,
        "validation_checksum": validated.validation_checksum,
        "result_checksum": result_checksum,
        "result_structure_hash": result_structure_hash,
    }


def _accepted_event_payload(
    event: AcceptedDeltaEvent, *, include_event_checksum: bool
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "event_version": event.event_version,
        "graph_code": event.graph_code,
        "base_version": event.base_version,
        "result_version": event.result_version,
        "idempotency_key": event.idempotency_key,
        "base_checksum": event.base_checksum,
        "base_structure_hash": event.base_structure_hash,
        "delta_checksum": event.delta_checksum,
        "validation_checksum": event.validation_checksum,
        "result_checksum": event.result_checksum,
        "result_structure_hash": event.result_structure_hash,
        "plan_checksum": event.plan_checksum,
        "delta": event.delta.model_dump(mode="json"),
        "validation_result": (
            event.validation_result.model_dump(mode="json")
            if event.validation_result is not None
            else None
        ),
    }
    if include_event_checksum:
        payload["event_checksum"] = event.event_checksum
    return payload


def _validate_checksum(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _CHECKSUM_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical SHA-256 checksum")
    return value


def _strict_json_value(value: object, *, path: str = "$") -> JsonValue:
    """Validate raw JSON without accepting Python/Pydantic type coercion."""

    if value is None or type(value) in {bool, int, str}:
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite JSON number")
        return value
    if type(value) is list:
        return [
            _strict_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} JSON object keys must be strings")
        return {key: _strict_json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    raise TypeError(f"{path} contains non-JSON type {type(value).__name__}")


def _strict_canonical_json_text(value: object) -> str:
    """Encode strict JSON with type-sensitive scalars and key-order independence."""

    return json.dumps(
        _strict_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_accepted_event(event: AcceptedDeltaEvent) -> None:
    if type(event) is not AcceptedDeltaEvent:
        raise TypeError("accepted event must be AcceptedDeltaEvent")
    if event.event_version != ACCEPTED_DELTA_EVENT_VERSION:
        raise ValueError("accepted event version mismatch")
    if type(event.base_version) is not int or type(event.result_version) is not int:
        raise TypeError("accepted event versions must be integers")
    if event.result_version != event.base_version + 1:
        raise ValueError("accepted event result_version must equal base_version + 1")
    for field in (
        "base_checksum",
        "base_structure_hash",
        "delta_checksum",
        "result_checksum",
        "result_structure_hash",
        "plan_checksum",
        "event_checksum",
    ):
        _validate_checksum(getattr(event, field), field=field)
    if event.validation_checksum is not None:
        _validate_checksum(event.validation_checksum, field="validation_checksum")
    delta = GraphDelta.model_validate(event.delta.model_dump(mode="python"))
    validate_transition_reset_metadata(delta)
    validation = (
        ValidationResult.model_validate(event.validation_result.model_dump(mode="python"))
        if event.validation_result is not None
        else None
    )
    if event.graph_code != str(delta.graph_code):
        raise ValueError("accepted event graph_code differs from Delta")
    if event.base_version != delta.base_version:
        raise ValueError("accepted event base_version differs from Delta")
    if event.idempotency_key != str(delta.idempotency_key):
        raise ValueError("accepted event idempotency_key differs from Delta")
    if canonical_checksum(delta) != event.delta_checksum:
        raise ValueError("accepted event Delta checksum mismatch")
    expected_validation_checksum = (
        canonical_checksum(validation) if validation is not None else None
    )
    if expected_validation_checksum != event.validation_checksum:
        raise ValueError("accepted event ValidationResult checksum mismatch")
    expected_event_checksum = canonical_checksum(
        _accepted_event_payload(event, include_event_checksum=False)
    )
    if expected_event_checksum != event.event_checksum:
        raise ValueError("accepted event checksum mismatch")


def accepted_delta_event_to_payload(event: AcceptedDeltaEvent) -> dict[str, JsonValue]:
    """Strictly serialize one replay event into a canonical JSON-compatible object."""

    try:
        _validate_accepted_event(event)
        payload = _accepted_event_payload(event, include_event_checksum=True)
        _strict_canonical_json_text(payload)
        canonical_checksum(payload)
        return payload
    except (TypeError, ValueError, ValidationError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID,
            "accepted event cannot be serialized canonically",
        ) from error


_ACCEPTED_EVENT_FIELDS = frozenset(
    {
        "event_version",
        "graph_code",
        "base_version",
        "result_version",
        "idempotency_key",
        "base_checksum",
        "base_structure_hash",
        "delta_checksum",
        "validation_checksum",
        "result_checksum",
        "result_structure_hash",
        "plan_checksum",
        "delta",
        "validation_result",
        "event_checksum",
    }
)


def accepted_delta_event_from_payload(payload: object) -> AcceptedDeltaEvent:
    """Strictly parse a persisted canonical event; extra/coerced fields are forbidden."""

    try:
        if type(payload) is not dict:
            raise TypeError("accepted event payload must be a JSON object")
        if set(payload) != _ACCEPTED_EVENT_FIELDS:
            raise ValueError("accepted event payload fields do not exactly match its version")
        for field in ("event_version", "graph_code", "idempotency_key"):
            if type(payload[field]) is not str:
                raise TypeError(f"{field} must be a string")
        for field in ("base_version", "result_version"):
            if type(payload[field]) is not int:
                raise TypeError(f"{field} must be an integer")
        for field in (
            "base_checksum",
            "base_structure_hash",
            "delta_checksum",
            "result_checksum",
            "result_structure_hash",
            "plan_checksum",
            "event_checksum",
        ):
            _validate_checksum(payload[field], field=field)
        if payload["validation_checksum"] is not None:
            _validate_checksum(payload["validation_checksum"], field="validation_checksum")
        if type(payload["delta"]) is not dict:
            raise TypeError("delta must be a JSON object")
        if (
            payload["validation_result"] is not None
            and type(payload["validation_result"]) is not dict
        ):
            raise TypeError("validation_result must be a JSON object or null")
        _strict_canonical_json_text(payload)
        delta = GraphDelta.model_validate(payload["delta"])
        validation = (
            ValidationResult.model_validate(payload["validation_result"])
            if payload["validation_result"] is not None
            else None
        )
        if _strict_canonical_json_text(payload["delta"]) != _strict_canonical_json_text(
            delta.model_dump(mode="json")
        ):
            raise ValueError("delta is not represented as canonical contract JSON")
        if validation is not None and _strict_canonical_json_text(
            payload["validation_result"]
        ) != _strict_canonical_json_text(validation.model_dump(mode="json")):
            raise ValueError("validation_result is not represented as canonical contract JSON")
        canonical_checksum(payload)
        event = AcceptedDeltaEvent(
            event_version=payload["event_version"],
            graph_code=payload["graph_code"],
            base_version=payload["base_version"],
            result_version=payload["result_version"],
            idempotency_key=payload["idempotency_key"],
            base_checksum=payload["base_checksum"],
            base_structure_hash=payload["base_structure_hash"],
            delta_checksum=payload["delta_checksum"],
            validation_checksum=payload["validation_checksum"],
            result_checksum=payload["result_checksum"],
            result_structure_hash=payload["result_structure_hash"],
            plan_checksum=payload["plan_checksum"],
            delta=delta,
            validation_result=validation,
            event_checksum=payload["event_checksum"],
        )
        _validate_accepted_event(event)
        return event
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID,
            "accepted event payload is not strict canonical 03B JSON",
        ) from error


def _build_accepted_event(
    validated: ValidatedGraphDelta,
    validation_result: ValidationResult | None,
    *,
    result_version: int,
    result_checksum: str,
    result_structure_hash: str,
    plan_checksum: str,
) -> AcceptedDeltaEvent:
    delta = validated.delta
    event = AcceptedDeltaEvent(
        event_version=ACCEPTED_DELTA_EVENT_VERSION,
        graph_code=str(delta.graph_code),
        base_version=delta.base_version,
        result_version=result_version,
        idempotency_key=str(delta.idempotency_key),
        base_checksum=validated.base_checksum,
        base_structure_hash=validated.base_structure_hash,
        delta_checksum=validated.delta_checksum,
        validation_checksum=validated.validation_checksum,
        result_checksum=result_checksum,
        result_structure_hash=result_structure_hash,
        plan_checksum=plan_checksum,
        delta=delta,
        validation_result=validation_result,
        event_checksum="sha256:" + "0" * 64,
    )
    return replace(
        event,
        event_checksum=canonical_checksum(
            _accepted_event_payload(event, include_event_checksum=False)
        ),
    )


def plan_graph_delta(
    base: GECEGInstance,
    delta: GraphDelta,
    *,
    validation_result: ValidationResult | None = None,
    accepted_events: tuple[AcceptedDeltaEvent, ...] = (),
) -> GraphDeltaPlan:
    """Create a complete deterministic plan and replay event, but persist nothing."""

    validated = validate_graph_delta(base, delta, validation_result=validation_result)
    canonical_validation, validation_checksum = _validate_evidence_proof(
        validated.delta, validation_result
    )
    if validation_checksum != validated.validation_checksum:
        _fail(
            GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID,
            "validation checksum changed during canonical planning",
        )
    result = _build_result(base, validated)
    result_checksum = canonical_checksum(result)
    result_hash = graph_structure_hash(result)
    plan_payload = build_delta_plan_checksum_payload(
        validated,
        result_version=result.graph_version,
        result_checksum=result_checksum,
        result_structure_hash=result_hash,
    )
    plan_checksum = canonical_checksum(plan_payload)
    event = _build_accepted_event(
        validated,
        canonical_validation,
        result_version=result.graph_version,
        result_checksum=result_checksum,
        result_structure_hash=result_hash,
        plan_checksum=plan_checksum,
    )
    retry = False
    for accepted in accepted_events:
        accepted_delta_event_to_payload(accepted)
        if accepted.idempotency_key != str(validated.delta.idempotency_key):
            continue
        if accepted != event:
            _fail(
                GraphEngineErrorCode.DELTA_IDEMPOTENCY_CONFLICT,
                "idempotency key is already bound to different content",
            )
        retry = True
    return GraphDeltaPlan(
        payload_version=DELTA_PLAN_PAYLOAD_VERSION,
        validated=validated,
        validation_result=canonical_validation,
        result_instance=result,
        result_checksum=result_checksum,
        result_structure_hash=result_hash,
        plan_checksum=plan_checksum,
        accepted_event=event,
        idempotent_retry=retry,
    )


def apply_graph_delta(
    base: GECEGInstance,
    plan: GraphDeltaPlan,
    *,
    accepted_events: tuple[AcceptedDeltaEvent, ...] = (),
) -> GraphDeltaApplicationResult:
    """Treat the supplied plan as untrusted and rederive its sole legal form from inputs."""

    try:
        if type(plan) is not GraphDeltaPlan:
            raise TypeError("plan must be GraphDeltaPlan")
        rebuilt = plan_graph_delta(
            base,
            plan.validated.delta,
            validation_result=plan.validation_result,
            accepted_events=accepted_events,
        )
    except (AttributeError, TypeError, ValueError, ValidationError, GraphEngineError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID,
            "untrusted plan cannot be rebuilt from its base, Delta, and validation proof",
        ) from error
    if plan != rebuilt:
        _fail(
            GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID,
            "untrusted plan differs from the uniquely rebuilt canonical plan",
        )
    return GraphDeltaApplicationResult(
        instance=rebuilt.result_instance,
        result_checksum=rebuilt.result_checksum,
        structure_hash=rebuilt.result_structure_hash,
        delta_checksum=rebuilt.validated.delta_checksum,
        accepted_event=rebuilt.accepted_event,
        idempotent_retry=rebuilt.idempotent_retry,
    )


def replay_accepted_deltas(
    snapshot: GECEGInstance, events: tuple[AcceptedDeltaEvent, ...]
) -> GECEGInstance:
    """Replay an ordered accepted-event stream and verify every recorded checksum."""

    current = snapshot
    for supplied_event in events:
        try:
            event = accepted_delta_event_from_payload(
                accepted_delta_event_to_payload(supplied_event)
            )
            if (
                event.graph_code != str(current.graph_code)
                or event.base_version != current.graph_version
                or event.result_version != current.graph_version + 1
            ):
                raise ValueError("accepted event is not contiguous with the replay snapshot")
            plan = plan_graph_delta(
                current,
                event.delta,
                validation_result=event.validation_result,
            )
            if plan.accepted_event != event:
                raise ValueError("accepted event differs from deterministic replay")
            application = apply_graph_delta(current, plan)
            if (
                application.result_checksum != event.result_checksum
                or application.structure_hash != event.result_structure_hash
            ):
                raise ValueError("replayed result checksums differ from accepted event")
            current = application.instance
        except (TypeError, ValueError, ValidationError, GraphEngineError) as error:
            raise GraphEngineError(
                GraphEngineErrorCode.DELTA_REPLAY_MISMATCH,
                "accepted event failed strict deterministic replay",
            ) from error
    return current
