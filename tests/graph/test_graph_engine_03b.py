"""Stage 03B deterministic GraphDelta validation, planning, application, and replay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from packages.contracts import (
    ActorType,
    AddEdgeOperation,
    AddNodeOperation,
    DataAvailability,
    EdgeType,
    GECEGInstance,
    GraphDelta,
    GraphEdge,
    GraphNodeInstance,
    GraphNodeStatus,
    GraphUpdateSource,
    NodeType,
    ProposalKind,
    UpdateNodeStatusOperation,
    ValidationResult,
    VersionedReference,
)
from packages.graph_engine import (
    ACCEPTED_DELTA_EVENT_VERSION,
    GraphEngineError,
    GraphEngineErrorCode,
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    apply_graph_delta,
    canonical_checksum,
    graph_structure_hash,
    plan_graph_delta,
    replay_accepted_deltas,
    validate_graph_delta,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "packages" / "contracts" / "examples" / "v1"
SHA = "sha256:" + "a" * 64


def _example(model_type: type, filename: str):
    return model_type.model_validate(json.loads((EXAMPLES / filename).read_text()))


def _rule_ref(code: str = "RULE:SYNTHETIC_RULE") -> VersionedReference:
    return VersionedReference(code=code, version="1.0.0", checksum=SHA)


def _base() -> GECEGInstance:
    return _example(GECEGInstance, "GECEGInstance.valid.json")


def _candidate(
    *,
    base_version: int = 1,
    idempotency_key: str = "TASK:OID_DELTA001",
    to_status: GraphNodeStatus = GraphNodeStatus.SUPPORTED,
) -> GraphDelta:
    data = _example(GraphDelta, "GraphDelta.valid.json").model_dump(mode="python")
    data["base_version"] = base_version
    data["idempotency_key"] = idempotency_key
    data["provenance"]["rule_versions"] = (_rule_ref(),)
    operation = data["operations"][0]
    operation["to_status"] = to_status
    return GraphDelta.model_validate(data)


def _validation(delta: GraphDelta) -> ValidationResult:
    data = _example(ValidationResult, "ValidationResult.valid.json").model_dump(mode="python")
    data["graph_delta_candidate"] = delta
    data["provenance"]["rule_versions"] = (_rule_ref(),)
    operation = delta.operations[0]
    if isinstance(operation, UpdateNodeStatusOperation) and operation.evidence_binding:
        data["evidence_bindings"] = (operation.evidence_binding,)
    return ValidationResult.model_validate(data)


def _rule_delta(
    operations: tuple,
    *,
    base_version: int = 1,
    idempotency_key: str = "TASK:OID_RULEDELTA1",
) -> GraphDelta:
    candidate = _candidate(base_version=base_version, idempotency_key=idempotency_key)
    data = candidate.model_dump(mode="python")
    data.update(
        proposed_by=ActorType.RULE_ENGINE,
        proposal_kind=ProposalKind.GRAPH_DELTA_VALIDATION,
        operations=operations,
    )
    data["provenance"]["actor"] = ActorType.RULE_ENGINE
    return GraphDelta.model_validate(data)


def _error_code(error: pytest.ExceptionInfo[GraphEngineError]) -> GraphEngineErrorCode:
    return error.value.code


def test_legal_candidate_validate_plan_apply_is_deterministic_and_immutable() -> None:
    base = _base()
    before = base.model_dump(mode="json")
    delta = _candidate()
    validation = _validation(delta)

    validated = validate_graph_delta(base, delta, validation_result=validation)
    first_plan = plan_graph_delta(base, delta, validation_result=validation)
    second_plan = plan_graph_delta(base, delta, validation_result=validation)
    first = apply_graph_delta(base, first_plan)
    second = apply_graph_delta(base, second_plan)

    assert validated.delta == delta
    assert first_plan == second_plan
    assert first == second
    assert first.instance.graph_version == 2
    assert first.result_checksum == canonical_checksum(first.instance)
    assert first.structure_hash == graph_structure_hash(first.instance)
    assert first.instance.nodes[1].status is GraphNodeStatus.SUPPORTED
    assert base.model_dump(mode="json") == before
    assert GECEGInstance.model_validate(first.instance.model_dump(mode="json")) == first.instance


def test_legal_multi_operation_delta_preserves_operation_order() -> None:
    data_node = GraphNodeInstance(
        node_code="ARTIFACT:SYNTHETIC_DELTA",
        node_type=NodeType.DATA,
        concept_code="CONCEPT:SYNTHETIC_DELTA",
        status=GraphNodeStatus.PENDING,
        availability=DataAvailability.DERIVABLE,
    )
    tool_node = GraphNodeInstance(
        node_code="TOOL:SYNTHETIC_DELTA",
        node_type=NodeType.TOOL,
        concept_code="CONCEPT:SYNTHETIC_DELTA_TOOL",
        status=GraphNodeStatus.PENDING,
        availability=DataAvailability.UNAVAILABLE,
    )
    delta = _rule_delta(
        (
            AddNodeOperation(op="ADD_NODE", node=data_node),
            AddNodeOperation(op="ADD_NODE", node=tool_node),
            AddEdgeOperation(
                op="ADD_EDGE",
                edge=GraphEdge(
                    source_node_code=data_node.node_code,
                    target_node_code="GOAL:SYNTHETIC_SCREEN",
                    edge_type=EdgeType.SUPPORTS,
                    rule_ref=_rule_ref(),
                ),
            ),
            AddEdgeOperation(
                op="ADD_EDGE",
                edge=GraphEdge(
                    source_node_code=data_node.node_code,
                    target_node_code=tool_node.node_code,
                    edge_type=EdgeType.REQUIRES_INPUT,
                    rule_ref=_rule_ref(),
                ),
            ),
        )
    )
    result = apply_graph_delta(_base(), plan_graph_delta(_base(), delta))
    assert tuple(node.node_code for node in result.instance.nodes[-2:]) == (
        data_node.node_code,
        tool_node.node_code,
    )
    assert tuple(edge.edge_type for edge in result.instance.edges[-2:]) == (
        EdgeType.SUPPORTS,
        EdgeType.REQUIRES_INPUT,
    )


@pytest.mark.parametrize(
    ("delta", "code"),
    (
        (_candidate(base_version=2), GraphEngineErrorCode.DELTA_BASE_VERSION_MISMATCH),
        (
            _candidate().model_copy(update={"graph_code": "GRAPH:OID_OTHER001"}),
            GraphEngineErrorCode.DELTA_GRAPH_CODE_MISMATCH,
        ),
        (
            _candidate().model_copy(
                update={
                    "scope": _candidate().scope.model_copy(update={"run_id": "RUN:OID_OTHER001"})
                }
            ),
            GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
        ),
    ),
)
def test_graph_base_and_scope_are_bound(delta: GraphDelta, code: GraphEngineErrorCode) -> None:
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(_base(), delta, validation_result=None)
    assert _error_code(error) is code


def test_validated_evidence_requires_matching_validation_and_existing_binding() -> None:
    delta = _candidate()
    with pytest.raises(GraphEngineError) as missing:
        plan_graph_delta(_base(), delta)
    assert _error_code(missing) is GraphEngineErrorCode.DELTA_VALIDATION_REQUIRED

    bad_operation = delta.operations[0].model_copy(
        update={"evidence_binding": "BINDING:OID_UNKNOWN01"}
    )
    bad_delta = delta.model_copy(update={"operations": (bad_operation,)})
    with pytest.raises(GraphEngineError) as unknown:
        plan_graph_delta(_base(), bad_delta, validation_result=_validation(bad_delta))
    assert _error_code(unknown) is GraphEngineErrorCode.DELTA_EVIDENCE_BINDING_UNKNOWN


def test_reason_codes_require_checksum_bound_rule_provenance() -> None:
    delta = _candidate().model_copy(
        update={"provenance": _candidate().provenance.model_copy(update={"rule_versions": ()})}
    )
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(_base(), delta, validation_result=_validation(delta))
    assert _error_code(error) is GraphEngineErrorCode.DELTA_RULE_PROVENANCE_MISSING


def test_from_status_noop_jump_and_illegal_source_are_rejected() -> None:
    base = _base()
    delta = _candidate()
    wrong_from = delta.operations[0].model_copy(
        update={
            "from_status": GraphNodeStatus.SUPPORTED,
            "to_status": GraphNodeStatus.REFUTED,
        }
    )
    with pytest.raises(GraphEngineError) as mismatch:
        plan_graph_delta(
            base,
            delta.model_copy(update={"operations": (wrong_from,)}),
            validation_result=_validation(delta.model_copy(update={"operations": (wrong_from,)})),
        )
    assert _error_code(mismatch) is GraphEngineErrorCode.DELTA_STATUS_FROM_MISMATCH

    no_op = delta.operations[0].model_copy(update={"to_status": GraphNodeStatus.PENDING})
    no_op_delta = delta.model_copy(update={"operations": (no_op,)})
    with pytest.raises(GraphEngineError) as same:
        plan_graph_delta(base, no_op_delta, validation_result=_validation(no_op_delta))
    assert _error_code(same) is GraphEngineErrorCode.DELTA_STATUS_NOOP

    jump = UpdateNodeStatusOperation.model_construct(
        op="UPDATE_NODE_STATUS",
        node_code="EVIDENCE:IMAGE_USABLE",
        from_status=GraphNodeStatus.PENDING,
        to_status=GraphNodeStatus.PENDING,
        source=GraphUpdateSource.TOOL_EXECUTION,
        evidence_binding=None,
    )
    malformed = delta.model_copy(update={"operations": (jump,)})
    with pytest.raises(GraphEngineError) as source:
        plan_graph_delta(base, malformed)
    assert _error_code(source) is GraphEngineErrorCode.DELTA_UPDATE_SOURCE_INVALID

    base_unknown_node = base.nodes[1].model_copy(update={"status": GraphNodeStatus.UNKNOWN})
    base_unknown = base.model_copy(update={"nodes": (base.nodes[0], base_unknown_node)})
    jump = UpdateNodeStatusOperation.model_construct(
        op="UPDATE_NODE_STATUS",
        node_code="EVIDENCE:IMAGE_USABLE",
        from_status=GraphNodeStatus.UNKNOWN,
        to_status=GraphNodeStatus.SUPPORTED,
        source=GraphUpdateSource.DETERMINISTIC_RULE,
        evidence_binding=None,
    )
    valid_rule_delta = _rule_delta(
        (
            UpdateNodeStatusOperation(
                op="UPDATE_NODE_STATUS",
                node_code="EVIDENCE:IMAGE_USABLE",
                from_status=GraphNodeStatus.PENDING,
                to_status=GraphNodeStatus.UNKNOWN,
                source=GraphUpdateSource.DETERMINISTIC_RULE,
            ),
        )
    )
    with pytest.raises(GraphEngineError) as skipped:
        plan_graph_delta(base_unknown, valid_rule_delta.model_copy(update={"operations": (jump,)}))
    assert _error_code(skipped) is GraphEngineErrorCode.DELTA_STATUS_TRANSITION_INVALID


def test_unknown_node_duplicate_operation_and_fact_overwrite_are_rejected() -> None:
    unknown = _rule_delta(
        (
            UpdateNodeStatusOperation(
                op="UPDATE_NODE_STATUS",
                node_code="EVIDENCE:UNKNOWN_NODE",
                from_status=GraphNodeStatus.PENDING,
                to_status=GraphNodeStatus.UNKNOWN,
                source=GraphUpdateSource.DETERMINISTIC_RULE,
            ),
        )
    )
    with pytest.raises(GraphEngineError) as missing:
        plan_graph_delta(_base(), unknown)
    assert _error_code(missing) is GraphEngineErrorCode.DELTA_UNKNOWN_NODE

    operation = _candidate().operations[0]
    duplicate = _candidate().model_copy(update={"operations": (operation, operation)})
    with pytest.raises(GraphEngineError) as repeated:
        plan_graph_delta(_base(), duplicate, validation_result=_validation(_candidate()))
    assert _error_code(repeated) is GraphEngineErrorCode.DELTA_DUPLICATE_OPERATION

    overwrite = _rule_delta((AddNodeOperation(op="ADD_NODE", node=_base().nodes[1]),))
    with pytest.raises(GraphEngineError) as existing:
        plan_graph_delta(_base(), overwrite)
    assert _error_code(existing) is GraphEngineErrorCode.DELTA_FACT_OVERWRITE_FORBIDDEN


def test_unknown_duplicate_illegal_and_self_edges_are_rejected() -> None:
    cases = (
        (
            GraphEdge(
                source_node_code="EVIDENCE:IMAGE_USABLE",
                target_node_code="GOAL:UNKNOWN",
                edge_type=EdgeType.SUPPORTS,
                rule_ref=_rule_ref(),
            ),
            GraphEngineErrorCode.DELTA_UNKNOWN_EDGE_ENDPOINT,
        ),
        (_base().edges[0], GraphEngineErrorCode.DELTA_FACT_OVERWRITE_FORBIDDEN),
        (
            GraphEdge(
                source_node_code="EVIDENCE:IMAGE_USABLE",
                target_node_code="EVIDENCE:IMAGE_USABLE",
                edge_type=EdgeType.SUPPORTS,
                rule_ref=_rule_ref(),
            ),
            GraphEngineErrorCode.DELTA_SELF_REFERENCE,
        ),
        (
            GraphEdge(
                source_node_code="EVIDENCE:IMAGE_USABLE",
                target_node_code="GOAL:SYNTHETIC_SCREEN",
                edge_type=EdgeType.REQUIRES_INPUT,
                rule_ref=_rule_ref(),
            ),
            GraphEngineErrorCode.DELTA_EDGE_NODE_TYPE_MISMATCH,
        ),
    )
    for edge, code in cases:
        with pytest.raises(GraphEngineError) as error:
            plan_graph_delta(_base(), _rule_delta((AddEdgeOperation(op="ADD_EDGE", edge=edge),)))
        assert _error_code(error) is code


def test_orphan_added_evidence_is_rejected_without_partial_result() -> None:
    orphan = GraphNodeInstance(
        node_code="EVIDENCE:ORPHAN_DELTA",
        node_type=NodeType.EVIDENCE,
        concept_code="CONCEPT:ORPHAN_DELTA",
        status=GraphNodeStatus.PENDING,
        availability=DataAvailability.UNAVAILABLE,
    )
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(_base(), _rule_delta((AddNodeOperation(op="ADD_NODE", node=orphan),)))
    assert _error_code(error) is GraphEngineErrorCode.DELTA_ORPHANED_GRAPH


def test_same_idempotency_key_same_content_retries_and_different_content_conflicts() -> None:
    base = _base()
    delta = _candidate()
    validation = _validation(delta)
    first = apply_graph_delta(base, plan_graph_delta(base, delta, validation_result=validation))
    retry_plan = plan_graph_delta(
        base,
        delta,
        validation_result=validation,
        accepted_events=(first.accepted_event,),
    )
    assert retry_plan.idempotent_retry is True
    assert apply_graph_delta(
        base,
        retry_plan,
        accepted_events=(first.accepted_event,),
    ) == replace(first, idempotent_retry=True)

    conflict = _candidate(to_status=GraphNodeStatus.REFUTED)
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(
            base,
            conflict,
            validation_result=_validation(conflict),
            accepted_events=(first.accepted_event,),
        )
    assert _error_code(error) is GraphEngineErrorCode.DELTA_IDEMPOTENCY_CONFLICT


def test_apply_rejects_a_different_same_version_base() -> None:
    base = _base()
    delta = _candidate()
    plan = plan_graph_delta(base, delta, validation_result=_validation(delta))
    changed_node = base.nodes[0].model_copy(update={"status": GraphNodeStatus.UNKNOWN})
    changed_base = base.model_copy(update={"nodes": (changed_node, *base.nodes[1:])})
    with pytest.raises(GraphEngineError) as error:
        apply_graph_delta(changed_base, plan)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID


def test_apply_rejects_synchronously_forged_result_status_and_hashes() -> None:
    base = _base()
    delta = _candidate()
    plan = plan_graph_delta(base, delta, validation_result=_validation(delta))
    forged_node = plan.result_instance.nodes[1].model_copy(
        update={"status": GraphNodeStatus.REFUTED}
    )
    forged_instance = plan.result_instance.model_copy(
        update={"nodes": (plan.result_instance.nodes[0], forged_node)}
    )
    forged_hash = graph_structure_hash(forged_instance)
    forged_event = replace(
        plan.accepted_event,
        result_structure_hash=forged_hash,
    )
    forged_plan = replace(
        plan,
        result_instance=forged_instance,
        result_structure_hash=forged_hash,
        accepted_event=forged_event,
    )

    with pytest.raises(GraphEngineError):
        apply_graph_delta(base, forged_plan)


def test_apply_rejects_forged_accepted_event_idempotency_key() -> None:
    base = _base()
    delta = _candidate()
    plan = plan_graph_delta(base, delta, validation_result=_validation(delta))
    forged_plan = replace(
        plan,
        accepted_event=replace(
            plan.accepted_event,
            idempotency_key="TASK:OID_FORGED001",
        ),
    )

    with pytest.raises(GraphEngineError):
        apply_graph_delta(base, forged_plan)


def _canonical_plan():
    base = _base()
    delta = _candidate()
    return base, plan_graph_delta(base, delta, validation_result=_validation(delta))


@pytest.mark.parametrize(
    "field",
    (
        "payload_version",
        "validated.base_checksum",
        "validated.base_structure_hash",
        "validated.delta_checksum",
        "validated.validation_checksum",
        "validated.delta",
        "validation_result",
        "result_instance",
        "result_structure_hash",
        "result_checksum",
        "plan_checksum",
        "accepted_event",
        "idempotent_retry",
    ),
)
def test_apply_rebuilds_and_rejects_every_tampered_plan_field(field: str) -> None:
    base, plan = _canonical_plan()
    alternate_checksum = "sha256:" + "b" * 64
    if field == "payload_version":
        tampered = replace(plan, payload_version="03B_DELTA_PLAN_FORGED")
    elif field.startswith("validated."):
        nested = field.split(".", 1)[1]
        value = (
            _candidate(to_status=GraphNodeStatus.REFUTED)
            if nested == "delta"
            else alternate_checksum
        )
        tampered = replace(
            plan,
            validated=replace(plan.validated, **{nested: value}),
        )
    elif field == "validation_result":
        tampered = replace(
            plan,
            validation_result=plan.validation_result.model_copy(update={"support_score": 0.1}),
        )
    elif field == "result_instance":
        forged_node = plan.result_instance.nodes[1].model_copy(
            update={"status": GraphNodeStatus.REFUTED}
        )
        tampered = replace(
            plan,
            result_instance=plan.result_instance.model_copy(
                update={"nodes": (plan.result_instance.nodes[0], forged_node)}
            ),
        )
    elif field == "accepted_event":
        tampered = replace(
            plan,
            accepted_event=replace(
                plan.accepted_event,
                idempotency_key="TASK:OID_FORGED001",
            ),
        )
    elif field == "idempotent_retry":
        tampered = replace(plan, idempotent_retry=True)
    else:
        tampered = replace(plan, **{field: alternate_checksum})

    with pytest.raises(GraphEngineError) as error:
        apply_graph_delta(base, tampered)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID


def test_apply_rederives_retry_flag_from_explicit_history() -> None:
    base, plan = _canonical_plan()
    retry_plan = plan_graph_delta(
        base,
        plan.validated.delta,
        validation_result=plan.validation_result,
        accepted_events=(plan.accepted_event,),
    )
    assert retry_plan.idempotent_retry is True
    with pytest.raises(GraphEngineError) as error:
        apply_graph_delta(
            base,
            replace(retry_plan, idempotent_retry=False),
            accepted_events=(plan.accepted_event,),
        )
    assert _error_code(error) is GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID


@pytest.mark.parametrize(
    "field",
    (
        "status",
        "node",
        "availability",
        "binding",
        "edge",
        "graph_identity",
        "graph_version",
        "extensions",
        "provenance",
    ),
)
def test_apply_rejects_every_tampered_result_component(field: str) -> None:
    base, plan = _canonical_plan()
    instance = plan.result_instance
    if field in {"status", "node", "availability", "binding"}:
        updates = {
            "status": {"status": GraphNodeStatus.REFUTED},
            "node": {"concept_code": "CONCEPT:FORGED_NODE"},
            "availability": {"availability": DataAvailability.UNAVAILABLE},
            "binding": {"evidence_bindings": ("BINDING:OID_FORGED01",)},
        }[field]
        forged_node = instance.nodes[1].model_copy(update=updates)
        forged = instance.model_copy(update={"nodes": (instance.nodes[0], forged_node)})
    elif field == "edge":
        forged_edge = instance.edges[0].model_copy(update={"edge_type": EdgeType.REFUTES})
        forged = instance.model_copy(update={"edges": (forged_edge,)})
    elif field == "graph_identity":
        forged = instance.model_copy(update={"graph_code": "GRAPH:OID_FORGED01"})
    elif field == "graph_version":
        forged = instance.model_copy(update={"graph_version": 3})
    elif field == "extensions":
        forged = instance.model_copy(update={"extensions": {"display_label": "forged"}})
    else:
        forged = instance.model_copy(
            update={
                "provenance": instance.provenance.model_copy(
                    update={"generator_version": "03B.FORGED"}
                )
            }
        )
    tampered = replace(plan, result_instance=forged)

    with pytest.raises(GraphEngineError) as error:
        apply_graph_delta(base, tampered)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID


@pytest.mark.parametrize(
    "field",
    (
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
    ),
)
def test_apply_rejects_every_tampered_accepted_event_field(field: str) -> None:
    base, plan = _canonical_plan()
    alternate_checksum = "sha256:" + "b" * 64
    values = {
        "event_version": "03B_ACCEPTED_DELTA_FORGED",
        "graph_code": "GRAPH:OID_FORGED01",
        "base_version": 2,
        "result_version": 3,
        "idempotency_key": "TASK:OID_FORGED001",
        "base_checksum": alternate_checksum,
        "base_structure_hash": alternate_checksum,
        "delta_checksum": alternate_checksum,
        "validation_checksum": alternate_checksum,
        "result_checksum": alternate_checksum,
        "result_structure_hash": alternate_checksum,
        "plan_checksum": alternate_checksum,
        "delta": _candidate(to_status=GraphNodeStatus.REFUTED),
        "validation_result": plan.validation_result.model_copy(update={"support_score": 0.1}),
        "event_checksum": alternate_checksum,
    }
    tampered = replace(
        plan,
        accepted_event=replace(plan.accepted_event, **{field: values[field]}),
    )

    with pytest.raises(GraphEngineError) as error:
        apply_graph_delta(base, tampered)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_PLAN_INTEGRITY_INVALID


def test_accepted_event_strict_json_round_trip_and_malformed_matrix() -> None:
    _, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    assert payload["event_version"] == ACCEPTED_DELTA_EVENT_VERSION
    assert accepted_delta_event_from_payload(payload) == plan.accepted_event

    malformed: list[object] = []
    missing = dict(payload)
    missing.pop("plan_checksum")
    malformed.append(missing)
    extra = dict(payload)
    extra["unknown"] = True
    malformed.append(extra)
    wrong_version = dict(payload)
    wrong_version["event_version"] = "03B_ACCEPTED_DELTA_FORGED"
    malformed.append(wrong_version)
    illegal_type = dict(payload)
    illegal_type["base_version"] = "1"
    malformed.append(illegal_type)
    forged_key = dict(payload)
    forged_key["idempotency_key"] = "TASK:OID_FORGED001"
    malformed.append(forged_key)
    forged_delta = json.loads(json.dumps(payload))
    forged_delta["delta"]["idempotency_key"] = "TASK:OID_FORGED001"
    malformed.append(forged_delta)
    forged_validation = json.loads(json.dumps(payload))
    forged_validation["validation_result"]["support_score"] = 0.1
    malformed.append(forged_validation)
    for field in (
        "base_checksum",
        "base_structure_hash",
        "delta_checksum",
        "validation_checksum",
        "result_checksum",
        "result_structure_hash",
        "plan_checksum",
        "event_checksum",
    ):
        changed = dict(payload)
        changed[field] = "sha256:" + "b" * 64
        malformed.append(changed)
    malformed.extend((None, [], object()))

    for value in malformed:
        with pytest.raises(GraphEngineError) as error:
            accepted_delta_event_from_payload(value)
        assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


def test_accepted_event_rejects_coerced_nested_delta_versions() -> None:
    _, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    payload["delta"]["base_version"] = "1"
    payload["validation_result"]["graph_delta_candidate"]["base_version"] = "1"

    with pytest.raises(GraphEngineError) as error:
        accepted_delta_event_from_payload(payload)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


@pytest.mark.parametrize("coerced", ("1", 1.0, True))
def test_accepted_event_rejects_all_coerced_nested_integer_types(coerced: object) -> None:
    _, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    payload["delta"]["base_version"] = coerced
    payload["validation_result"]["graph_delta_candidate"]["base_version"] = coerced

    with pytest.raises(GraphEngineError) as error:
        accepted_delta_event_from_payload(payload)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


def test_accepted_event_rejects_coerced_nested_validation_score() -> None:
    _, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    payload["validation_result"]["support_score"] = "0.9"

    with pytest.raises(GraphEngineError) as error:
        accepted_delta_event_from_payload(payload)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


@pytest.mark.parametrize(
    "path",
    (
        ("delta", "provenance", "created_at"),
        ("validation_result", "provenance", "created_at"),
    ),
)
def test_accepted_event_rejects_equivalent_noncanonical_provenance_time(
    path: tuple[str, str, str],
) -> None:
    _, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    target = payload[path[0]][path[1]]
    target[path[2]] = target[path[2]].replace("Z", "+00:00")

    with pytest.raises(GraphEngineError) as error:
        accepted_delta_event_from_payload(payload)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


def test_accepted_event_rejects_omitted_default_and_nonfinite_json_number() -> None:
    _, plan = _canonical_plan()
    payload_without_default = accepted_delta_event_to_payload(plan.accepted_event)
    payload_without_default["validation_result"].pop("refute_score")
    payload_with_nan = accepted_delta_event_to_payload(plan.accepted_event)
    payload_with_nan["delta"]["extensions"] = {"noncanonical_number": float("nan")}

    for payload in (payload_without_default, payload_with_nan):
        with pytest.raises(GraphEngineError) as error:
            accepted_delta_event_from_payload(payload)
        assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


def test_accepted_event_accepts_object_key_reordering_without_changing_arrays() -> None:
    base = _base()
    delta = _candidate()
    delta = delta.model_copy(
        update={
            "provenance": delta.provenance.model_copy(
                update={
                    "source_codes": (
                        "SOURCE:SYNTHETIC_01B",
                        "SOURCE:SYNTHETIC_03B",
                    )
                }
            )
        }
    )
    plan = plan_graph_delta(base, delta, validation_result=_validation(delta))
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    expected_array = list(payload["delta"]["provenance"]["source_codes"])
    payload["delta"] = dict(reversed(tuple(payload["delta"].items())))
    payload["validation_result"] = dict(reversed(tuple(payload["validation_result"].items())))
    reordered = dict(reversed(tuple(payload.items())))

    parsed = accepted_delta_event_from_payload(reordered)

    assert parsed == plan.accepted_event
    assert (
        list(parsed.delta.model_dump(mode="json")["provenance"]["source_codes"]) == expected_array
    )

    changed_array = json.loads(json.dumps(reordered))
    changed_array["delta"]["provenance"]["source_codes"].reverse()
    with pytest.raises(GraphEngineError) as error:
        accepted_delta_event_from_payload(changed_array)
    assert _error_code(error) is GraphEngineErrorCode.DELTA_EVENT_PAYLOAD_INVALID


@pytest.mark.parametrize("field", ("result_checksum", "result_structure_hash", "plan_checksum"))
def test_replay_rejects_self_consistently_resigned_result_or_plan_checksum(field: str) -> None:
    base, plan = _canonical_plan()
    payload = accepted_delta_event_to_payload(plan.accepted_event)
    payload[field] = "sha256:" + "b" * 64
    unsigned = dict(payload)
    unsigned.pop("event_checksum")
    payload["event_checksum"] = canonical_checksum(unsigned)
    event = accepted_delta_event_from_payload(payload)

    with pytest.raises(GraphEngineError) as error:
        replay_accepted_deltas(base, (event,))
    assert _error_code(error) is GraphEngineErrorCode.DELTA_REPLAY_MISMATCH


def test_accepted_delta_event_replay_matches_direct_application_hash() -> None:
    base = _base()
    first_delta = _candidate()
    first = apply_graph_delta(
        base,
        plan_graph_delta(base, first_delta, validation_result=_validation(first_delta)),
    )
    second_delta = _rule_delta(
        (
            UpdateNodeStatusOperation(
                op="UPDATE_NODE_STATUS",
                node_code="EVIDENCE:IMAGE_USABLE",
                from_status=GraphNodeStatus.SUPPORTED,
                to_status=GraphNodeStatus.REFUTED,
                source=GraphUpdateSource.DETERMINISTIC_RULE,
            ),
        ),
        base_version=2,
        idempotency_key="TASK:OID_RULEDELTA2",
    )
    second = apply_graph_delta(first.instance, plan_graph_delta(first.instance, second_delta))
    replayed = replay_accepted_deltas(base, (first.accepted_event, second.accepted_event))
    assert replayed == second.instance
    assert graph_structure_hash(replayed) == second.structure_hash


def test_public_graph_engine_import_tree_has_no_forbidden_runtime_dependency() -> None:
    package_root = ROOT / "packages" / "graph_engine"
    source = "\n".join(path.read_text() for path in sorted(package_root.glob("*.py"))).lower()
    for forbidden in ("from db", "import db", "openai", "agents.", "tool_executor"):
        assert forbidden not in source
