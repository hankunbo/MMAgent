"""Stage 03C deterministic disease-independent propagation tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apps.graph_versioning import GraphVersionAuditContext, build_graph_version_commit
from packages.contracts import (
    ActorType,
    CapabilityAvailability,
    CapabilityGap,
    ClinicalEffect,
    DataAvailability,
    EdgeType,
    EvidenceObservationStatus,
    EvidenceValidity,
    ExpertAssessment,
    ExpertAssessmentDecision,
    GECEGInstance,
    GraphEdge,
    GraphNodeInstance,
    GraphNodeStatus,
    NodeType,
    PlanAction,
    QCStatus,
    ToolExecutionStatus,
    ValidationResult,
    VersionedReference,
)
from packages.graph_engine import (
    CLINICAL_EDGE_TYPES,
    GraphEngineError,
    GraphEngineErrorCode,
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    apply_graph_delta,
    canonical_checksum,
    plan_graph_delta,
    replay_accepted_deltas,
)
from packages.rule_engine import (
    RULE_PACKAGE_PAYLOAD_VERSION,
    BoundCapabilityGap,
    BoundValidation,
    GateDefinition,
    LegalAction,
    PropagationBudget,
    PropagationContext,
    PropagationHistory,
    RuleEngineError,
    RuleEngineErrorCode,
    RulePackage,
    SnapshotBinding,
    bind_capability_gap,
    bind_validation,
    compute_rule_package_checksum,
    evaluate_propagation,
    propagate,
    rule_package_checksum_payload,
    snapshot_binding_for,
    transition_reset_audit_from_delta,
    transition_reset_audit_from_event_payload,
    validate_expert_action,
    validate_rule_package,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "packages" / "contracts" / "examples" / "v1"
SHA = "sha256:" + "a" * 64
RULE_REF = VersionedReference(code="RULE:SYNTHETIC_RULE", version="1.0.0", checksum=SHA)
_LAST_PACKAGE_GRAPH: GECEGInstance | None = None


def _example(model_type: type, filename: str):
    return model_type.model_validate(json.loads((EXAMPLES / filename).read_text()))


def _node(
    code: str,
    node_type: NodeType,
    status: GraphNodeStatus,
    availability: DataAvailability = DataAvailability.AVAILABLE,
) -> GraphNodeInstance:
    return GraphNodeInstance(
        node_code=code,
        node_type=node_type,
        concept_code=f"CONCEPT:{code.split(':')[-1]}",
        status=status,
        availability=availability,
    )


def _graph(
    nodes: tuple[GraphNodeInstance, ...],
    edges: tuple[GraphEdge, ...],
    *,
    goal: str = "GOAL:SYNTHETIC_SCREEN",
) -> GECEGInstance:
    base = _example(GECEGInstance, "GECEGInstance.valid.json")
    return GECEGInstance.model_validate(
        base.model_dump(mode="python") | {"goal_node_code": goal, "nodes": nodes, "edges": edges}
    )


def _gate_graph(
    operator: str,
    statuses: tuple[GraphNodeStatus, ...],
    *,
    availabilities: tuple[DataAvailability, ...] | None = None,
) -> tuple[GECEGInstance, GateDefinition]:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    availability = availabilities or tuple(DataAvailability.AVAILABLE for _ in statuses)
    children = tuple(
        _node(
            f"EVIDENCE:SYNTHETIC_{index}",
            NodeType.EVIDENCE,
            status,
            availability[index - 1],
        )
        for index, status in enumerate(statuses, start=1)
    )
    edge_type = EdgeType.AND_CHILD if operator == "AND" else EdgeType.OR_CHILD
    edges = tuple(
        GraphEdge(
            source_node_code=child.node_code,
            target_node_code=goal.node_code,
            edge_type=edge_type,
            rule_ref=RULE_REF,
        )
        for child in children
    )
    gate = GateDefinition(
        gate_code=f"RULE:SYNTHETIC_{operator}_GATE",
        parent_node_code=str(goal.node_code),
        operator=operator,
        child_node_codes=tuple(str(child.node_code) for child in children),
        rule_ref=RULE_REF,
    )
    return _graph((goal, *children), edges), gate


def _package(graph: GECEGInstance, gates: tuple[GateDefinition, ...] = ()) -> RulePackage:
    global _LAST_PACKAGE_GRAPH
    _LAST_PACKAGE_GRAPH = graph
    provenance = graph.provenance.model_copy(
        update={
            "actor": ActorType.RULE_ENGINE,
            "generator_version": "03C.synthetic.1",
            "rule_versions": (RULE_REF,),
        }
    )
    package = RulePackage(
        payload_version=RULE_PACKAGE_PAYLOAD_VERSION,
        package_code="RULE:SYNTHETIC_PACKAGE",
        package_version="1.0.0",
        package_checksum="sha256:" + "0" * 64,
        release_status="PUBLISHED",
        graph_code=str(graph.graph_code),
        template_ref=graph.template_ref,
        rule_refs=(RULE_REF,),
        gates=gates,
        provenance=provenance,
    )
    return replace(package, package_checksum=compute_rule_package_checksum(package))


def _context(package: RulePackage, **updates: object) -> PropagationContext:
    graph = updates.pop("snapshot_graph", None) or _LAST_PACKAGE_GRAPH
    assert isinstance(graph, GECEGInstance)
    validations = updates.pop("validations", ())
    capability_gaps = updates.pop("capability_gaps", ())
    assert isinstance(validations, tuple)
    assert isinstance(capability_gaps, tuple)
    values = {
        "expected_package_version": package.package_version,
        "expected_package_checksum": package.package_checksum,
        "snapshot_binding": SnapshotBinding(
            str(graph.graph_code), graph.graph_version, canonical_checksum(graph)
        ),
        "budget": PropagationBudget(4, 0, 4, True),
        "validations": tuple(_bind_validation_for_test(graph, item) for item in validations),
        "capability_gaps": tuple(_bind_gap_for_test(graph, item) for item in capability_gaps),
    }
    values.update(updates)
    return PropagationContext(**values)


def _bind_validation_for_test(graph: GECEGInstance, item: object) -> object:
    if isinstance(item, BoundValidation):
        return item
    try:
        return bind_validation(graph, item)  # type: ignore[arg-type]
    except RuleEngineError:
        return item


def _bind_gap_for_test(graph: GECEGInstance, item: object) -> object:
    if isinstance(item, BoundCapabilityGap):
        return item
    try:
        return bind_capability_gap(graph, item)  # type: ignore[arg-type]
    except RuleEngineError:
        return item


def _gap(graph: GECEGInstance, target: str) -> CapabilityGap:
    data = _example(CapabilityGap, "CapabilityGap.valid.json").model_dump(mode="python")
    data["target_node_code"] = target
    data["origin_run_id"] = graph.scope.run_id
    data["scope"]["graph_code"] = graph.graph_code
    return CapabilityGap.model_validate(data)


def _validation(
    graph: GECEGInstance,
    target: str,
    *,
    tool: ToolExecutionStatus | None,
    qc: QCStatus,
    observation: EvidenceObservationStatus,
    validity: EvidenceValidity,
    effect: ClinicalEffect,
    capability: CapabilityAvailability = CapabilityAvailability.AVAILABLE,
) -> ValidationResult:
    data = _example(ValidationResult, "ValidationResult.valid.json").model_dump(mode="python")
    data.update(
        target_node_code=target,
        tool_execution_status=tool,
        qc_status=qc,
        observation=observation,
        validity=validity,
        clinical_effect=effect,
        capability=capability,
        support_score=None,
        refute_score=None,
        graph_delta_candidate=None,
        evidence_bindings=(),
        scope=graph.scope,
    )
    data["provenance"]["rule_versions"] = (RULE_REF,)
    return ValidationResult.model_validate(data)


def _direct_graph(*edge_types: EdgeType) -> GECEGInstance:
    if edge_types == (EdgeType.CONFLICTS_WITH,):
        goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
        target = _node("CLAIM:SYNTHETIC_TARGET", NodeType.CLAIM, GraphNodeStatus.PENDING)
        source = _node("EVIDENCE:SYNTHETIC_REL_0", NodeType.EVIDENCE, GraphNodeStatus.SUPPORTED)
        return _graph(
            (goal, target, source),
            (
                GraphEdge(
                    source_node_code=source.node_code,
                    target_node_code=target.node_code,
                    edge_type=EdgeType.CONFLICTS_WITH,
                    rule_ref=RULE_REF,
                ),
                GraphEdge(
                    source_node_code=target.node_code,
                    target_node_code=goal.node_code,
                    edge_type=EdgeType.SUPPORTS,
                    rule_ref=RULE_REF,
                ),
            ),
        )
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    sources = tuple(
        _node(f"EVIDENCE:SYNTHETIC_REL_{index}", NodeType.EVIDENCE, GraphNodeStatus.SUPPORTED)
        for index in range(len(edge_types))
    )
    edges = tuple(
        GraphEdge(
            source_node_code=source.node_code,
            target_node_code=goal.node_code,
            edge_type=edge_type,
            rule_ref=RULE_REF,
        )
        for source, edge_type in zip(sources, edge_types, strict=True)
    )
    return _graph((goal, *sources), edges)


def test_and_all_satisfied_proposes_one_rule_engine_delta_without_mutating_base() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.SUPPORTED))
    package = _package(graph, (gate,))
    before = graph.model_dump(mode="json")

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is not None
    assert result.candidate_delta.proposed_by is ActorType.RULE_ENGINE
    assert len(result.candidate_delta.operations) == 1
    assert result.candidate_delta.operations[0].to_status is GraphNodeStatus.SUPPORTED
    assert result.evaluation.support.contributing_nodes == (
        "EVIDENCE:SYNTHETIC_1",
        "EVIDENCE:SYNTHETIC_2",
    )
    assert result.evaluation.completeness.required_count == 2
    assert result.evaluation.completeness.resolved_count == 2
    assert graph.model_dump(mode="json") == before


def test_and_missing_child_does_not_support_or_refute() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.PENDING))
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is None
    assert not result.evaluation.support.contributing_nodes
    assert not result.evaluation.refutation.contributing_nodes
    assert result.legal_actions[0].action is PlanAction.SELECT_EVIDENCE_TASK
    assert result.stop_suggestion is None


def test_and_required_capability_gap_safely_stops_without_medical_refutation() -> None:
    graph, gate = _gate_graph(
        "AND",
        (GraphNodeStatus.SUPPORTED, GraphNodeStatus.PENDING),
        availabilities=(DataAvailability.AVAILABLE, DataAvailability.UNAVAILABLE),
    )
    package = _package(graph, (gate,))
    gap = _gap(graph, "EVIDENCE:SYNTHETIC_2")

    result = propagate(graph, package, _context(package, capability_gaps=(gap,)))

    assert result.candidate_delta is None
    assert result.stop_suggestion is not None
    assert result.stop_suggestion.reason.value == "CAPABILITY_UNAVAILABLE"
    assert not result.evaluation.refutation.contributing_nodes
    assert {item.channel for item in result.evaluation.blocked_evidence} >= {"CAPABILITY_GAP"}


def test_or_success_keeps_unfinished_branch_and_does_not_stop() -> None:
    graph, gate = _gate_graph("OR", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.PENDING))
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    node_result = result.evaluation.node_results[0]
    assert result.candidate_delta is not None
    assert node_result.derived_status is GraphNodeStatus.SUPPORTED
    assert node_result.unresolved_nodes == ("EVIDENCE:SYNTHETIC_2",)
    assert result.stop_suggestion is None


def test_or_support_and_refuted_branch_still_supports_and_traces_failed_branch() -> None:
    graph, gate = _gate_graph("OR", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED))
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is not None
    assert result.candidate_delta.operations[0].to_status is GraphNodeStatus.SUPPORTED
    assert result.evaluation.support.contributing_nodes == ("EVIDENCE:SYNTHETIC_1",)
    assert result.evaluation.refutation.contributing_nodes == ("EVIDENCE:SYNTHETIC_2",)


def test_missing_data_cannot_propagate_even_if_node_status_claims_supported() -> None:
    graph, gate = _gate_graph(
        "AND",
        (GraphNodeStatus.SUPPORTED,),
        availabilities=(DataAvailability.UNAVAILABLE,),
    )
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is None
    assert not result.evaluation.support.contributing_nodes
    assert {item.channel for item in result.evaluation.blocked_evidence} >= {"DATA_MISSING"}


def test_or_blocked_branch_returns_switch_to_executable_alternative() -> None:
    graph, gate = _gate_graph(
        "OR",
        (GraphNodeStatus.PENDING, GraphNodeStatus.PENDING),
        availabilities=(DataAvailability.UNAVAILABLE, DataAvailability.ACQUIRABLE),
    )
    package = _package(graph, (gate,))
    gap = _gap(graph, "EVIDENCE:SYNTHETIC_1")

    result = propagate(graph, package, _context(package, capability_gaps=(gap,)))

    assert result.stop_suggestion is None
    assert any(
        action.action is PlanAction.SWITCH_PATH
        and action.target_node_code == "EVIDENCE:SYNTHETIC_2"
        for action in result.legal_actions
    )


def test_or_all_blocked_safely_stops() -> None:
    graph, gate = _gate_graph(
        "OR",
        (GraphNodeStatus.PENDING, GraphNodeStatus.PENDING),
        availabilities=(DataAvailability.UNAVAILABLE, DataAvailability.UNAVAILABLE),
    )
    package = _package(graph, (gate,))
    gaps = (_gap(graph, "EVIDENCE:SYNTHETIC_1"), _gap(graph, "EVIDENCE:SYNTHETIC_2"))
    result = propagate(graph, package, _context(package, capability_gaps=gaps))
    assert result.stop_suggestion is not None
    assert result.stop_suggestion.reason.value == "CAPABILITY_UNAVAILABLE"


@pytest.mark.parametrize(
    ("edge_type", "expected"),
    (
        (EdgeType.SUPPORTS, GraphNodeStatus.SUPPORTED),
        (EdgeType.REFUTES, GraphNodeStatus.REFUTED),
        (EdgeType.EXCLUDES, GraphNodeStatus.EXCLUDED),
        (EdgeType.FALSIFIES, GraphNodeStatus.REFUTED),
        (EdgeType.CONFLICTS_WITH, GraphNodeStatus.CONFLICTED),
    ),
)
def test_each_clinical_relation_has_its_own_handler(
    edge_type: EdgeType, expected: GraphNodeStatus
) -> None:
    graph = _direct_graph(edge_type)
    package = _package(graph)
    result = propagate(graph, package, _context(package))
    assert result.candidate_delta is not None
    assert result.candidate_delta.operations[0].to_status is expected


def test_support_and_refutation_are_conflicted_not_or_overwritten() -> None:
    graph = _direct_graph(EdgeType.SUPPORTS, EdgeType.REFUTES)
    package = _package(graph)
    result = propagate(graph, package, _context(package))
    assert result.candidate_delta is not None
    assert result.candidate_delta.operations[0].to_status is GraphNodeStatus.CONFLICTED
    assert len(result.evaluation.support.contributing_nodes) == 1
    assert len(result.evaluation.refutation.contributing_nodes) == 1
    assert len(result.evaluation.conflict.contributing_nodes) == 2


def test_data_flow_edges_are_traced_but_never_propagate_clinical_support() -> None:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    evidence = _node("EVIDENCE:SYNTHETIC_SIGNAL", NodeType.EVIDENCE, GraphNodeStatus.PENDING)
    data = _node("ARTIFACT:SYNTHETIC_INPUT", NodeType.DATA, GraphNodeStatus.PENDING)
    tool = _node("TOOL:SYNTHETIC_EXECUTOR", NodeType.TOOL, GraphNodeStatus.SUPPORTED)
    output = _node("ARTIFACT:SYNTHETIC_OUTPUT", NodeType.DATA, GraphNodeStatus.PENDING)
    qc = _node("CLAIM:SYNTHETIC_QC", NodeType.QC, GraphNodeStatus.PENDING)
    graph = _graph(
        (goal, evidence, data, tool, output, qc),
        (
            GraphEdge(
                source_node_code=evidence.node_code,
                target_node_code=goal.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            ),
            GraphEdge(
                source_node_code=data.node_code,
                target_node_code=tool.node_code,
                edge_type=EdgeType.REQUIRES_INPUT,
            ),
            GraphEdge(
                source_node_code=tool.node_code,
                target_node_code=output.node_code,
                edge_type=EdgeType.PRODUCES,
            ),
            GraphEdge(
                source_node_code=output.node_code,
                target_node_code=qc.node_code,
                edge_type=EdgeType.VALIDATED_BY,
            ),
            GraphEdge(
                source_node_code=evidence.node_code,
                target_node_code=qc.node_code,
                edge_type=EdgeType.VALIDATED_BY,
            ),
        ),
    )
    package = _package(graph)
    result = propagate(graph, package, _context(package))
    assert result.candidate_delta is None
    assert len(result.evaluation.data_flow_edges) == 4
    assert not result.evaluation.support.contributing_nodes
    assert result.evaluation.completeness.required_count == 1


def test_failure_and_evidence_channels_remain_distinct() -> None:
    graph, gate = _gate_graph(
        "AND",
        tuple(GraphNodeStatus.PENDING for _ in range(6)),
        availabilities=(
            DataAvailability.AVAILABLE,
            DataAvailability.AVAILABLE,
            DataAvailability.AVAILABLE,
            DataAvailability.UNAVAILABLE,
            DataAvailability.AVAILABLE,
            DataAvailability.AVAILABLE,
        ),
    )
    package = _package(graph, (gate,))
    nodes = tuple(f"EVIDENCE:SYNTHETIC_{index}" for index in range(1, 7))
    validations = (
        _validation(
            graph,
            nodes[0],
            tool=ToolExecutionStatus.FAILED,
            qc=QCStatus.UNASSESSED,
            observation=EvidenceObservationStatus.UNOBSERVED,
            validity=EvidenceValidity.INCONCLUSIVE,
            effect=ClinicalEffect.NONE,
        ),
        _validation(
            graph,
            nodes[1],
            tool=ToolExecutionStatus.TIMED_OUT,
            qc=QCStatus.UNASSESSED,
            observation=EvidenceObservationStatus.UNOBSERVED,
            validity=EvidenceValidity.INCONCLUSIVE,
            effect=ClinicalEffect.NONE,
        ),
        _validation(
            graph,
            nodes[2],
            tool=ToolExecutionStatus.SUCCEEDED,
            qc=QCStatus.FAIL,
            observation=EvidenceObservationStatus.INCONCLUSIVE,
            validity=EvidenceValidity.INVALID,
            effect=ClinicalEffect.NONE,
        ),
        _validation(
            graph,
            nodes[4],
            tool=ToolExecutionStatus.SUCCEEDED,
            qc=QCStatus.PASS,
            observation=EvidenceObservationStatus.NEGATIVE,
            validity=EvidenceValidity.VALID,
            effect=ClinicalEffect.NONE,
        ),
        _validation(
            graph,
            nodes[5],
            tool=ToolExecutionStatus.SUCCEEDED,
            qc=QCStatus.PASS,
            observation=EvidenceObservationStatus.NEGATIVE,
            validity=EvidenceValidity.VALID,
            effect=ClinicalEffect.REFUTES,
        ),
    )
    result = propagate(graph, package, _context(package, validations=validations))
    assert {item.channel for item in result.evaluation.blocked_evidence} >= {
        "TOOL_EXECUTION_FAILED",
        "TOOL_EXECUTION_TIMEOUT",
        "QC_FAILED",
        "DATA_MISSING",
        "MEDICAL_NEGATIVE",
        "EVIDENCE_REFUTATION",
    }
    assert result.candidate_delta is None


def test_unobserved_tool_gap_has_its_own_non_medical_channel() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.BLOCKED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED_TOOL_GAP,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
        capability=CapabilityAvailability.UNAVAILABLE_CAPABILITY,
    )

    result = propagate(graph, package, _context(package, validations=(validation,)))

    assert {item.channel for item in result.evaluation.blocked_evidence} >= {"UNOBSERVED_TOOL_GAP"}
    assert result.stop_suggestion is not None
    assert result.stop_suggestion.reason.value == "CAPABILITY_UNAVAILABLE"
    assert not result.evaluation.refutation.contributing_nodes


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ("draft", RuleEngineErrorCode.PACKAGE_NOT_PUBLISHED),
        ("version", RuleEngineErrorCode.PACKAGE_VERSION_MISMATCH),
        ("checksum", RuleEngineErrorCode.PACKAGE_CHECKSUM_MISMATCH),
        ("scope", RuleEngineErrorCode.PACKAGE_SCOPE_MISMATCH),
    ),
)
def test_unusable_rule_packages_become_knowledge_gap_safe_stops(
    mutation: str, code: RuleEngineErrorCode
) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.SUPPORTED))
    package = _package(graph, (gate,))
    context = _context(package)
    if mutation == "draft":
        package = replace(package, release_status="DRAFT")
    elif mutation == "version":
        context = replace(context, expected_package_version="2.0.0")
    elif mutation == "checksum":
        context = replace(context, expected_package_checksum="sha256:" + "b" * 64)
    else:
        package = replace(package, graph_code="GRAPH:OID_OTHER001")
        package = replace(package, package_checksum=compute_rule_package_checksum(package))
        context = replace(context, expected_package_checksum=package.package_checksum)
    result = propagate(graph, package, context)
    assert result.candidate_delta is None
    assert result.knowledge_gaps[0].code == code.value
    assert result.stop_suggestion is not None
    assert result.stop_suggestion.reason.value == "UNSAFE_TO_CONTINUE"


def test_missing_rule_package_is_structured_safe_stop() -> None:
    graph, _ = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    result = propagate(
        graph,
        None,
        PropagationContext(
            "1.0.0",
            SHA,
            snapshot_binding_for(graph),
            budget=PropagationBudget(1, 0, 1, True),
        ),
    )
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.PACKAGE_MISSING.value
    assert result.stop_suggestion is not None


def test_cycle_duplicate_gate_and_and_or_mix_are_stably_rejected() -> None:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    first = _node("CLAIM:SYNTHETIC_A", NodeType.CLAIM, GraphNodeStatus.PENDING)
    second = _node("CLAIM:SYNTHETIC_B", NodeType.CLAIM, GraphNodeStatus.PENDING)
    cycle_graph = _graph(
        (goal, first, second),
        (
            GraphEdge(
                source_node_code=first.node_code,
                target_node_code=second.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            ),
            GraphEdge(
                source_node_code=second.node_code,
                target_node_code=first.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            ),
            GraphEdge(
                source_node_code=second.node_code,
                target_node_code=goal.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            ),
        ),
    )
    cycle_package = _package(cycle_graph)
    with pytest.raises(RuleEngineError) as cycle:
        validate_rule_package(
            cycle_graph,
            cycle_package,
            expected_version=cycle_package.package_version,
            expected_checksum=cycle_package.package_checksum,
        )
    assert cycle.value.code is RuleEngineErrorCode.DEPENDENCY_CYCLE

    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING, GraphNodeStatus.PENDING))
    duplicate_package = _package(graph, (gate, replace(gate, gate_code="RULE:SYNTHETIC_DUP")))
    result = propagate(graph, duplicate_package, _context(duplicate_package))
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.DUPLICATE_GATE.value

    mixed_edges = tuple(graph.edges) + tuple(
        edge.model_copy(update={"edge_type": EdgeType.OR_CHILD}) for edge in graph.edges
    )
    mixed_graph = _graph(graph.nodes, mixed_edges)
    or_gate = replace(gate, gate_code="RULE:SYNTHETIC_OR", operator="OR")
    mixed_package = _package(mixed_graph, (gate, or_gate))
    result = propagate(mixed_graph, mixed_package, _context(mixed_package))
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.GATE_MIXED_OPERATOR.value


def test_unknown_gate_node_and_unbound_edge_rule_are_stably_rejected() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (replace(gate, child_node_codes=("EVIDENCE:UNKNOWN",)),))
    result = propagate(graph, package, _context(package))
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.UNKNOWN_NODE.value

    wrong_ref = RULE_REF.model_copy(update={"version": "9.9.9"})
    bad_graph = graph.model_copy(
        update={"edges": (graph.edges[0].model_copy(update={"rule_ref": wrong_ref}),)}
    )
    package = _package(bad_graph, (gate,))
    result = propagate(bad_graph, package, _context(package))
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.RULE_REFERENCE_INVALID.value


def test_set_order_permutations_are_deterministic_and_array_semantics_are_preserved() -> None:
    graph, gate = _gate_graph("OR", (GraphNodeStatus.PENDING, GraphNodeStatus.SUPPORTED))
    second_ref = VersionedReference(code="RULE:SYNTHETIC_SECOND", version="1.0.0", checksum=SHA)
    package = _package(graph, (gate,))
    package = replace(
        package,
        rule_refs=(second_ref, RULE_REF),
        provenance=package.provenance.model_copy(update={"rule_versions": (RULE_REF, second_ref)}),
    )
    package = replace(package, package_checksum=compute_rule_package_checksum(package))
    reordered = replace(
        package,
        rule_refs=tuple(reversed(package.rule_refs)),
        gates=(replace(gate, child_node_codes=tuple(reversed(gate.child_node_codes))),),
    )
    assert compute_rule_package_checksum(reordered) == package.package_checksum
    first = propagate(graph, package, _context(package))
    second = propagate(graph, reordered, _context(reordered))
    assert first == second
    assert first.evaluation.node_results[0].unresolved_nodes == ("EVIDENCE:SYNTHETIC_1",)


def test_validation_and_capability_gap_set_order_is_deterministic() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING, GraphNodeStatus.PENDING))
    package = _package(graph, (gate,))
    validations = tuple(
        _validation(
            graph,
            f"EVIDENCE:SYNTHETIC_{index}",
            tool=ToolExecutionStatus.FAILED,
            qc=QCStatus.UNASSESSED,
            observation=EvidenceObservationStatus.UNOBSERVED,
            validity=EvidenceValidity.INCONCLUSIVE,
            effect=ClinicalEffect.NONE,
        )
        for index in (1, 2)
    )
    gaps = tuple(_gap(graph, f"EVIDENCE:SYNTHETIC_{index}") for index in (1, 2))

    first = propagate(
        graph,
        package,
        _context(package, validations=validations, capability_gaps=gaps),
    )
    second = propagate(
        graph,
        package,
        _context(
            package,
            validations=tuple(reversed(validations)),
            capability_gaps=tuple(reversed(gaps)),
        ),
    )

    assert first == second


def test_fixed_point_and_budget_exhaustion_have_distinct_safe_stops() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0],
                graph.nodes[1].model_copy(update={"availability": DataAvailability.UNAVAILABLE}),
            )
        }
    )
    package = _package(graph, (gate,))
    fixed = propagate(
        graph,
        package,
        _context(
            package,
            history=PropagationHistory((canonical_checksum(graph),)),
        ),
    )
    assert fixed.stop_suggestion is not None
    assert fixed.stop_suggestion.reason_codes == ("RULE:NO_PROGRESS",)
    budget = propagate(
        graph,
        package,
        _context(package, budget=PropagationBudget(1, 1, 0, True)),
    )
    assert budget.stop_suggestion is not None
    assert budget.stop_suggestion.reason.value == "BUDGET_EXHAUSTED"


def test_budget_and_fixed_point_suppress_otherwise_valid_candidate_delta() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    package = _package(graph, (gate,))

    budget = propagate(
        graph,
        package,
        _context(package, budget=PropagationBudget(1, 1, 0, True)),
    )
    fixed = propagate(
        graph,
        package,
        _context(package, history=PropagationHistory((canonical_checksum(graph),))),
    )

    assert budget.candidate_delta is None
    assert budget.stop_suggestion is not None
    assert budget.stop_suggestion.reason.value == "BUDGET_EXHAUSTED"
    assert fixed.candidate_delta is None
    assert fixed.stop_suggestion is not None
    assert fixed.stop_suggestion.reason_codes == ("RULE:NO_PROGRESS",)


def test_each_graph_version_gets_one_sorted_batch_before_next_propagation_round() -> None:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    claims = tuple(
        _node(f"CLAIM:SYNTHETIC_{index}", NodeType.CLAIM, GraphNodeStatus.PENDING)
        for index in (1, 2)
    )
    evidence = tuple(
        _node(f"EVIDENCE:SYNTHETIC_{index}", NodeType.EVIDENCE, GraphNodeStatus.SUPPORTED)
        for index in (1, 2)
    )
    edges = tuple(
        GraphEdge(
            source_node_code=evidence[index].node_code,
            target_node_code=claims[index].node_code,
            edge_type=EdgeType.SUPPORTS,
            rule_ref=RULE_REF,
        )
        for index in (0, 1)
    ) + tuple(
        GraphEdge(
            source_node_code=claim.node_code,
            target_node_code=goal.node_code,
            edge_type=EdgeType.AND_CHILD,
            rule_ref=RULE_REF,
        )
        for claim in claims
    )
    graph = _graph((goal, *claims, *evidence), edges)
    gate = GateDefinition(
        gate_code="RULE:SYNTHETIC_AND_GOAL",
        parent_node_code=str(goal.node_code),
        operator="AND",
        child_node_codes=tuple(str(claim.node_code) for claim in claims),
        rule_ref=RULE_REF,
    )
    package = _package(graph, (gate,))

    first = propagate(graph, package, _context(package))
    assert first.candidate_delta is not None
    assert tuple(operation.node_code for operation in first.candidate_delta.operations) == tuple(
        str(claim.node_code) for claim in claims
    )
    assert all(
        operation.to_status is GraphNodeStatus.SUPPORTED
        for operation in first.candidate_delta.operations
    )
    version_two = apply_graph_delta(graph, plan_graph_delta(graph, first.candidate_delta)).instance
    second = propagate(
        version_two,
        package,
        _context(
            package,
            snapshot_graph=version_two,
            budget=PropagationBudget(4, 1, 3, True),
        ),
    )
    assert second.candidate_delta is not None
    assert tuple(operation.node_code for operation in second.candidate_delta.operations) == (
        str(goal.node_code),
    )
    version_three = apply_graph_delta(
        version_two, plan_graph_delta(version_two, second.candidate_delta)
    ).instance
    final = propagate(
        version_three,
        package,
        _context(
            package,
            snapshot_graph=version_three,
            budget=PropagationBudget(4, 2, 2, True),
        ),
    )
    assert final.candidate_delta is None
    assert final.stop_suggestion is not None
    assert final.stop_suggestion.reason.value == "GOAL_SUPPORTED"
    assert graph.graph_version + 2 == version_three.graph_version


def test_applied_conflict_becomes_safe_conflicting_evidence_stop_next_round() -> None:
    graph = _direct_graph(EdgeType.SUPPORTS, EdgeType.REFUTES)
    package = _package(graph)
    first = propagate(graph, package, _context(package))
    assert first.candidate_delta is not None
    conflicted = apply_graph_delta(graph, plan_graph_delta(graph, first.candidate_delta)).instance

    second = propagate(
        conflicted,
        package,
        _context(
            package,
            snapshot_graph=conflicted,
            budget=PropagationBudget(4, 1, 3, True),
        ),
    )

    assert second.candidate_delta is None
    assert second.stop_suggestion is not None
    assert second.stop_suggestion.reason.value == "CONFLICTING_EVIDENCE"


def test_illegal_expert_suggestion_is_rejected_against_exact_legal_set() -> None:
    assessment = _example(ExpertAssessment, "ExpertAssessment.valid.json")
    legal = (LegalAction(PlanAction.STOP, None, 1, SHA, "RULE:SYNTHETIC_RULE"),)
    with pytest.raises(RuleEngineError) as error:
        validate_expert_action(assessment, legal)
    assert error.value.code is RuleEngineErrorCode.EXPERT_ACTION_INVALID

    data = assessment.model_dump(mode="python")
    data.update(
        decision=ExpertAssessmentDecision.SELECT_ALLOWED_ALTERNATIVE,
        selected_action_code=PlanAction.STOP,
        allowed_action_codes=(PlanAction.STOP,),
    )
    selected = validate_expert_action(ExpertAssessment.model_validate(data), legal)
    assert selected == legal[0]


def test_candidate_delta_must_pass_03b_apply_strict_event_commit_and_replay() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.SUPPORTED))
    package = _package(graph, (gate,))
    before = graph.model_dump(mode="json")
    propagation = propagate(graph, package, _context(package))
    assert propagation.candidate_delta is not None

    plan = plan_graph_delta(graph, propagation.candidate_delta)
    application = apply_graph_delta(graph, plan)
    payload = accepted_delta_event_to_payload(application.accepted_event)
    restored_event = accepted_delta_event_from_payload(payload)
    replayed = replay_accepted_deltas(graph, (restored_event,))
    commit = build_graph_version_commit(
        graph,
        plan,
        GraphVersionAuditContext(
            event_id="MESSAGE:OID_EVT03C001",
            correlation_code="MESSAGE:OID_CORR03C01",
            occurred_at=package.provenance.created_at,
        ),
    )

    assert graph.model_dump(mode="json") == before
    assert replayed == application.instance == commit.command.graph
    assert replayed.graph_version == graph.graph_version + 1
    assert replayed.nodes[0].status is GraphNodeStatus.SUPPORTED


def test_public_rule_engine_import_has_no_db_agent_llm_or_tool_runtime_dependency() -> None:
    import subprocess
    import sys

    script = """
import sys
import packages.rule_engine
for forbidden in ('db', 'agents', 'openai', 'Algorithm_Tool', 'Seg_Model'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_old_regression_goal_stop_must_not_contradict_current_refuted_derivation() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.REFUTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": GraphNodeStatus.SUPPORTED}),
                graph.nodes[1],
            )
        }
    )
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is not None
    assert result.candidate_delta.operations[0].to_status is GraphNodeStatus.REFUTED
    assert result.stop_suggestion is None


def test_old_regression_qc_failed_refuted_child_cannot_refute_and_parent() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.REFUTED,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.SUCCEEDED,
        qc=QCStatus.FAIL,
        observation=EvidenceObservationStatus.INCONCLUSIVE,
        validity=EvidenceValidity.INVALID,
        effect=ClinicalEffect.NONE,
    )

    result = propagate(graph, package, _context(package, validations=(validation,)))

    assert {item.channel for item in result.evaluation.blocked_evidence} >= {"QC_FAILED"}
    assert result.candidate_delta is None
    assert not result.evaluation.refutation.contributing_nodes


def test_old_regression_illegal_tool_supports_goal_is_structurally_rejected() -> None:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    tool = _node("TOOL:SYNTHETIC_EXECUTOR", NodeType.TOOL, GraphNodeStatus.SUPPORTED)
    graph = _graph(
        (goal, tool),
        (
            GraphEdge(
                source_node_code=tool.node_code,
                target_node_code=goal.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            ),
        ),
    )
    package = _package(graph)

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is None
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.GRAPH_CONTRACT_INVALID.value


def test_old_regression_cross_run_validation_is_rejected_before_blocking_current_graph() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    ).model_copy(update={"scope": graph.scope.model_copy(update={"run_id": "RUN:OID_RUN99999"})})

    result = propagate(graph, package, _context(package, validations=(validation,)))

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value
    assert not result.evaluation.blocked_evidence


def test_old_regression_provenance_source_set_order_does_not_change_package_identity() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    first = package.provenance.model_copy(
        update={"source_codes": ("RULE:SYNTHETIC_SOURCE_A", "RULE:SYNTHETIC_SOURCE_B")}
    )
    second = first.model_copy(update={"source_codes": tuple(reversed(first.source_codes))})

    first_package = replace(package, provenance=first)
    second_package = replace(package, provenance=second)

    assert compute_rule_package_checksum(first_package) == compute_rule_package_checksum(
        second_package
    )


def test_old_regression_wrong_package_type_returns_structured_safe_result() -> None:
    graph, _ = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    context = PropagationContext(
        "1.0.0",
        SHA,
        snapshot_binding_for(graph),
        budget=PropagationBudget(1, 0, 1, True),
    )

    result = propagate(graph, object(), context)  # type: ignore[arg-type]

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.PACKAGE_MISSING.value
    assert result.stop_suggestion is not None


@pytest.mark.parametrize(
    ("edge_type", "expected_status"),
    (
        (EdgeType.SUPPORTS, GraphNodeStatus.SUPPORTED),
        (EdgeType.REFUTES, GraphNodeStatus.REFUTED),
        (EdgeType.EXCLUDES, GraphNodeStatus.EXCLUDED),
        (EdgeType.FALSIFIES, GraphNodeStatus.REFUTED),
        (EdgeType.CONFLICTS_WITH, GraphNodeStatus.CONFLICTED),
    ),
)
@pytest.mark.parametrize(
    "condition", ("AVAILABLE", "QC_FAIL", "TOOL_FAIL", "CAPABILITY", "MISSING")
)
def test_all_clinical_relations_use_one_trusted_contribution_boundary(
    edge_type: EdgeType,
    expected_status: GraphNodeStatus,
    condition: str,
) -> None:
    graph = _direct_graph(edge_type)
    relation = next(edge for edge in graph.edges if edge.edge_type is edge_type)
    source_code = str(relation.source_node_code)
    if condition == "MISSING":
        graph = graph.model_copy(
            update={
                "nodes": tuple(
                    node.model_copy(update={"availability": DataAvailability.UNAVAILABLE})
                    if str(node.node_code) == source_code
                    else node
                    for node in graph.nodes
                )
            }
        )
    package = _package(graph)
    context_updates: dict[str, object] = {}
    if condition == "QC_FAIL":
        context_updates["validations"] = (
            _validation(
                graph,
                source_code,
                tool=ToolExecutionStatus.SUCCEEDED,
                qc=QCStatus.FAIL,
                observation=EvidenceObservationStatus.INCONCLUSIVE,
                validity=EvidenceValidity.INVALID,
                effect=ClinicalEffect.NONE,
            ),
        )
    elif condition == "TOOL_FAIL":
        context_updates["validations"] = (
            _validation(
                graph,
                source_code,
                tool=ToolExecutionStatus.FAILED,
                qc=QCStatus.UNASSESSED,
                observation=EvidenceObservationStatus.UNOBSERVED,
                validity=EvidenceValidity.INCONCLUSIVE,
                effect=ClinicalEffect.NONE,
            ),
        )
    elif condition == "CAPABILITY":
        context_updates["capability_gaps"] = (_gap(graph, source_code),)

    result = propagate(graph, package, _context(package, **context_updates))

    target_code = str(relation.target_node_code)
    target_operations = (
        tuple(
            operation
            for operation in result.candidate_delta.operations
            if str(operation.node_code) == target_code
        )
        if result.candidate_delta is not None
        else ()
    )
    if condition == "AVAILABLE":
        assert target_operations
        assert target_operations[0].to_status is expected_status
    else:
        assert not target_operations
        assert source_code not in result.evaluation.support.contributing_nodes
        assert source_code not in result.evaluation.refutation.contributing_nodes
        assert source_code not in result.evaluation.conflict.contributing_nodes


@pytest.mark.parametrize(
    ("operator", "statuses", "expected"),
    (
        ("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.SUPPORTED), GraphNodeStatus.SUPPORTED),
        ("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED), GraphNodeStatus.REFUTED),
        ("AND", (GraphNodeStatus.REFUTED, GraphNodeStatus.PENDING), GraphNodeStatus.REFUTED),
        ("AND", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.PENDING), None),
        ("OR", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED), GraphNodeStatus.SUPPORTED),
        ("OR", (GraphNodeStatus.SUPPORTED, GraphNodeStatus.PENDING), GraphNodeStatus.SUPPORTED),
        ("OR", (GraphNodeStatus.REFUTED, GraphNodeStatus.REFUTED), GraphNodeStatus.REFUTED),
        ("OR", (GraphNodeStatus.REFUTED, GraphNodeStatus.PENDING), None),
    ),
)
def test_complete_generic_gate_truth_table(
    operator: str,
    statuses: tuple[GraphNodeStatus, ...],
    expected: GraphNodeStatus | None,
) -> None:
    graph, gate = _gate_graph(operator, statuses)
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    operation = result.candidate_delta.operations[0] if result.candidate_delta else None
    assert (operation.to_status if operation else None) is expected


@pytest.mark.parametrize("old_status", (GraphNodeStatus.REFUTED, GraphNodeStatus.CONFLICTED))
def test_current_supported_derivation_corrects_old_non_supported_goal_without_old_stop(
    old_status: GraphNodeStatus,
) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": old_status}),
                graph.nodes[1],
            )
        }
    )
    package = _package(graph, (gate,))

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is not None
    assert result.candidate_delta.operations[0].to_status is GraphNodeStatus.SUPPORTED
    assert result.stop_suggestion is None


@pytest.mark.parametrize(
    ("source_type", "edge_type"),
    (
        (NodeType.TOOL, EdgeType.SUPPORTS),
        (NodeType.DATA, EdgeType.REFUTES),
        (NodeType.EVIDENCE, EdgeType.PRODUCES),
    ),
)
def test_public_graph_structure_boundary_rejects_illegal_endpoint_combinations(
    source_type: NodeType, edge_type: EdgeType
) -> None:
    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    source_code = {
        NodeType.TOOL: "TOOL:SYNTHETIC_SOURCE",
        NodeType.DATA: "ARTIFACT:SYNTHETIC_SOURCE",
        NodeType.EVIDENCE: "EVIDENCE:SYNTHETIC_SOURCE",
    }[source_type]
    source = _node(source_code, source_type, GraphNodeStatus.SUPPORTED)
    target = (
        goal
        if edge_type in {EdgeType.SUPPORTS, EdgeType.REFUTES}
        else _node("ARTIFACT:SYNTHETIC_TARGET", NodeType.DATA, GraphNodeStatus.PENDING)
    )
    edges = [
        GraphEdge(
            source_node_code=source.node_code,
            target_node_code=target.node_code,
            edge_type=edge_type,
            rule_ref=RULE_REF if edge_type in CLINICAL_EDGE_TYPES else None,
        )
    ]
    nodes = (goal, source) if target is goal else (goal, source, target)
    if target is not goal:
        edges.append(
            GraphEdge(
                source_node_code=source.node_code,
                target_node_code=goal.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=RULE_REF,
            )
        )
    graph = _graph(nodes, tuple(edges))
    package = _package(graph)

    result = propagate(graph, package, _context(package))

    assert result.candidate_delta is None
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.GRAPH_CONTRACT_INVALID.value


@pytest.mark.parametrize("scope_field", ("case_code", "run_id", "graph_code"))
def test_validation_scope_must_exactly_match_current_graph(scope_field: str) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    )
    replacement = {
        "case_code": "CASE:ANON_CASE0002",
        "run_id": "RUN:OID_RUN99999",
        "graph_code": "GRAPH:OID_GRAPH999",
    }[scope_field]
    validation = validation.model_copy(
        update={"scope": validation.scope.model_copy(update={scope_field: replacement})}
    )

    result = propagate(graph, package, _context(package, validations=(validation,)))

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value
    assert not result.evaluation.blocked_evidence


@pytest.mark.parametrize("scope_field", ("case_code", "run_id", "graph_code", "origin_run_id"))
def test_capability_gap_scope_must_match_case_run_and_graph(scope_field: str) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    gap = _gap(graph, "EVIDENCE:SYNTHETIC_1")
    replacement = {
        "case_code": "CASE:ANON_CASE0002",
        "run_id": "RUN:OID_RUN99999",
        "graph_code": "GRAPH:OID_GRAPH999",
        "origin_run_id": "RUN:OID_RUN99999",
    }[scope_field]
    gap = (
        gap.model_copy(update={"origin_run_id": replacement})
        if scope_field == "origin_run_id"
        else gap.model_copy(
            update={"scope": gap.scope.model_copy(update={scope_field: replacement})}
        )
    )

    result = propagate(graph, package, _context(package, capability_gaps=(gap,)))

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value


@pytest.mark.parametrize("mutation", ("graph_code", "graph_version", "graph_checksum"))
def test_snapshot_binding_rejects_cross_graph_old_version_and_old_checksum(mutation: str) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    context = _context(package)
    replacement: object = {
        "graph_code": "GRAPH:OID_GRAPH999",
        "graph_version": graph.graph_version + 1,
        "graph_checksum": "sha256:" + "b" * 64,
    }[mutation]
    context = replace(
        context,
        snapshot_binding=replace(context.snapshot_binding, **{mutation: replacement}),
    )

    result = propagate(graph, package, context)

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH.value


def test_all_provenance_set_fields_are_canonical_for_checksum_and_result() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    second_ref = VersionedReference(code="RULE:SYNTHETIC_SECOND", version="2.0.0", checksum=SHA)
    knowledge_ref = VersionedReference(
        code="RULE:SYNTHETIC_KNOWLEDGE", version="1.0.0", checksum=SHA
    )
    tool_ref = VersionedReference(code="TOOL:SYNTHETIC_TOOL", version="1.0.0", checksum=SHA)
    provenance = package.provenance.model_copy(
        update={
            "source_codes": ("RULE:SYNTHETIC_SOURCE_B", "RULE:SYNTHETIC_SOURCE_A"),
            "rule_versions": (second_ref, RULE_REF),
            "knowledge_versions": (knowledge_ref, RULE_REF),
            "tool_versions": (tool_ref, second_ref),
        }
    )
    first = replace(package, provenance=provenance)
    second = replace(
        package,
        provenance=provenance.model_copy(
            update={
                "source_codes": tuple(reversed(provenance.source_codes)),
                "rule_versions": tuple(reversed(provenance.rule_versions)),
                "knowledge_versions": tuple(reversed(provenance.knowledge_versions)),
                "tool_versions": tuple(reversed(provenance.tool_versions)),
            }
        ),
    )
    checksum = compute_rule_package_checksum(first)
    first = replace(first, package_checksum=checksum)
    second = replace(second, package_checksum=checksum)

    assert rule_package_checksum_payload(first) == rule_package_checksum_payload(second)
    assert propagate(graph, first, _context(first)) == propagate(graph, second, _context(second))


@pytest.mark.parametrize("malformation", ("gates", "nan", "context"))
def test_malformed_inputs_return_structured_safe_results_without_raw_exceptions(
    malformation: str,
) -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package: object = _package(graph, (gate,))
    context: object = _context(package)
    if malformation == "gates":
        package = replace(package, gates=object())  # type: ignore[arg-type]
    elif malformation == "nan":
        package = replace(package, package_version=float("nan"))  # type: ignore[arg-type]
    else:
        context = object()

    result = propagate(graph, package, context)  # type: ignore[arg-type]

    assert result.candidate_delta is None
    assert result.knowledge_gaps
    assert result.stop_suggestion is not None
    assert result.stop_suggestion.reason.value == "UNSAFE_TO_CONTINUE"


def test_illegal_direct_transition_uses_versioned_pending_reset_then_03b_derivation() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": GraphNodeStatus.EXCLUDED}),
                graph.nodes[1],
            )
        }
    )
    package = _package(graph, (gate,))

    first = propagate(graph, package, _context(package))
    assert first.evaluation.transition_blocks[0].strategy_version == "03C_TRANSITION_RESET_V1"
    assert first.candidate_delta is not None
    assert first.candidate_delta.operations[0].to_status is GraphNodeStatus.PENDING
    first_plan = plan_graph_delta(graph, first.candidate_delta)
    version_two = apply_graph_delta(graph, first_plan).instance
    first_event = accepted_delta_event_from_payload(
        accepted_delta_event_to_payload(first_plan.accepted_event)
    )
    assert replay_accepted_deltas(graph, (first_event,)) == version_two

    second = propagate(
        version_two,
        package,
        _context(package, snapshot_graph=version_two, budget=PropagationBudget(4, 1, 3, True)),
    )
    assert second.candidate_delta is not None
    assert second.candidate_delta.operations[0].to_status is GraphNodeStatus.SUPPORTED
    second_plan = plan_graph_delta(version_two, second.candidate_delta)
    version_three = apply_graph_delta(version_two, second_plan).instance
    second_event = accepted_delta_event_from_payload(
        accepted_delta_event_to_payload(second_plan.accepted_event)
    )
    assert replay_accepted_deltas(graph, (first_event, second_event)) == version_three
    assert version_three.nodes[0].status is GraphNodeStatus.SUPPORTED


def test_old_second_review_validation_cannot_be_laundered_by_new_context_binding() -> None:
    version_one, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    validation = _validation(
        version_one,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    )
    bound_validation = bind_validation(version_one, validation)
    version_two = version_one.model_copy(update={"graph_version": 2})
    package = _package(version_two, (gate,))

    result = propagate(
        version_two,
        package,
        _context(package, snapshot_graph=version_two, validations=(bound_validation,)),
    )

    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH.value
    assert not result.evaluation.blocked_evidence


def test_old_second_review_validated_package_cannot_cross_snapshot() -> None:
    version_one, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    package = _package(version_one, (gate,))
    validated = validate_rule_package(
        version_one,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    version_two = version_one.model_copy(
        update={
            "graph_version": 2,
            "edges": tuple(
                edge.model_copy(update={"edge_type": EdgeType.SUPPORTS})
                for edge in version_one.edges
            ),
        }
    )

    with pytest.raises(RuleEngineError) as caught:
        evaluate_propagation(
            version_two,
            validated,
            _context(package, snapshot_graph=version_two),
        )

    assert caught.value.code is RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH

    laundered = replace(validated, snapshot_binding=snapshot_binding_for(version_two))
    with pytest.raises(RuleEngineError) as laundered_error:
        evaluate_propagation(
            version_two,
            laundered,
            _context(package, snapshot_graph=version_two),
        )
    assert laundered_error.value.code is RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH


def test_old_second_review_reset_identity_is_persisted_in_accepted_event_delta() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": GraphNodeStatus.EXCLUDED}),
                graph.nodes[1],
            )
        }
    )
    package = _package(graph, (gate,))
    result = propagate(graph, package, _context(package))
    assert result.candidate_delta is not None
    event_payload = accepted_delta_event_to_payload(
        plan_graph_delta(graph, result.candidate_delta).accepted_event
    )

    persisted_reasons = set(event_payload["delta"]["reason_codes"])
    assert "RULE:TRANSITION_RESET_V1" in persisted_reasons
    assert "RULE:TRANSITION_RESET_REQUIRED" in persisted_reasons
    assert "RULE:TRANSITION_RESET_OP_000_DERIVED_SUPPORTED" in persisted_reasons


def test_each_context_item_requires_its_own_current_snapshot_binding() -> None:
    version_one, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    validation = _validation(
        version_one,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    )
    gap = _gap(version_one, "EVIDENCE:SYNTHETIC_1")
    bound_validation = bind_validation(version_one, validation)
    bound_gap = bind_capability_gap(version_one, gap)
    assert bound_validation.source_snapshot.graph_checksum in (
        bound_validation.item.provenance.input_checksums
    )
    assert bound_gap.source_snapshot.graph_checksum in bound_gap.item.provenance.input_checksums

    version_two = version_one.model_copy(update={"graph_version": 2})
    package = _package(version_two, (gate,))
    for field, item in (
        ("validations", (bound_validation,)),
        ("capability_gaps", (bound_gap,)),
    ):
        result = propagate(
            version_two,
            package,
            _context(package, snapshot_graph=version_two, **{field: item}),
        )
        assert result.knowledge_gaps[0].code == (
            RuleEngineErrorCode.SNAPSHOT_BINDING_MISMATCH.value
        )
        assert not result.evaluation.blocked_evidence


def test_bare_context_items_are_rejected_but_current_bound_items_propagate() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    )
    bare = PropagationContext(
        package.package_version,
        package.package_checksum,
        snapshot_binding_for(graph),
        validations=(validation,),  # type: ignore[arg-type]
        budget=PropagationBudget(4, 0, 4, True),
    )
    rejected = propagate(graph, package, bare)
    assert rejected.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value
    bare_gap = replace(
        bare,
        validations=(),
        capability_gaps=(_gap(graph, "EVIDENCE:SYNTHETIC_1"),),  # type: ignore[arg-type]
    )
    rejected_gap = propagate(graph, package, bare_gap)
    assert rejected_gap.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value

    accepted = propagate(
        graph,
        package,
        _context(package, validations=(bind_validation(graph, validation),)),
    )
    assert not accepted.knowledge_gaps
    assert {item.channel for item in accepted.evaluation.blocked_evidence} >= {
        "TOOL_EXECUTION_FAILED"
    }


def test_bound_item_checksum_tampering_is_rejected() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validation = _validation(
        graph,
        "EVIDENCE:SYNTHETIC_1",
        tool=ToolExecutionStatus.FAILED,
        qc=QCStatus.UNASSESSED,
        observation=EvidenceObservationStatus.UNOBSERVED,
        validity=EvidenceValidity.INCONCLUSIVE,
        effect=ClinicalEffect.NONE,
    )
    bound = replace(bind_validation(graph, validation), binding_checksum="sha256:" + "b" * 64)
    result = propagate(graph, package, _context(package, validations=(bound,)))
    assert result.knowledge_gaps[0].code == RuleEngineErrorCode.CONTEXT_INVALID.value


def test_reset_event_payload_alone_recovers_strategy_and_final_target() -> None:
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": GraphNodeStatus.EXCLUDED}),
                graph.nodes[1],
            )
        }
    )
    package = _package(graph, (gate,))
    propagation = propagate(graph, package, _context(package))
    assert propagation.candidate_delta is not None
    plan = plan_graph_delta(graph, propagation.candidate_delta)
    payload = accepted_delta_event_to_payload(plan.accepted_event)

    audit = transition_reset_audit_from_event_payload(payload)
    assert audit.strategy_version == "03C_TRANSITION_RESET_V1"
    assert audit.reason_code == "RULE:TRANSITION_RESET_REQUIRED"
    assert audit.entries[0].current_status is GraphNodeStatus.EXCLUDED
    assert audit.entries[0].reset_status is GraphNodeStatus.PENDING
    assert audit.entries[0].derived_status is GraphNodeStatus.SUPPORTED
    assert audit.delta_checksum == plan.accepted_event.delta_checksum
    assert (
        replay_accepted_deltas(graph, (accepted_delta_event_from_payload(payload),))
        == apply_graph_delta(graph, plan).instance
    )


def test_missing_or_operation_inconsistent_reset_metadata_is_rejected() -> None:
    reset_graph, reset_gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    reset_graph = reset_graph.model_copy(
        update={
            "nodes": (
                reset_graph.nodes[0].model_copy(update={"status": GraphNodeStatus.EXCLUDED}),
                reset_graph.nodes[1],
            )
        }
    )
    reset_package = _package(reset_graph, (reset_gate,))
    reset_result = propagate(reset_graph, reset_package, _context(reset_package))
    assert reset_result.candidate_delta is not None
    reset_delta = reset_result.candidate_delta
    clinical_reasons = tuple(
        code for code in reset_delta.reason_codes if str(code) == "RULE:SYNTHETIC_RULE"
    )
    clinical_refs = tuple(
        ref
        for ref in reset_delta.provenance.rule_versions
        if str(ref.code) == "RULE:SYNTHETIC_RULE"
    )
    missing = reset_delta.model_copy(
        update={
            "reason_codes": clinical_reasons,
            "provenance": reset_delta.provenance.model_copy(
                update={"rule_versions": clinical_refs}
            ),
        }
    )
    with pytest.raises(GraphEngineError) as missing_error:
        plan_graph_delta(reset_graph, missing)
    assert missing_error.value.code is GraphEngineErrorCode.TRANSITION_AUDIT_INVALID

    normal_graph, normal_gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    normal_package = _package(normal_graph, (normal_gate,))
    normal_result = propagate(normal_graph, normal_package, _context(normal_package))
    assert normal_result.candidate_delta is not None
    reset_reasons = tuple(
        code for code in reset_delta.reason_codes if str(code).startswith("RULE:TRANSITION_RESET_")
    )
    reset_refs = tuple(
        ref
        for ref in reset_delta.provenance.rule_versions
        if str(ref.code).startswith("RULE:TRANSITION_RESET_")
    )
    inconsistent = normal_result.candidate_delta.model_copy(
        update={
            "reason_codes": tuple(
                sorted({*normal_result.candidate_delta.reason_codes, *reset_reasons})
            ),
            "provenance": normal_result.candidate_delta.provenance.model_copy(
                update={
                    "rule_versions": tuple(
                        sorted(
                            {*normal_result.candidate_delta.provenance.rule_versions, *reset_refs},
                            key=lambda ref: (str(ref.code), ref.version, ref.checksum or ""),
                        )
                    )
                }
            ),
        }
    )
    with pytest.raises(GraphEngineError) as inconsistent_error:
        plan_graph_delta(normal_graph, inconsistent)
    assert inconsistent_error.value.code is GraphEngineErrorCode.TRANSITION_AUDIT_INVALID

    with pytest.raises(RuleEngineError):
        transition_reset_audit_from_delta(
            reset_delta.model_copy(
                update={
                    "reason_codes": tuple(
                        code
                        for code in reset_delta.reason_codes
                        if str(code) != "RULE:TRANSITION_RESET_REQUIRED"
                    )
                }
            )
        )
