"""Independent synthetic 03D adversarial and metamorphic rule-engine checks."""

import itertools
import random
from dataclasses import replace

import pytest
from test_rule_engine_03c import _context, _gate_graph, _package

from packages.contracts import (
    GraphNodeStatus,
)
from packages.graph_engine import (
    GraphEngineError,
    apply_graph_delta,
    canonical_checksum,
    plan_graph_delta,
)
from packages.rule_engine import (
    RuleEngineError,
    compute_rule_package_checksum,
    evaluate_propagation,
    propagate,
    snapshot_binding_for,
    transition_reset_audit_from_delta,
    validate_rule_package,
)


def _reset_candidate():
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    graph = graph.model_copy(
        update={
            "nodes": (
                graph.nodes[0].model_copy(update={"status": GraphNodeStatus.EXCLUDED}),
                *graph.nodes[1:],
            ),
        }
    )
    package = _package(graph, (gate,))
    result = propagate(graph, package, _context(package, snapshot_graph=graph))
    assert result.candidate_delta is not None
    return graph, result.candidate_delta


@pytest.mark.parametrize("field", ("expected_package_version", "expected_package_checksum"))
def test_evaluate_cannot_bypass_the_context_rule_identity(field):
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    package = _package(graph, (gate,))
    validated = validate_rule_package(
        graph,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    value = "9.0.0" if field.endswith("version") else "sha256:" + "0" * 64
    context = replace(_context(package, snapshot_graph=graph), **{field: value})
    with pytest.raises(RuleEngineError):
        evaluate_propagation(graph, validated, context)


@pytest.mark.parametrize("mutation", ("all", "marker", "reason", "target", "reference"))
def test_reset_metadata_must_be_rejected_by_plan_itself(mutation):
    graph, delta = _reset_candidate()
    prefix = "RULE:TRANSITION_RESET_"
    remove = {
        "marker": "RULE:TRANSITION_RESET_V1",
        "reason": "RULE:TRANSITION_RESET_REQUIRED",
        "target": "RULE:TRANSITION_RESET_OP_000_DERIVED_SUPPORTED",
    }.get(mutation)
    reasons = tuple(
        code
        for code in delta.reason_codes
        if not (mutation == "all" and str(code).startswith(prefix)) and str(code) != remove
    )
    refs = tuple(
        ref
        for ref in delta.provenance.rule_versions
        if not (mutation in {"all", "reference"} and str(ref.code).startswith(prefix))
    )
    broken = delta.model_copy(
        update={
            "reason_codes": reasons,
            "provenance": delta.provenance.model_copy(update={"rule_versions": refs}),
        }
    )
    before = graph.model_dump(mode="json")
    with pytest.raises(GraphEngineError):
        plan_graph_delta(graph, broken)
    assert graph.model_dump(mode="json") == before


@pytest.mark.parametrize("target", (GraphNodeStatus.EXCLUDED, GraphNodeStatus.UNKNOWN))
def test_reset_audit_rejects_a_final_target_inconsistent_with_its_policy(target):
    _, delta = _reset_candidate()
    old = "RULE:TRANSITION_RESET_OP_000_DERIVED_SUPPORTED"
    new = f"RULE:TRANSITION_RESET_OP_000_DERIVED_{target.value}"
    ref = next(item for item in delta.provenance.rule_versions if str(item.code) == old)
    replacement = ref.model_copy(
        update={
            "code": new,
            "checksum": canonical_checksum({"strategy_version": ref.version, "reason_code": new}),
        }
    )
    broken = delta.model_copy(
        update={
            "reason_codes": tuple(new if str(code) == old else code for code in delta.reason_codes),
            "provenance": delta.provenance.model_copy(
                update={
                    "rule_versions": tuple(
                        replacement if item == ref else item
                        for item in delta.provenance.rule_versions
                    ),
                }
            ),
        }
    )
    with pytest.raises(RuleEngineError):
        transition_reset_audit_from_delta(broken)


@pytest.mark.parametrize(
    "operator,decisive,stop",
    (
        ("OR", GraphNodeStatus.SUPPORTED, "GOAL_SUPPORTED"),
        ("AND", GraphNodeStatus.REFUTED, "GOAL_REFUTED"),
    ),
)
def test_decisive_applied_gate_stops_without_work_on_irrelevant_branch(operator, decisive, stop):
    graph, gate = _gate_graph(operator, (decisive, GraphNodeStatus.PENDING))
    package = _package(graph, (gate,))
    first = propagate(graph, package, _context(package, snapshot_graph=graph))
    assert first.candidate_delta is not None
    updated = apply_graph_delta(graph, plan_graph_delta(graph, first.candidate_delta)).instance
    second = propagate(updated, package, _context(package, snapshot_graph=updated))
    assert second.candidate_delta is None
    assert second.stop_suggestion is not None
    assert second.stop_suggestion.reason.value == stop
    assert not second.legal_actions
    assert second.evaluation.node_results[0].unresolved_nodes


def test_version_only_old_validated_artifact_cannot_be_laundered_by_outer_binding():
    graph, gate = _gate_graph("AND", (GraphNodeStatus.SUPPORTED,))
    package = _package(graph, (gate,))
    validated = validate_rule_package(
        graph,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    updated = graph.model_copy(update={"graph_version": graph.graph_version + 1})
    laundered = replace(
        validated,
        snapshot_binding=snapshot_binding_for(updated),
        canonical_checksum=compute_rule_package_checksum(package),
    )
    with pytest.raises(RuleEngineError):
        evaluate_propagation(updated, laundered, _context(package, snapshot_graph=updated))


@pytest.mark.parametrize("seed", range(24))
def test_random_gate_truth_and_rule_input_permutations(seed):
    rng = random.Random(seed)
    operator = rng.choice(("AND", "OR"))
    states = tuple(rng.choice(tuple(GraphNodeStatus)) for _ in range(rng.randint(1, 5)))
    graph, gate = _gate_graph(operator, states)
    package = _package(graph, (gate,))
    context = _context(package, snapshot_graph=graph)
    baseline = propagate(graph, package, context)
    if operator == "AND":
        expected = (
            GraphNodeStatus.SUPPORTED
            if all(state is GraphNodeStatus.SUPPORTED for state in states)
            else GraphNodeStatus.REFUTED
            if GraphNodeStatus.REFUTED in states
            else None
        )
    else:
        expected = (
            GraphNodeStatus.SUPPORTED
            if GraphNodeStatus.SUPPORTED in states
            else GraphNodeStatus.REFUTED
            if all(state is GraphNodeStatus.REFUTED for state in states)
            else None
        )
    assert baseline.evaluation.node_results[0].derived_status is expected
    for permutation in itertools.islice(itertools.permutations(gate.child_node_codes), 8):
        changed_package = replace(package, gates=(replace(gate, child_node_codes=permutation),))
        assert propagate(graph, changed_package, context) == baseline
    if baseline.candidate_delta is not None:
        plan = plan_graph_delta(graph, baseline.candidate_delta)
        application = apply_graph_delta(graph, plan)
        assert application.instance.graph_version == graph.graph_version + 1
        assert graph.nodes[0].status is GraphNodeStatus.PENDING


@pytest.mark.parametrize("item_kind", ("validation", "capability"))
@pytest.mark.parametrize("mutation", ("bare", "old", "outer_binding", "outer_checksums"))
def test_per_item_snapshot_identity_cannot_be_replaced_by_context_or_wrapper(item_kind, mutation):
    from test_rule_engine_03c import _gap, _validation

    from packages.contracts import (
        ClinicalEffect,
        EvidenceObservationStatus,
        EvidenceValidity,
        QCStatus,
        ToolExecutionStatus,
    )
    from packages.rule_engine import bind_capability_gap, bind_validation

    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    if item_kind == "validation":
        item = _validation(
            graph,
            "EVIDENCE:SYNTHETIC_1",
            tool=ToolExecutionStatus.FAILED,
            qc=QCStatus.UNASSESSED,
            observation=EvidenceObservationStatus.UNOBSERVED,
            validity=EvidenceValidity.INCONCLUSIVE,
            effect=ClinicalEffect.NONE,
        )
        bound = bind_validation(graph, item)
        field = "validations"
        kind = "VALIDATION_RESULT"
    else:
        item = _gap(graph, "EVIDENCE:SYNTHETIC_1")
        bound = bind_capability_gap(graph, item)
        field = "capability_gaps"
        kind = "CAPABILITY_GAP"
    updated = graph.model_copy(update={"graph_version": 2})
    if mutation == "bare":
        changed = item
    elif mutation == "old":
        changed = bound
    else:
        changed = replace(bound, source_snapshot=snapshot_binding_for(updated))
        if mutation == "outer_checksums":
            changed = replace(
                changed,
                binding_checksum=canonical_checksum(
                    {
                        "payload_version": changed.payload_version,
                        "kind": kind,
                        "item": changed.item,
                        "source_snapshot": changed.source_snapshot,
                        "item_checksum": changed.item_checksum,
                    }
                ),
            )
    package = _package(updated, (gate,))
    context = replace(_context(package, snapshot_graph=updated), **{field: (changed,)})
    result = propagate(updated, package, context)
    assert result.knowledge_gaps
    assert result.candidate_delta is None
    assert not result.evaluation.blocked_evidence


def test_bound_input_permutations_and_duplicates_preserve_complete_result():
    from test_rule_engine_03c import _gap, _validation

    from packages.contracts import (
        ClinicalEffect,
        EvidenceObservationStatus,
        EvidenceValidity,
        QCStatus,
        ToolExecutionStatus,
    )
    from packages.rule_engine import bind_capability_gap, bind_validation

    graph, gate = _gate_graph("OR", (GraphNodeStatus.PENDING,) * 3)
    package = _package(graph, (gate,))
    validations = tuple(
        bind_validation(
            graph,
            _validation(
                graph,
                f"EVIDENCE:SYNTHETIC_{index}",
                tool=ToolExecutionStatus.FAILED,
                qc=QCStatus.UNASSESSED,
                observation=EvidenceObservationStatus.UNOBSERVED,
                validity=EvidenceValidity.INCONCLUSIVE,
                effect=ClinicalEffect.NONE,
            ),
        )
        for index in (1, 2)
    )
    gaps = tuple(
        bind_capability_gap(graph, _gap(graph, f"EVIDENCE:SYNTHETIC_{index}")) for index in (2, 3)
    )
    context = replace(
        _context(package, snapshot_graph=graph), validations=validations, capability_gaps=gaps
    )
    baseline = propagate(graph, package, context)
    for left in itertools.permutations(validations):
        for right in itertools.permutations(gaps):
            mutated = replace(
                context, validations=(*left, left[0]), capability_gaps=(*right, right[0])
            )
            assert propagate(graph, package, mutated) == baseline


@pytest.mark.parametrize("version", (True, 1.0))
@pytest.mark.parametrize("target", ("validated", "validation", "capability"))
def test_snapshot_bindings_reject_python_equal_noncanonical_field_types(version, target):
    from test_rule_engine_03c import _gap, _validation

    from packages.contracts import (
        ClinicalEffect,
        EvidenceObservationStatus,
        EvidenceValidity,
        QCStatus,
        ToolExecutionStatus,
    )
    from packages.rule_engine import bind_capability_gap, bind_validation

    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validated = validate_rule_package(
        graph,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    context = _context(package, snapshot_graph=graph)
    fake_binding = replace(snapshot_binding_for(graph), graph_version=version)
    if target == "validated":
        validated = replace(validated, snapshot_binding=fake_binding)
    else:
        if target == "validation":
            bound = bind_validation(
                graph,
                _validation(
                    graph,
                    "EVIDENCE:SYNTHETIC_1",
                    tool=ToolExecutionStatus.FAILED,
                    qc=QCStatus.UNASSESSED,
                    observation=EvidenceObservationStatus.UNOBSERVED,
                    validity=EvidenceValidity.INCONCLUSIVE,
                    effect=ClinicalEffect.NONE,
                ),
            )
            field, kind = "validations", "VALIDATION_RESULT"
        else:
            bound = bind_capability_gap(graph, _gap(graph, "EVIDENCE:SYNTHETIC_1"))
            field, kind = "capability_gaps", "CAPABILITY_GAP"
        bound = replace(bound, source_snapshot=fake_binding)
        bound = replace(
            bound,
            binding_checksum=canonical_checksum(
                {
                    "payload_version": bound.payload_version,
                    "kind": kind,
                    "item": bound.item,
                    "source_snapshot": fake_binding,
                    "item_checksum": bound.item_checksum,
                }
            ),
        )
        context = replace(context, **{field: (bound,)})
    with pytest.raises(RuleEngineError):
        evaluate_propagation(graph, validated, context)


@pytest.mark.parametrize("budget", (None, object(), (4, 0, 4, True)))
def test_public_evaluate_rejects_malformed_budget_with_rule_error(budget):
    graph, gate = _gate_graph("AND", (GraphNodeStatus.PENDING,))
    package = _package(graph, (gate,))
    validated = validate_rule_package(
        graph,
        package,
        expected_version=package.package_version,
        expected_checksum=package.package_checksum,
    )
    with pytest.raises(RuleEngineError):
        evaluate_propagation(
            graph, validated, replace(_context(package, snapshot_graph=graph), budget=budget)
        )


@pytest.mark.parametrize(
    "mutation", ("missing_all", "wrong_operation", "wrong_index", "duplicate_ref")
)
def test_reset_event_boundary_rejects_forgery_even_after_checksums_are_recomputed(mutation):
    from packages.graph_engine import (
        accepted_delta_event_from_payload,
        accepted_delta_event_to_payload,
    )

    graph, delta = _reset_candidate()
    event = plan_graph_delta(graph, delta).accepted_event
    payload = accepted_delta_event_to_payload(event)
    if mutation == "missing_all":
        payload["delta"]["reason_codes"] = [
            code
            for code in payload["delta"]["reason_codes"]
            if not code.startswith("RULE:TRANSITION_RESET_")
        ]
        payload["delta"]["provenance"]["rule_versions"] = [
            ref
            for ref in payload["delta"]["provenance"]["rule_versions"]
            if not ref["code"].startswith("RULE:TRANSITION_RESET_")
        ]
    elif mutation == "wrong_operation":
        payload["delta"]["operations"][0]["to_status"] = "SUPPORTED"
    elif mutation == "wrong_index":
        payload["delta"]["reason_codes"] = [
            code.replace("OP_000_", "OP_001_") for code in payload["delta"]["reason_codes"]
        ]
    else:
        refs = payload["delta"]["provenance"]["rule_versions"]
        refs.append(next(ref for ref in refs if ref["code"] == "RULE:TRANSITION_RESET_REQUIRED"))
    payload["delta_checksum"] = canonical_checksum(payload["delta"])
    payload["event_checksum"] = canonical_checksum(
        {key: value for key, value in payload.items() if key != "event_checksum"}
    )
    with pytest.raises(GraphEngineError):
        accepted_delta_event_from_payload(payload)


def test_mixed_derived_changes_commit_reset_round_before_ordinary_updates_and_replay():
    from test_rule_engine_03c import RULE_REF, _graph, _node

    from packages.contracts import EdgeType, GraphEdge, NodeType
    from packages.graph_engine import (
        accepted_delta_event_from_payload,
        accepted_delta_event_to_payload,
        replay_accepted_deltas,
    )
    from packages.rule_engine import GateDefinition, PropagationBudget

    goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, GraphNodeStatus.PENDING)
    claims = (
        _node("CLAIM:SYNTHETIC_LEFT", NodeType.CLAIM, GraphNodeStatus.EXCLUDED),
        _node("CLAIM:SYNTHETIC_RIGHT", NodeType.CLAIM, GraphNodeStatus.PENDING),
    )
    evidence = tuple(
        _node(f"EVIDENCE:SYNTHETIC_{index}", NodeType.EVIDENCE, GraphNodeStatus.SUPPORTED)
        for index in (1, 2)
    )
    edges = tuple(
        GraphEdge(
            source_node_code=child.node_code,
            target_node_code=parent.node_code,
            edge_type=kind,
            rule_ref=RULE_REF,
        )
        for child, parent, kind in (
            (evidence[0], claims[0], EdgeType.SUPPORTS),
            (evidence[1], claims[1], EdgeType.SUPPORTS),
            (claims[0], goal, EdgeType.AND_CHILD),
            (claims[1], goal, EdgeType.AND_CHILD),
        )
    )
    initial = _graph((goal, *claims, *evidence), edges)
    gate = GateDefinition(
        "RULE:SYNTHETIC_AND",
        str(goal.node_code),
        "AND",
        tuple(str(node.node_code) for node in claims),
        RULE_REF,
    )
    package = _package(initial, (gate,))
    current = initial
    events = []
    expected_changes = (
        (("CLAIM:SYNTHETIC_LEFT", GraphNodeStatus.PENDING),),
        (
            ("CLAIM:SYNTHETIC_LEFT", GraphNodeStatus.SUPPORTED),
            ("CLAIM:SYNTHETIC_RIGHT", GraphNodeStatus.SUPPORTED),
        ),
        (("GOAL:SYNTHETIC_SCREEN", GraphNodeStatus.SUPPORTED),),
    )
    for index, expected in enumerate(expected_changes):
        result = propagate(
            current,
            package,
            _context(
                package, snapshot_graph=current, budget=PropagationBudget(5, index, 5 - index, True)
            ),
        )
        assert result.candidate_delta is not None
        assert (
            tuple((str(op.node_code), op.to_status) for op in result.candidate_delta.operations)
            == expected
        )
        plan = plan_graph_delta(current, result.candidate_delta)
        current = apply_graph_delta(current, plan).instance
        events.append(
            accepted_delta_event_from_payload(accepted_delta_event_to_payload(plan.accepted_event))
        )
    assert replay_accepted_deltas(initial, tuple(events)) == current
    assert (
        transition_reset_audit_from_delta(events[0].delta).entries[0].derived_status
        is GraphNodeStatus.SUPPORTED
    )
    final = propagate(current, package, _context(package, snapshot_graph=current))
    assert final.stop_suggestion.reason.value == "GOAL_SUPPORTED"


@pytest.mark.parametrize(
    "case", ("or_extra_refutation", "and_extra_support", "no_gate", "gate_child_relation")
)
def test_decisive_gate_does_not_hide_unresolved_explicit_relation(case):
    from test_rule_engine_03c import RULE_REF, _graph, _node

    from packages.contracts import DataAvailability, EdgeType, GraphEdge, NodeType, PlanAction

    status = GraphNodeStatus.REFUTED if case == "and_extra_support" else GraphNodeStatus.SUPPORTED
    if case == "no_gate":
        goal = _node("GOAL:SYNTHETIC_SCREEN", NodeType.GOAL, status)
        first = _node("EVIDENCE:SYNTHETIC_1", NodeType.EVIDENCE, GraphNodeStatus.SUPPORTED)
        graph = _graph(
            (goal, first),
            (
                GraphEdge(
                    source_node_code=first.node_code,
                    target_node_code=goal.node_code,
                    edge_type=EdgeType.SUPPORTS,
                    rule_ref=RULE_REF,
                ),
            ),
        )
        gates = ()
    else:
        graph, gate = _gate_graph(
            "AND" if case == "and_extra_support" else "OR", (status, GraphNodeStatus.PENDING)
        )
        gates = (gate,)
        graph = graph.model_copy(
            update={
                "nodes": (graph.nodes[0].model_copy(update={"status": status}), *graph.nodes[1:])
            }
        )
    target = "EVIDENCE:SYNTHETIC_2" if case == "gate_child_relation" else "EVIDENCE:SYNTHETIC_3"
    source = _node(target, NodeType.EVIDENCE, GraphNodeStatus.PENDING, DataAvailability.ACQUIRABLE)
    nodes = tuple(source if str(node.node_code) == target else node for node in graph.nodes)
    if not any(str(node.node_code) == target for node in nodes):
        nodes = (*nodes, source)
    edge = GraphEdge(
        source_node_code=target,
        target_node_code=graph.goal_node_code,
        edge_type=EdgeType.SUPPORTS if case == "and_extra_support" else EdgeType.REFUTES,
        rule_ref=RULE_REF,
    )
    graph = graph.model_copy(update={"nodes": nodes, "edges": (*graph.edges, edge)})
    package = _package(graph, gates)
    result = propagate(graph, package, _context(package, snapshot_graph=graph))
    assert result.candidate_delta is None
    assert result.stop_suggestion is None
    assert any(
        action.target_node_code == target
        and action.action
        in {
            PlanAction.REQUEST_DATA,
            PlanAction.SWITCH_PATH,
        }
        for action in result.legal_actions
    )
