"""Independent 03D metamorphic checks over synthetic graphs and strict input ports."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.contracts import (
    ActorType,
    CaseBundle,
    DataAvailability,
    EdgeType,
    GECEGInstance,
    GECEGTemplate,
    GraphDelta,
    GraphEdge,
    GraphNodeStatus,
    GraphUpdateSource,
    NodeType,
    ProposalKind,
    TemplateNode,
    UpdateNodeStatusOperation,
    VersionedReference,
)
from packages.graph_engine import (
    Applicability,
    GraphEngineError,
    GraphEngineErrorCode,
    InstantiationContext,
    NodeResolution,
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    apply_graph_delta,
    canonical_checksum,
    graph_structure_hash,
    instantiate_template,
    plan_graph_delta,
    replay_accepted_deltas,
    validate_instance_structure,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "packages/contracts/examples/v1"


def _example(model_type: type, name: str):
    return model_type.model_validate_json((EXAMPLES / f"{name}.valid.json").read_text())


def _rule_ref() -> VersionedReference:
    return VersionedReference(
        code="RULE:SYNTHETIC_03D", version="1.0.0", checksum="sha256:" + "d" * 64
    )


def _context(template: GECEGTemplate, resolutions: tuple[NodeResolution, ...]):
    return InstantiationContext(
        graph_code="GRAPH:OID_GRAPH03D1",
        graph_version=1,
        template_checksum=canonical_checksum(template),
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
        generator_version="03D.1.0",
        source_codes=("SOURCE:SYNTHETIC_03D", "SOURCE:SYNTHETIC_03D_ALT"),
        resolutions=resolutions,
        rule_versions=(_rule_ref(),),
        run_id="RUN:OID_RUN03D1",
    )


def _synthetic_template(seed: int) -> GECEGTemplate:
    """Generate a rooted tree independently of graph_engine's connectivity routine."""
    rng = random.Random(seed)
    base = _example(GECEGTemplate, "GECEGTemplate")
    nodes = [base.nodes[0]]
    edges = []
    for index in range(rng.randint(3, 9)):
        node = TemplateNode(
            node_code=f"EVIDENCE:SYNTHETIC_03D_{index}",
            node_type=NodeType.EVIDENCE,
            concept_code=f"CONCEPT:SYNTHETIC_03D_{index}",
            required=False,
        )
        parent = rng.choice(nodes)
        nodes.append(node)
        edges.append(
            GraphEdge(
                source_node_code=node.node_code,
                target_node_code=parent.node_code,
                edge_type=EdgeType.SUPPORTS,
                rule_ref=_rule_ref(),
            )
        )
    return GECEGTemplate.model_validate(
        {**base.model_dump(mode="python"), "nodes": tuple(nodes), "edges": tuple(edges)}
    )


def _rule_delta(base: GECEGInstance, operations: tuple, *, key: int = 1) -> GraphDelta:
    template = _example(GraphDelta, "GraphDelta")
    payload = template.model_dump(mode="python")
    payload.update(
        graph_code=base.graph_code,
        base_version=base.graph_version,
        scope=base.scope,
        laterality=base.laterality,
        coordinate_space=base.coordinate_space,
        unit=base.unit,
        proposed_by=ActorType.RULE_ENGINE,
        proposal_kind=ProposalKind.GRAPH_DELTA_VALIDATION,
        reason_codes=(_rule_ref().code,),
        idempotency_key=f"TASK:OID_SYNTH03D{key}",
        operations=operations,
    )
    payload["provenance"].update(actor=ActorType.RULE_ENGINE, rule_versions=(_rule_ref(),))
    return GraphDelta.model_validate(payload)


def _operation(node, target: GraphNodeStatus) -> UpdateNodeStatusOperation:
    return UpdateNodeStatusOperation(
        op="UPDATE_NODE_STATUS",
        node_code=node.node_code,
        from_status=node.status,
        to_status=target,
        source=GraphUpdateSource.DETERMINISTIC_RULE,
    )


@pytest.mark.parametrize("seed", range(16))
def test_random_legal_graphs_set_permutation_complete_checksum_trace_and_replay(seed: int):
    rng = random.Random(seed)
    template = _synthetic_template(seed)
    case = _example(CaseBundle, "CaseBundle")
    resolutions = tuple(
        NodeResolution(
            node_code=str(node.node_code),
            applicability=Applicability.APPLICABLE,
            availability=DataAvailability.AVAILABLE,
            observation_concept_codes=("CONCEPT:IMAGE_AVAILABLE",),
        )
        for node in template.nodes[1:]
    )
    context = _context(template, resolutions)
    before = canonical_checksum((template, case, context))
    expected = instantiate_template(template, case, context)
    for _ in range(5):
        permuted = list(resolutions)
        rng.shuffle(permuted)
        actual = instantiate_template(
            template,
            case,
            replace(
                context,
                resolutions=tuple(permuted),
                source_codes=tuple(reversed(context.source_codes)),
            ),
        )
        assert actual == expected
        assert canonical_checksum(actual.instance) == canonical_checksum(expected.instance)
        assert canonical_checksum(actual.trace) == canonical_checksum(expected.trace)
        assert graph_structure_hash(actual.instance) == graph_structure_hash(expected.instance)
    assert canonical_checksum((template, case, context)) == before

    current = expected.instance
    snapshots = [current]
    events = []
    for round_index in range(4):
        candidates = list(current.nodes)
        rng.shuffle(candidates)
        selected = candidates[: rng.randint(1, len(candidates))]
        operations = tuple(
            _operation(
                node,
                GraphNodeStatus.REFUTED
                if node.status is GraphNodeStatus.SUPPORTED
                else GraphNodeStatus.SUPPORTED,
            )
            for node in selected
        )
        delta = _rule_delta(current, operations, key=seed * 10 + round_index)
        frozen_base = canonical_checksum(current)
        plan = plan_graph_delta(current, delta)
        assert plan_graph_delta(current, delta) == plan
        applied = apply_graph_delta(current, plan)
        assert applied.instance.graph_version == current.graph_version + 1
        assert canonical_checksum(current) == frozen_base
        assert validate_instance_structure(applied.instance) == applied.instance
        event = accepted_delta_event_from_payload(
            json.loads(json.dumps(accepted_delta_event_to_payload(applied.accepted_event)))
        )
        events.append(event)
        current = applied.instance
        snapshots.append(current)
    for index, snapshot in enumerate(snapshots):
        replayed = replay_accepted_deltas(snapshot, tuple(events[index:]))
        assert replayed == current
        assert canonical_checksum(replayed) == canonical_checksum(current)
        assert graph_structure_hash(replayed) == graph_structure_hash(current)


@pytest.mark.parametrize("seed", range(8))
def test_random_pruning_keeps_only_goal_connected_graphs_or_rejects(seed: int):
    template = _synthetic_template(seed)
    case = _example(CaseBundle, "CaseBundle")
    for removed in template.nodes[1:]:
        context = _context(
            template,
            (
                NodeResolution(
                    node_code=str(removed.node_code),
                    applicability=Applicability.NOT_APPLICABLE,
                    availability=DataAvailability.NOT_APPLICABLE,
                ),
            ),
        )
        # A rooted tree retains an orphan exactly when the removed node had a child.
        has_children = any(edge.target_node_code == removed.node_code for edge in template.edges)
        if has_children:
            with pytest.raises(GraphEngineError) as error:
                instantiate_template(template, case, context)
            assert error.value.code is GraphEngineErrorCode.PRUNING_DISCONNECTS_GRAPH
        else:
            result = instantiate_template(template, case, context)
            assert validate_instance_structure(result.instance) == result.instance
            assert removed.node_code not in {node.node_code for node in result.instance.nodes}
            assert result.trace.pruned_nodes[0].node_code == removed.node_code


def test_duplicate_accepted_history_is_idempotent_but_duplicate_replay_is_rejected():
    base = _example(GECEGInstance, "GECEGInstance")
    delta = _rule_delta(base, (_operation(base.nodes[0], GraphNodeStatus.SUPPORTED),))
    original = plan_graph_delta(base, delta)
    for copies in (1, 2, 5):
        history = (original.accepted_event,) * copies
        retry = plan_graph_delta(base, delta, accepted_events=history)
        assert retry.idempotent_retry is True
        applied = apply_graph_delta(base, retry, accepted_events=history)
        assert applied.instance == original.result_instance
        assert applied.instance.graph_version == 2
    with pytest.raises(GraphEngineError) as error:
        replay_accepted_deltas(base, (original.accepted_event,) * 2)
    assert error.value.code is GraphEngineErrorCode.DELTA_REPLAY_MISMATCH
    assert base.graph_version == 1


@pytest.mark.parametrize("mutation", ("omitted", "reordered", "duplicate"))
def test_replay_event_sequence_is_ordered_and_rejects_a_single_stream_mutation(mutation: str):
    base = _example(GECEGInstance, "GECEGInstance")
    current = base
    events = []
    for index, status in enumerate(
        (GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED, GraphNodeStatus.SUPPORTED)
    ):
        delta = _rule_delta(current, (_operation(current.nodes[0], status),), key=index)
        applied = apply_graph_delta(current, plan_graph_delta(current, delta))
        events.append(applied.accepted_event)
        current = applied.instance
    assert replay_accepted_deltas(base, tuple(events)) == current
    if mutation == "omitted":
        del events[1]
    elif mutation == "reordered":
        events[1], events[2] = events[2], events[1]
    else:
        events.insert(1, events[0])
    before = base.model_dump_json()
    with pytest.raises(GraphEngineError) as error:
        replay_accepted_deltas(base, tuple(events))
    assert error.value.code is GraphEngineErrorCode.DELTA_REPLAY_MISMATCH
    assert base.model_dump_json() == before


@pytest.mark.parametrize("mutation", ("unknown", "source", "from", "duplicate", "scope"))
def test_illegal_single_mutation_after_valid_operation_has_no_partial_result(mutation: str):
    base = _example(GECEGInstance, "GECEGInstance")
    first = _operation(base.nodes[0], GraphNodeStatus.SUPPORTED)
    second = _operation(base.nodes[1], GraphNodeStatus.SUPPORTED)
    delta = _rule_delta(base, (first, second))
    expected_codes = {
        "unknown": GraphEngineErrorCode.DELTA_UNKNOWN_NODE,
        "source": GraphEngineErrorCode.DELTA_UPDATE_SOURCE_INVALID,
        "from": GraphEngineErrorCode.DELTA_STATUS_FROM_MISMATCH,
        "duplicate": GraphEngineErrorCode.DELTA_DUPLICATE_OPERATION,
        "scope": GraphEngineErrorCode.DELTA_SCOPE_MISMATCH,
    }
    if mutation == "scope":
        invalid = delta.model_copy(
            update={"scope": delta.scope.model_copy(update={"run_id": "RUN:OID_OTHER03D"})}
        )
    else:
        if mutation == "unknown":
            second = second.model_copy(update={"node_code": "EVIDENCE:UNKNOWN_03D"})
        elif mutation == "source":
            second = second.model_copy(update={"source": GraphUpdateSource.EXPERT_ASSESSMENT})
        elif mutation == "from":
            second = second.model_copy(update={"from_status": GraphNodeStatus.REFUTED})
        else:
            second = first
        invalid = delta.model_copy(update={"operations": (first, second)})
    before = base.model_dump_json()
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(base, invalid)
    assert error.value.code is expected_codes[mutation]
    assert base.model_dump_json() == before
    assert plan_graph_delta(base, delta).result_instance.graph_version == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("availability", "AVAILABLE"),
        ("applicability", "APPLICABLE"),
        ("capability", "UNAVAILABLE_CAPABILITY"),
    ),
)
def test_resolution_enum_strings_cannot_bypass_source_or_capability_checks(field, value):
    template = _example(GECEGTemplate, "GECEGTemplate")
    resolution = NodeResolution(
        node_code=str(template.nodes[1].node_code),
        applicability=Applicability.APPLICABLE,
        availability=DataAvailability.UNAVAILABLE,
    )
    context = _context(template, (replace(resolution, **{field: value}),))
    with pytest.raises(GraphEngineError) as error:
        instantiate_template(template, _example(CaseBundle, "CaseBundle"), context)
    assert error.value.code is GraphEngineErrorCode.INVALID_RESOLUTION


@pytest.mark.parametrize("mutation", ("none", "object", "scope", "status", "validation"))
def test_malformed_delta_ports_raise_stable_errors_without_partial_result(mutation: str):
    base = _example(GECEGInstance, "GECEGInstance")
    delta = _rule_delta(base, (_operation(base.nodes[0], GraphNodeStatus.SUPPORTED),))
    validation = None
    if mutation == "none":
        delta = None
    elif mutation == "object":
        delta = object()
    elif mutation == "scope":
        delta = delta.model_copy(update={"scope": None})
    elif mutation == "status":
        delta = delta.model_copy(
            update={"operations": (delta.operations[0].model_copy(update={"to_status": object()}),)}
        )
    else:
        delta = _example(GraphDelta, "GraphDelta")
        delta = delta.model_copy(
            update={
                "provenance": delta.provenance.model_copy(
                    update={
                        "rule_versions": (
                            VersionedReference(
                                code=delta.reason_codes[0],
                                version="1.0.0",
                                checksum="sha256:" + "a" * 64,
                            ),
                        )
                    }
                )
            }
        )
        validation = object()
    before = base.model_dump_json()
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(base, delta, validation_result=validation)
    assert error.value.code is (
        GraphEngineErrorCode.DELTA_VALIDATION_MISMATCH
        if mutation == "validation"
        else GraphEngineErrorCode.DELTA_CONTRACT_INVALID
    )
    assert base.model_dump_json() == before


@pytest.mark.parametrize(
    "actor",
    (
        ActorType.PLANNER,
        ActorType.TRACE_AGENT,
        ActorType.TOOL_AGENT,
        ActorType.VERIFICATION_AGENT,
        ActorType.EXPERT_AGENT,
    ),
)
def test_agent_cannot_claim_deterministic_rule_source_without_validation(actor: ActorType):
    base = _example(GECEGInstance, "GECEGInstance")
    delta = _rule_delta(base, (_operation(base.nodes[1], GraphNodeStatus.SUPPORTED),))
    forged = delta.model_copy(
        update={
            "proposed_by": actor,
            "proposal_kind": ProposalKind.GRAPH_DELTA_CANDIDATE,
            "provenance": delta.provenance.model_copy(update={"actor": actor}),
        }
    )
    if actor is ActorType.VERIFICATION_AGENT:
        # Verification is a legal candidate actor even after full contract parsing.
        forged = GraphDelta.model_validate(forged.model_dump(mode="python"))
    before = base.model_dump_json()
    with pytest.raises(GraphEngineError) as error:
        plan_graph_delta(base, forged)
    assert error.value.code is GraphEngineErrorCode.DELTA_UPDATE_SOURCE_INVALID
    assert base.model_dump_json() == before


@pytest.mark.parametrize("actor", (ActorType.RULE_ENGINE, ActorType.GRAPH_ENGINE))
def test_only_deterministic_engine_actors_may_use_rule_status_source(actor: ActorType):
    base = _example(GECEGInstance, "GECEGInstance")
    delta = _rule_delta(base, (_operation(base.nodes[0], GraphNodeStatus.SUPPORTED),))
    data = delta.model_dump(mode="python")
    data["proposed_by"] = actor
    data["provenance"]["actor"] = actor
    delta = GraphDelta.model_validate(data)
    assert (
        plan_graph_delta(base, delta).result_instance.nodes[0].status is GraphNodeStatus.SUPPORTED
    )
