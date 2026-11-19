"""Independent, synthetic mechanism snapshots; these are not clinical golden cases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel

from packages.contracts import DataAvailability, EdgeType, GraphNodeStatus
from packages.graph_engine import (
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    apply_graph_delta,
    canonical_checksum,
    graph_structure_hash,
    plan_graph_delta,
    replay_accepted_deltas,
)
from packages.rule_engine import (
    PropagationBudget,
    PropagationContext,
    PropagationHistory,
    propagate,
    snapshot_binding_for,
)
from tests.graph.test_rule_engine_03c import _gate_graph, _package

GOLDEN = Path(__file__).with_name("fixtures") / "03d_synthetic_graphs.json"
SCENARIOS = (
    "and_supported",
    "or_mixed",
    "refuted",
    "explicit_conflict",
    "excluded",
    "falsified",
    "unavailable",
    "alternative",
    "budget_stop",
    "fixed_point",
)


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if value is None or type(value) in {str, int, bool, float}:
        return value
    raise TypeError(type(value).__name__)


def golden_record(name: str) -> dict[str, object]:
    """Check a handwritten semantic oracle before comparing full frozen output."""
    status = GraphNodeStatus
    operator = "OR" if name in {"or_mixed", "alternative"} else "AND"
    states = (status.SUPPORTED, status.SUPPORTED)
    availability = (DataAvailability.AVAILABLE, DataAvailability.AVAILABLE)
    expected_derived = status.SUPPORTED
    expected_stop = "GOAL_SUPPORTED"
    if name == "or_mixed":
        states = (status.SUPPORTED, status.REFUTED)
    elif name == "refuted":
        states = (status.SUPPORTED, status.REFUTED)
        expected_derived, expected_stop = status.REFUTED, "GOAL_REFUTED"
    elif name in {"unavailable", "alternative"}:
        states = (status.PENDING, status.PENDING)
        availability = (
            DataAvailability.UNAVAILABLE,
            DataAvailability.ACQUIRABLE if name == "alternative" else DataAvailability.UNAVAILABLE,
        )
        expected_derived = None
        expected_stop = "INSUFFICIENT_EVIDENCE" if name == "unavailable" else None
    graph, gate = _gate_graph(operator, states, availabilities=availability)
    gates = (gate,)
    if name in {"explicit_conflict", "excluded", "falsified"}:
        relation = {
            "explicit_conflict": EdgeType.REFUTES,
            "excluded": EdgeType.EXCLUDES,
            "falsified": EdgeType.FALSIFIES,
        }[name]
        edges = (
            graph.edges[0].model_copy(update={"edge_type": EdgeType.SUPPORTS}),
            graph.edges[1].model_copy(update={"edge_type": relation}),
        )
        if name != "explicit_conflict":
            edges = tuple(edge.model_copy(update={"edge_type": relation}) for edge in edges)
        graph = graph.model_copy(update={"edges": edges})
        gates = ()
        expected_derived, expected_stop = {
            "explicit_conflict": (status.CONFLICTED, "CONFLICTING_EVIDENCE"),
            "excluded": (status.EXCLUDED, "INSUFFICIENT_EVIDENCE"),
            "falsified": (status.REFUTED, "GOAL_REFUTED"),
        }[name]
    package = _package(graph, gates)
    context = PropagationContext(
        package.package_version,
        package.package_checksum,
        snapshot_binding_for(graph),
        budget=PropagationBudget(5, 0, 5, True),
    )
    if name == "budget_stop":
        context = replace(context, budget=PropagationBudget(5, 0, 0, True))
        expected_stop = "BUDGET_EXHAUSTED"
    elif name == "fixed_point":
        context = replace(context, history=PropagationHistory((canonical_checksum(graph),)))
        expected_stop = "INSUFFICIENT_EVIDENCE"
    original = canonical_checksum(graph)
    result = propagate(graph, package, context)
    assert result == propagate(graph, package, context)
    assert not result.knowledge_gaps
    goal_result = next(
        item for item in result.evaluation.node_results if item.node_code == graph.goal_node_code
    )
    assert goal_result.derived_status is expected_derived
    events = ()
    final = graph
    settled = result
    if result.candidate_delta is not None:
        assert result.stop_suggestion is None
        application = apply_graph_delta(graph, plan_graph_delta(graph, result.candidate_delta))
        final = application.instance
        assert final.graph_version == graph.graph_version + 1
        events = (accepted_delta_event_to_payload(application.accepted_event),)
        decoded = tuple(accepted_delta_event_from_payload(event) for event in events)
        replayed = replay_accepted_deltas(graph, decoded)
        assert replayed == final
        assert graph_structure_hash(replayed) == graph_structure_hash(final)
        assert canonical_checksum(replayed) == canonical_checksum(final)
        settled = propagate(
            final, package, replace(context, snapshot_binding=snapshot_binding_for(final))
        )
        assert settled.candidate_delta is None
    assert canonical_checksum(graph) == original
    stop = settled.stop_suggestion.reason.value if settled.stop_suggestion else None
    assert stop == expected_stop
    if name == "alternative":
        assert {action.action.value for action in result.legal_actions} == {"SWITCH_PATH"}
    if name == "or_mixed":
        assert result.evaluation.refutation.contributing_nodes
        assert not result.evaluation.conflict.contributing_nodes
    return _json_value(
        {
            "input_graph": graph,
            "input_rule_package": package,
            "input_context": context,
            "result": result,
            "accepted_events": events,
            "final_graph": final,
            "final_checksum": canonical_checksum(final),
            "final_structure_hash": graph_structure_hash(final),
            "settled_result": settled,
        }
    )


@pytest.mark.parametrize("name", SCENARIOS)
def test_synthetic_golden_graph_state_trace_actions_and_replay(name: str) -> None:
    expected = json.loads(GOLDEN.read_text())
    assert expected["fixture_version"] == "03D_SYNTHETIC_MECHANISMS_V1"
    assert golden_record(name) == expected["scenarios"][name]


@pytest.mark.parametrize("hash_seed", ["1", "23", "random"])
def test_golden_output_is_identical_across_fresh_hash_seeds(hash_seed: str) -> None:
    code = """
import json
from tests.graph.test_golden_graphs_03d import GOLDEN, SCENARIOS, golden_record
expected = json.loads(GOLDEN.read_text())['scenarios']
assert {name: golden_record(name) for name in SCENARIOS} == expected
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ | {"PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
