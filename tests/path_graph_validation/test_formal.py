from dataclasses import replace

import pytest

from packages.contracts import EdgeType, GraphEdge
from packages.contracts import GraphNodeStatus as S
from packages.graph_engine import canonical_checksum
from packages.path_graph_validation import FormalInputs, FormalPathReader
from tests.deterministic_pipeline.test_reporting import facts as facts
from tests.graph.test_rule_engine_03c import RULE_REF, _context, _gate_graph, _package


def reader_for(graph, gates=(), *, events=(), audits=(), snapshots=None):
    package = _package(graph, gates)
    if not gates and graph.edges:
        from packages.rule_engine import compute_rule_package_checksum

        refs = tuple(dict.fromkeys(e.rule_ref for e in graph.edges if e.rule_ref))
        package = replace(
            package,
            rule_refs=refs,
            provenance=package.provenance.model_copy(update={"rule_versions": refs}),
        )
        package = replace(package, package_checksum=compute_rule_package_checksum(package))
    source = FormalInputs(
        graph, package, _context(package, snapshot_graph=graph), tuple(events), tuple(audits)
    )
    calls = []

    def validate(source, accepted):
        assert source.graph == graph
        calls.append(accepted)

    reader = FormalPathReader(
        read_current=lambda: source,
        read_snapshot=lambda code, version: snapshots[version],
        validate_current_sources=validate,
        purpose="synthetic_mechanism",
    )
    return reader, source, calls


@pytest.mark.parametrize(
    "operator,statuses,expected",
    [
        ("AND", (S.SUPPORTED, S.PENDING), None),
        ("AND", (S.SUPPORTED, S.SUPPORTED), "SUPPORTED"),
        ("OR", (S.SUPPORTED, S.REFUTED), "SUPPORTED"),
        ("OR", (S.REFUTED, S.REFUTED), "REFUTED"),
        ("OR", (S.SUPPORTED, S.PENDING), "SUPPORTED"),
    ],
)
def test_original_gate_semantics_no_or_conflict(operator, statuses, expected):
    graph, gate = _gate_graph(operator, statuses)
    before = canonical_checksum(graph)
    reader, _, calls = reader_for(graph, (gate,))
    result = reader()
    assert result.paths[0]["result"]["derived_status"] == expected
    assert not result.review_facts["conflict"]
    assert result.paths[0]["gate"]["rule_ref"]["checksum"] == RULE_REF.checksum
    assert result.graph["graph_version"] == graph.graph_version
    assert canonical_checksum(graph) == before and len(calls) == 2


@pytest.mark.parametrize(
    "relation,expected",
    [
        (EdgeType.SUPPORTS, "SUPPORTED"),
        (EdgeType.REFUTES, "CONFLICTED"),
        (EdgeType.EXCLUDES, "CONFLICTED"),
        (EdgeType.FALSIFIES, "CONFLICTED"),
    ],
)
def test_explicit_relations_remain_distinct(relation, expected):
    graph, gate = _gate_graph("OR", (S.SUPPORTED, S.SUPPORTED))
    edge = GraphEdge(
        source_node_code=graph.nodes[2].node_code,
        target_node_code=graph.goal_node_code,
        edge_type=relation,
        rule_ref=RULE_REF,
    )
    graph = graph.model_copy(update={"edges": (*graph.edges, edge)})
    r = reader_for(graph, (gate,))[0]()
    assert r.goals[0]["derived_status"] == expected
    assert r.relations[-1]["edge_type"] == relation.value


def test_rule_version_scope_and_unaccepted_context_rejected():
    graph, gate = _gate_graph("AND", (S.SUPPORTED,))
    reader, source, _ = reader_for(graph, (gate,))
    reader.read_current = lambda: replace(
        source, context=replace(source.context, expected_package_version="old")
    )
    with pytest.raises(ValueError):
        reader()
    reader.read_current = lambda: replace(
        source, context=replace(source.context, validations=(object(),))
    )
    with pytest.raises(ValueError, match="unaccepted"):
        reader()


def test_source_withdrawal_and_snapshot_race_fail_closed():
    graph, gate = _gate_graph("AND", (S.SUPPORTED,))
    reader, source, _ = reader_for(graph, (gate,))

    def withdrawn(*args):
        raise ValueError("source_withdrawn")

    reader.validate_sources = withdrawn
    with pytest.raises(ValueError, match="source_withdrawn"):
        reader()
    reader.validate_sources = lambda *args: None
    count = []

    def read():
        count.append(1)
        return source if len(count) == 1 else replace(source, events=({"kind": "changed"},))

    reader.read_current = read
    with pytest.raises(ValueError, match="snapshot_changed"):
        reader()


def test_conflict_relation_targets_original_allowed_proposition_type():
    graph, gate = _gate_graph("OR", (S.SUPPORTED, S.SUPPORTED))
    edge = GraphEdge(
        source_node_code=graph.nodes[2].node_code,
        target_node_code=graph.nodes[1].node_code,
        edge_type=EdgeType.CONFLICTS_WITH,
        rule_ref=RULE_REF,
    )
    graph = graph.model_copy(update={"edges": (*graph.edges, edge)})
    result = reader_for(graph, (gate,))[0]()
    node = next(
        r
        for r in result.binding["trace"]["node_results"]
        if r["node_code"] == graph.nodes[1].node_code
    )
    assert node["derived_status"] == "CONFLICTED"
    assert result.review_facts["conflict"]


def test_old_accepted_source_deduplicated_and_reset_rejected(facts):
    from tests.deterministic_pipeline.test_review_07c import transition

    f = facts
    reader, source, _ = reader_for(
        f.graph, events=f.events * 2, audits=f.audits, snapshots=f.snapshots
    )
    r = reader()
    assert len(r.evidence) == 1
    assert r.evidence[0]["reference"]["source_snapshot"]["graph_version"] < r.graph["graph_version"]
    assert r.evidence[0]["effect"] == "SUPPORTS"
    graph, event = transition(f, f.graph, f.target, "SUPPORTED", "PENDING", reset_to="EXCLUDED")
    f.snapshots[graph.graph_version] = graph
    reset = reader_for(graph, events=f.events, audits=[*f.audits, event], snapshots=f.snapshots)[
        0
    ]()
    assert reset.evidence == ()
    assert reset.graph["graph_version"] == graph.graph_version
    assert next(n for n in graph.nodes if n.node_code == f.target).status is S.PENDING


@pytest.mark.parametrize("field", ["binding_checksum", "target_node_code", "run_id"])
def test_forged_accepted_trace_not_consumed(facts, field):
    import copy

    f = facts
    events = copy.deepcopy(f.events)
    bound = next(e["validation"] for e in events if e["kind"] == "validation")
    if field == "binding_checksum":
        bound[field] = "sha256:" + "0" * 64
    elif field == "target_node_code":
        bound["item"][field] = f.graph.goal_node_code
    else:
        bound["item"]["scope"][field] += "OTHER"
    r = reader_for(f.graph, events=events, audits=f.audits, snapshots=f.snapshots)[0]()
    assert not r.evidence
