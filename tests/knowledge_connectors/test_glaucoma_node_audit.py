import copy
import json

import pytest

from apps import glaucoma_node_audit as audit
from apps.glaucoma_node_audit import FINDINGS, PROPOSALS, SOURCES, build, markdown


def fixture_bytes():
    nodes = [
        {
            "key": key,
            "title": key,
            "parent": None,
            "database": {
                "record": {
                    "kind": "node",
                    "code": key,
                    "version": "1.0.0",
                    "checksum": "sha256:" + "0" * 64,
                }
            },
        }
        for key in FINDINGS
    ]
    graph = {"nodes": nodes}
    broad = {
        "topic_count": 22,
        "search_task_count": 66,
        "provider_hit_count": 275,
        "unique_identity_group_count": 231,
        "topic_results": [],
    }
    return graph, broad


def test_complete_unique_audit_and_no_observation_mutation(monkeypatch):
    graph, broad = fixture_bytes()
    monkeypatch.setattr(audit, "REVIEWED_GRAPH_SHA256", audit.digest(json.dumps(graph).encode()))
    before = copy.deepcopy(graph)
    result = build(json.dumps(graph).encode(), json.dumps(broad).encode())
    assert len(FINDINGS) == 52
    assert result["coverage"] == {"reviewed": 52, "total": 52, "unreviewed": []}
    assert all(not n["patient_observation_added"] for n in result["nodes"])
    assert all(set(n["sources"]) <= set(SOURCES) for n in result["nodes"])
    assert all(not p["implemented"] for p in PROPOSALS)
    assert graph == before
    assert result == build(json.dumps(graph).encode(), json.dumps(broad).encode())
    assert markdown(result).count("| ROOT ·") == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "new"])
def test_changed_node_set_cannot_look_fully_reviewed(mutation):
    graph, broad = fixture_bytes()
    if mutation == "missing":
        graph["nodes"].pop()
    elif mutation == "duplicate":
        graph["nodes"].append(graph["nodes"][0])
    else:
        graph["nodes"][0]["key"] = "NEW"
    with pytest.raises(ValueError, match="node_set_changed"):
        build(json.dumps(graph).encode(), json.dumps(broad).encode())


def test_same_nodes_changed_observation_requires_new_review(monkeypatch):
    graph, broad = fixture_bytes()
    monkeypatch.setattr(audit, "REVIEWED_GRAPH_SHA256", audit.digest(json.dumps(graph).encode()))
    graph["new_observation"] = {"value": 0.9}
    with pytest.raises(ValueError, match="snapshot_changed"):
        build(json.dumps(graph).encode(), json.dumps(broad).encode())
