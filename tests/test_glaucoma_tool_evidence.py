"""New tool binding and old graph compatibility, with synthetic patient facts."""

import copy
import json

import pytest

from apps.evidence_graph_explanation import explain
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_tool_evidence import NODE, upgrade
from tests.test_glaucoma_branch_measurement import fixture as attachment_fixture
from tests.test_glaucoma_structure_ui import browser


def connect_synthetic_inputs(value):
    """Complete the old attachment fixture's missing upstream node registration."""
    value = copy.deepcopy(value)
    branch = value["branch_geometry"]
    vein = next(
        a
        for a in value["case_evidence"]["artifacts"]
        if a["checksum"] == branch["record"]["input_bindings"]["vein_mask"]["sha256"]
    )
    node = next(n for n in value["nodes"] if n["key"] == "VEIN_MASK")
    node["patient_evidence"]["artifact_codes"] = [vein["asset_code"]]
    node["patient_evidence"]["tool"] = {"code": "synthetic_avod_fixture", "version": "1.0.0"}
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    return value


def fixture(monkeypatch):
    old, measured = attachment_fixture(monkeypatch)
    return old, connect_synthetic_inputs(measured)


def test_binding_updates_actual_nodes_edges_and_explanations(monkeypatch):
    old, measured = fixture(monkeypatch)
    preserved = copy.deepcopy(measured)
    current = upgrade(measured)
    assert measured == preserved
    for value in [old, measured, current, json.loads(json.dumps(current))]:
        validate_snapshot(value)
    assert upgrade(current) == current
    assert current["research_criteria"] == old["research_criteria"]
    assert current["case_completion"] == old["case_completion"]
    assert current["branch_geometry"] == measured["branch_geometry"]
    assert (
        current["source_associations"]["knowledge_records"]
        == old["source_associations"]["knowledge_records"]
    )
    nodes = {n["key"]: n for n in current["nodes"]}
    result = nodes[NODE]["patient_evidence"]
    assert result["status"] == "已有结果" and "90.00°" in result["summary"]
    assert "待补" not in nodes["VASCULAR_ASSOC"]["patient_evidence"]["summary"]
    assert result["patient_support_score"] is None
    flow = current["source_associations"]["tool_flows"][0]
    assert flow["input_combination"] == "AND"
    assert {i["node"] for i in flow["required_inputs"]} == {"VEIN_MASK", "DISC_MASK"}
    edges = [e for e in current["evidence_structure"]["dependency_edges"] if e["to"] == NODE]
    assert {e["from"] for e in edges} == {"VEIN_MASK", "DISC_MASK"}
    assert all(e["input_combination"] == "AND" for e in edges)
    assert explain(current)["nodes"][NODE]["result_summary"] == result["summary"]
    for key in ["VEIN_MASK", "DISC_MASK"]:
        role = current["evidence_structure"]["node_roles"][key]
        assert role["group"] == "dependency" and NODE in role["serves"]
    # Immutable historical v1.0 really did retain the pending node beneath its UI.
    historic_node = next(n for n in measured["nodes"] if n["key"] == NODE)
    assert historic_node["patient_evidence"]["status"] == "算法待补"


@pytest.mark.parametrize(
    "mutation", ["score", "port", "producer", "input", "output", "node", "edge"]
)
def test_rehash_cannot_change_tool_flow(monkeypatch, mutation):
    _, measured = fixture(monkeypatch)
    value = upgrade(measured)
    a = value["source_associations"]
    flow = a["tool_flows"][0]
    if mutation == "score":
        flow["patient_support_score"] = 0.8
    elif mutation == "port":
        flow["required_inputs"][0]["port"] = "disc_mask"
    elif mutation == "producer":
        flow["required_inputs"][0]["producer"]["version"] = "9.0.0"
    elif mutation == "input":
        flow["required_inputs"][0]["checksum"] = "sha256:" + "f" * 64
    elif mutation == "output":
        flow["output_ref"]["asset_code"] = "ASSET:OID_OTHER"
    elif mutation == "node":
        next(n for n in value["nodes"] if n["key"] == NODE)["patient_evidence"]["summary"] = "0°"
    else:
        next(e for e in a["dependency_edges"] if e["to"] == NODE)["from"] = "VESSEL_MASK"
    a["checksum"] = checksum({k: v for k, v in a.items() if k != "checksum"})
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    with pytest.raises(ValueError, match="association"):
        validate_snapshot(value)


def test_measurement_is_required_not_automatically_invented(monkeypatch):
    old, _ = fixture(monkeypatch)
    with pytest.raises(ValueError, match="association_measurement_required"):
        upgrade(old)


def test_same_input_mask_must_be_the_selected_graph_input(monkeypatch):
    from apps.glaucoma_tool_evidence import projection

    _, value = fixture(monkeypatch)
    next(n for n in value["nodes"] if n["key"] == "VEIN_MASK")["patient_evidence"][
        "artifact_codes"
    ] = []
    with pytest.raises(ValueError, match="association_selected_mask_mismatch"):
        projection(value, value["source_associations"]["nodes"], [])


def test_html_shows_source_to_inputs_to_result_without_stale_gap(monkeypatch):
    _, measured = fixture(monkeypatch)
    value = upgrade(measured)
    browser(
        value,
        r"""
check(text(elements.detail).includes('已贯通的取证链'),'flow missing at overview');
check(!text(elements.detail).includes('分支角算法待补'),'parent still pending');
run("show('VENULAR_BRANCH_ANGLE')");
check(text(elements.detail).includes('90.00°'),'measurement absent');
check(text(elements.detail).includes('同时需要以上两个输入'),'AND hidden');
check(text(elements.detail).includes('arccos'),'formula missing');
check(!text(elements.detail).includes('算法待补'),'pending text remains');
check(text(elements.detail).includes('同算法的对照分布'),'comparison boundary missing');
run("show('VEIN_MASK')");
check(text(elements.detail).includes('静脉分支角'),'input consumer missing');
check(run("S.node_roles.VEIN_MASK.group")==='dependency','mask left unconnected');
run('for(const n of D.nodes)show(n.key)');
""",
    )
