"""A saved evidence tree is driven by conditions, never by available tools."""

import copy

import pytest

from apps.evidence_graph_explanation import explain
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_criteria_v1_3 import apply as criteria
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_evidence_structure import apply, build
from tests.test_glaucoma_disease_conditions import fixture


def test_all_nodes_accounted_for_and_main_paths_have_source_conditions(monkeypatch):
    old = criteria(fixture(monkeypatch))
    before = copy.deepcopy(old)
    value = apply(old)
    assert old == before
    assert value["nodes"] == old["nodes"]
    assert value["relationships"] == old["relationships"]
    assert value["case_evidence"] == old["case_evidence"]
    assert value["research_criteria"] == old["research_criteria"]
    assert apply(value) == value
    validate_snapshot(old)
    validate_snapshot(value)
    s = value["evidence_structure"]
    assert s["counts"] == {"main": 19, "dependency": 10, "deferred": 4, "auxiliary": 19}
    assert len(s["node_roles"]) == 52
    assert set(s["main_node_keys"]) == {
        key for r in value["research_criteria"]["results"] for key in r["path"]
    }
    assert len(s["main_edges"]) == 18
    for edge in s["main_edges"]:
        assert edge["criterion_ids"]
        assert edge["from"] in s["main_node_keys"] and edge["to"] in s["main_node_keys"]
    for key in s["main_node_keys"]:
        assert s["node_roles"][key]["sources"]
        assert s["node_roles"][key]["criterion_ids"]
    assert len({r["criterion_id"] for r in value["research_criteria"]["results"]}) == 10


def test_mask_and_disc_size_have_different_typed_purposes(monkeypatch):
    value = apply(criteria(fixture(monkeypatch)))
    s = value["evidence_structure"]
    edges = {e["from"]: e for e in s["dependency_edges"]}
    assert edges["DISC_MASK"]["relation"] == "measurement_input"
    assert edges["DISC_SIZE"]["relation"] == "interpretation_modifier"
    assert edges["DISC_SIZE"]["sources"]
    assert edges["DISC_MASK"]["to"] == edges["DISC_SIZE"]["to"] == "CUP_RIM_MEASURE"
    assert "分母" in edges["DISC_GEOMETRY"]["purpose"]
    assert "毫米" in s["node_roles"]["DISC_SIZE"]["demand"]
    assert s["node_roles"]["FIELD_QC"]["serves"] == ["FIELD_VALUES"]
    assert s["node_roles"]["CCT"]["serves"] == ["IOP_VALUES"]
    assert "DISC_MASK" not in s["main_node_keys"]
    assert "DISC_SIZE" not in s["main_node_keys"]
    assert s["node_roles"]["DENSITY"]["group"] == "auxiliary"
    assert s["node_roles"]["MODEL_ASSESSMENT"]["group"] == "auxiliary"
    assert s["node_roles"]["RNFL_THICKNESS"]["group"] == "deferred"
    # Missing current input does NOT prune a source-defined medical condition.
    assert "FIELD_VALUES" in s["main_node_keys"]
    assert "IOP_VALUES" in s["main_node_keys"]
    e = explain(value)
    assert e["version"] == "evidence-explanation/1.4.0"
    assert "分母" in e["nodes"]["DISC_GEOMETRY"]["rationale"]
    assert "EGS" in e["nodes"]["DISC_SIZE"]["rationale"]
    assert "像素" not in e["nodes"]["STRUCTURE"]["demand"]
    for key in ("ROOT", "STRUCTURE", "RNFL", "CONTEXT"):
        assert "直接子项" not in e["nodes"][key]["result_summary"]
        assert "按条件计数" in e["nodes"][key]["result_summary"]
    assert "关联2项" in e["nodes"]["CONTEXT"]["result_summary"]
    assert "未提供物理盘径" in e["nodes"]["DISC_SIZE"]["result_summary"]
    assert s["node_roles"]["DISC_SIZE"]["result_status"] == "当前输入无法取证"


@pytest.mark.parametrize(
    "change", ["promote_mask", "drop_node", "wrong_target", "fake_source", "explain"]
)
def test_changed_roles_dependencies_sources_cannot_pass_by_rehash(monkeypatch, change):
    value = apply(criteria(fixture(monkeypatch)))
    s = value["evidence_structure"]
    if change == "promote_mask":
        s["main_node_keys"].append("DISC_MASK")
    elif change == "drop_node":
        s["node_roles"].pop("VESSEL_FEATURES")
    elif change == "wrong_target":
        s["dependency_edges"][0]["to"] = "RISK_FIELDS"
    elif change == "fake_source":
        s["node_roles"]["DISC_SIZE"]["sources"] = []
    else:
        value["display_explanations"] = explain(value)
        value["display_explanations"]["nodes"]["DISC_MASK"]["rationale"] = "independent diagnosis"
    s["checksum"] = checksum({k: v for k, v in s.items() if k != "checksum"})
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    with pytest.raises(ValueError, match="evidence_structure_"):
        validate_snapshot(value)


def test_new_tool_or_source_free_rule_cannot_create_a_medical_branch(monkeypatch):
    value = criteria(fixture(monkeypatch))
    changed = copy.deepcopy(value)
    changed["nodes"].append({"key": "NEW_TOOL", "parent": "STRUCTURE"})
    with pytest.raises(ValueError, match="unclassified_node"):
        build(changed)
    changed = copy.deepcopy(value)
    changed["research_criteria"]["policy"]["criteria"][0]["sources"] = []
    with pytest.raises(ValueError, match="source_condition_required"):
        build(changed)


def test_actual_export_saved_layer_has_no_extra_medical_effect(monkeypatch):
    value = apply(criteria(fixture(monkeypatch)))
    assert value["evidence_structure"]["policy"]["medical_effect"] == "not_assigned"
    assert all(r["medical_effect"] == "not_assigned" for r in value["research_criteria"]["results"])
