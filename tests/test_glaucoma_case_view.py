import copy

import pytest

from apps.evidence_graph_html import render_html
from apps.glaucoma_case_evidence import png_shape
from apps.glaucoma_case_view import TASK_NODES, project
from tests.database.test_glaucoma_bindings import fixture


def data():
    _, _, bindings = fixture()
    snapshot = {
        "nodes": copy.deepcopy(bindings["nodes"]),
        "relationships": bindings["relationships"],
        "records": [],
        "egs_pages": [],
        "source_chunks": [],
        "historical_pages": [],
        "exported_at": "synthetic",
        "title": "synthetic",
    }
    report = {
        "case_code": "CASE:ANON_EXAMPLE01",
        "sample": "synthetic.png",
        "results": {},
        "images": {},
        "input": {"dimensions": [2, 3], "asset_code": "ASSET:OID_INPUT001"},
        "artifacts": [],
        "fairclip": {
            "outputs": [
                {
                    "output": {"normalized_model_scores": [0.2, 0.8]},
                    "source": {"artifact_code": "ASSET:OID_MODEL001"},
                }
            ]
        },
    }
    for task in TASK_NODES:
        obs = {"task": task, "model": {"foreground_pixels": {"disc_mask": 3}}}
        if task in ("density", "width", "tortuosity", "fractal"):
            obs["measurement"] = {"value": 0.3, "unit": "1", "unavailable_reason": None}
        report["results"][task] = {
            "observation": obs,
            "execution": {
                "exit_code": 0,
                "manifest": {},
                "call": {"tool": task, "version": "1.0.0"},
                "input_checksum": "synthetic",
            },
        }
    return snapshot, report


def test_every_node_has_explicit_case_state_no_diagnostic_vote():
    s, r = data()
    value = project(s, r)
    nodes = {n["key"]: n for n in value["nodes"]}
    assert len(nodes) == 52
    assert all(n["patient_evidence"]["summary"] for n in nodes.values())
    assert nodes["ROOT"]["patient_evidence"]["status"] == "待判读"
    assert nodes["DENSITY"]["patient_evidence"]["results"][0]["value"] == 0.3
    assert nodes["IOP_VALUES"]["patient_evidence"]["status"] == "当前输入无法取证"
    assert nodes["CUP_RIM_MEASURE"]["patient_evidence"]["status"] == "可补工具·待搭建"
    assert not any(
        n["patient_evidence"]["status"] in {"SUPPORTED", "REFUTED"} for n in nodes.values()
    )
    output = render_html(value)
    assert "此节点未登记具体缺项" not in output
    assert "当前为 Draft" not in output
    assert "innerHTML" not in output and "connect-src 'none'" in output


@pytest.mark.parametrize("kind", ["scope", "failure", "missing_task"])
def test_cross_scope_failure_or_missing_execution_rejected(kind):
    s, r = data()
    if kind == "scope":
        r["artifacts"] = [{"scope": {"case_code": "CASE:ANON_OTHER001"}}]
    elif kind == "failure":
        r["results"]["density"]["execution"]["exit_code"] = 1
    else:
        del r["results"]["density"]
    with pytest.raises(ValueError):
        project(s, r)


def test_no_valid_metric_is_not_zero():
    s, r = data()
    r["results"]["tortuosity"]["observation"]["measurement"].update(
        value=None, unavailable_reason="no_valid_open_segments"
    )
    v = project(s, r)
    p = next(n["patient_evidence"] for n in v["nodes"] if n["key"] == "TORTUOSITY")
    assert p["status"] == "无有效测量" and p["results"][0]["value"] is None


def test_non_image_rejected():
    with pytest.raises(ValueError):
        png_shape(b"not an image")


def test_fovea_boundary_coordinate_preserved_but_flagged():
    s, r = data()
    r["results"]["fovea"]["observation"]["model"].update(
        fovea_xy=[2.0, 1.0], fovea_status="predicted"
    )
    v = project(s, r)
    p = next(n["patient_evidence"] for n in v["nodes"] if n["key"] == "FOVEA_LOCATION")
    assert p["status"] == "待核对" and p["images"] == ["fovea_position"]
    assert any(m["value"] == [2.0, 1.0] for m in p["results"])


def test_acquisition_distinguishes_unavailable_tools_and_absent_fields():
    s, r = data()
    v = project(s, r)
    n = {n["key"]: n["patient_evidence"] for n in v["nodes"]}
    assert all("acquisition" in p for p in n.values())
    for key in ("RNFL_THICKNESS", "IOP_VALUES", "CCT_VALUES", "ANGLE_VALUES", "FIELD_QC"):
        a = n[key]["acquisition"]
        assert a["category"] == "unavailable_current_input"
        assert not a["new_model_alone_can_resolve"] and not a["automatic_retry"]
        assert n[key]["results"] == []
    cup = n["CUP_RIM_MEASURE"]["acquisition"]
    assert cup["category"] == "tool_build_candidate"
    assert not cup["execution_ready"]
    assert cup["model_candidate"]["weight_file"] == "segman_b/segman_b.pth"
    assert cup["model_candidate"]["tool_registration"] is None
    assert n["RNFL_REFLECTANCE"]["acquisition"]["category"] == "visual_review_pending"
    fields = n["RISK_FIELDS"]["acquisition"]["fields"]
    assert fields["age"]["category"] == "unavailable_current_input"
    assert fields["age"]["case_value"] is None
    assert fields["family_history"]["category"] == "unavailable_current_input"
    assert n["DISC_GEOMETRY"]["acquisition"]["category"] == "partial_observation"


def test_unavailable_branches_propagate_without_diagnostic_score_or_double_count():
    s, r = data()
    before = copy.deepcopy(r)
    v = project(s, r)
    n = {n["key"]: n["patient_evidence"] for n in v["nodes"]}
    assert r == before
    assert n["FUNCTION"]["status"] == n["PRESSURE"]["status"] == "当前输入无法取证"
    assert n["ROOT"]["status"] == "待判读"
    assert n["RNFL"]["acquisition"]["category"] == "branch_summary"
    assert v["acquisition_summary"]["unavailable_current_input"] == 8
    assert sum(v["acquisition_summary"].values()) == len(set(v["acquisition_requirements"])) == 13
    assert n["DENSITY"]["results"][0]["value"] == 0.3
    assert v["acquisition_policy"]["missing_is_negative"] is False
    assert v["acquisition_policy"]["scope"] == "selected_case_slo_and_basic_demographics"
    from apps.evidence_graph_html import validate_snapshot

    validate_snapshot(v)
    html = render_html(v)
    assert "当前输入下的取证安排" in html and "公开权重已找到" in html
