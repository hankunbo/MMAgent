import copy

import pytest

from apps import glaucoma_model_selection as selection
from apps.evidence_graph_html import render_html, validate_snapshot
from apps.glaucoma_criteria_v1_2 import apply, build
from apps.glaucoma_evidence_plan import checksum
from tests.test_glaucoma_choice_and_conditions import fixture as old_fixture


def fixture():
    """Synthetic software fixture, never a patient or a production review."""
    v = old_fixture()
    c = v["case_evidence"]
    c["supplement"]["demographics"] = {
        "image_checksum": c["input"]["checksum"],
        "fields": {
            k: {"display_value": x}
            for k, x in {
                "age": 70,
                "family_history": None,
                "gender": "unknown",
                "race": "unknown",
                "ethnicity": "unknown",
                "language": "unknown",
                "maritalstatus": "unknown",
            }.items()
        },
    }
    record = c["fairclip"]["outputs"][0]
    record["output"]["input_checksum"] = c["input"]["checksum"]
    record["source"].update(
        scope={"case_code": c["case_code"]}, artifact_checksum="sha256:" + "c" * 64
    )
    return selection.apply(v)


def test_all_branches_neutral_titles_separate_metrics_and_visual_review():
    v = fixture()
    before = copy.deepcopy(v)
    out = apply(v)
    assert v == before
    validate_snapshot(out)
    r = out["research_criteria"]
    assert len(r["results"]) == 25
    assert len(r["node_summaries"]) == 52
    assert all(s["criterion_ids"] for s in r["node_summaries"].values())
    titles = {s["id"]: s["title"] for s in r["policy"]["criteria"]}
    assert titles["VCDR_GE_07"] == "垂直杯盘比"
    assert titles["CUP_PIXELS"] == "杯面积"
    assert titles["CUP_AREA_CONTEXT"] == "面积杯盘比"
    rim = next(x for x in r["results"] if x["criterion_id"] == "RIM_NOTCH")
    assert rim["comparison"] == "visually_reviewed" and rim["evidence_refs"]
    assert "未见明确" in rim["assessment"] and rim["medical_effect"] == "not_assigned"
    assert r["visual_review"]["method"] == "direct_image_inspection_not_deployed_vlm"
    assert r["clinical_result"] == "not_computed"
    assert "本例结果：" in render_html(out)
    for node in r["node_summaries"].values():
        assert len(node["criterion_ids"]) == len(set(node["criterion_ids"]))


def test_numeric_descriptions_missing_measurements_and_aliases():
    v = fixture()
    r = build(v)
    rows = {r["criterion_id"]: r for r in r["results"]}
    assert "30.00%" in rows["VESSEL_DENSITY"]["assessment"]
    assert rows["CLASSIFICATION"]["value"] == pytest.approx(0.6)
    assert "60.0000" in rows["CLASSIFICATION"]["assessment"]
    assert rows["IOP"]["value"] is None
    assert rows["IOP"]["comparison"] == "unavailable"
    assert rows["FAMILY"]["comparison"] == "unavailable"
    assert r["node_summaries"]["TASK_WIDTH"]["criterion_ids"] == ["VESSEL_WIDTH"]
    v["case_evidence"]["results"]["density"]["observation"]["measurement"]["value"] = None
    next(n for n in v["nodes"] if n["key"] == "DENSITY")["patient_evidence"]["results"][0][
        "value"
    ] = None
    rows = {r["criterion_id"]: r for r in build(v)["results"]}
    assert rows["VESSEL_DENSITY"]["value"] is None
    assert rows["VESSEL_DENSITY"]["comparison"] == "unavailable"


@pytest.mark.parametrize(
    "kind", ["input", "mask", "model_scope", "demo_scope", "measurement", "nan", "forged_review"]
)
def test_tampered_bindings_and_review_rejected(kind):
    v = fixture()
    if kind == "input":
        v["case_evidence"]["input"]["checksum"] = "wrong"
    elif kind == "mask":
        v["case_evidence"]["artifacts"][0]["checksum"] = "wrong"
    elif kind == "model_scope":
        v["case_evidence"]["fairclip"]["outputs"][0]["source"]["scope"]["case_code"] = "CASE:OTHER"
    elif kind == "demo_scope":
        v["case_evidence"]["supplement"]["demographics"]["image_checksum"] = "wrong"
    elif kind == "measurement":
        next(n for n in v["nodes"] if n["key"] == "DENSITY")["patient_evidence"]["results"][0][
            "value"
        ] = 0.9
    elif kind == "nan":
        v["case_evidence"]["fairclip"]["outputs"][0]["output"]["normalized_model_scores"][0] = (
            float("nan")
        )
    else:
        v = apply(v)
        v["research_criteria"]["visual_review"]["observations"]["RIM_NOTCH"]["assessment"] = (
            "perfectly normal"
        )
        v["checksum"] = checksum({k: x for k, x in v.items() if k != "checksum"})
        with pytest.raises(ValueError, match="criterion_result_mismatch"):
            validate_snapshot(v)
        return
    with pytest.raises(ValueError):
        build(v)
