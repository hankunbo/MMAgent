import copy

import pytest

from apps.evidence_graph_html import render_html, validate_snapshot
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_research_criteria import apply, build
from tests.test_glaucoma_case_view import data, project


def example(x=0.542):
    source, report = data()
    value = project(source, report)
    case = value["case_evidence"]
    image = case["input"]
    image.update(checksum="sha256:" + "a" * 64, scope={"case_code": case["case_code"]})
    case["artifacts"] = [
        {
            "asset_code": "ASSET:OID_MASK_METRICS",
            "checksum": "sha256:" + "b" * 64,
            "scope": {"case_code": case["case_code"]},
        }
    ]
    tool = {"code": "synthetic_segmentation", "version": "1.0.0"}
    case["supplement"] = {
        "case_code": case["case_code"],
        "input_checksum": image["checksum"],
        "artifact_codes": ["ASSET:OID_MASK_METRICS"],
        "registered_tool": tool,
        "mask_check": {"metrics": {"vcdr": {"value": x}}},
    }
    p = next(n for n in value["nodes"] if n["key"] == "CUP_RIM_MEASURE")["patient_evidence"]
    p.update(
        results=[{"name": "vcdr", "value": x, "unit": "1"}],
        tool=tool,
        artifact_codes=["ASSET:OID_MASK_METRICS"],
    )
    return value


def leaf(value):
    return next(n for n in value["nodes"] if n["key"] == "CUP_RIM_MEASURE")["patient_evidence"]


@pytest.mark.parametrize(
    "x,expected", [(0.542, "not_met"), (0.699999, "not_met"), (0.7, "met"), (0.9, "met")]
)
def test_exact_cutoff_and_no_disease_inference(x, expected):
    result = build(example(x))
    r = result["results"][0]
    assert r["comparison"] == expected
    assert r["applicability"] == "exploratory_only"
    assert r["medical_effect"] == "not_assigned"
    assert r["path"] == ["ROOT", "STRUCTURE", "CUPDISC", "CUP_RIM_MEASURE"]
    assert result["clinical_result"] == "not_computed"


def test_deduplicated_upward_references_and_unavailable_not_zero():
    s = example()
    before = copy.deepcopy(s)
    output = apply(s)
    assert s == before
    r = output["research_criteria"]
    assert len(r["results"]) == 8
    for key in ("ROOT", "STRUCTURE"):
        assert len(r["node_summaries"][key]["criterion_ids"]) == 8
    for key in ("CUPDISC", "CUP_RIM_MEASURE"):
        assert r["node_summaries"][key]["criterion_ids"].count("VCDR_GE_07") == 1
    for item in r["results"][1:]:
        assert item["value"] is None
        assert item["comparison"] in {"unavailable", "reference_only"}
    validate_snapshot(output)
    assert "判读条件与逐级回传" in render_html(output)


@pytest.mark.parametrize("x", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8"])
def test_invalid_numeric_rejected(x):
    with pytest.raises(ValueError, match="criterion_metric_invalid"):
        build(example(x))


@pytest.mark.parametrize(
    "kind", ["unit", "duplicate", "scope", "checksum", "tool", "binding", "missing_reason"]
)
def test_binding_and_definition_rejection(kind):
    value = example()
    p = leaf(value)
    if kind == "unit":
        p["results"][0]["unit"] = "pixel"
    elif kind == "duplicate":
        p["results"] *= 2
    elif kind == "scope":
        value["case_evidence"]["artifacts"][0]["scope"]["case_code"] = "CASE:OTHER"
    elif kind == "checksum":
        value["case_evidence"]["supplement"]["input_checksum"] = "wrong"
    elif kind == "tool":
        p["tool"] = {"code": "other"}
    elif kind == "binding":
        p["results"][0]["value"] = 0.8
    else:
        p["results"][0]["missing_reason"] = "unavailable"
    with pytest.raises(ValueError, match="criterion_"):
        build(value)


def test_area_ratio_cannot_substitute_and_forged_rehash_does_not_pass():
    value = example()
    leaf(value)["results"][0]["name"] = "acdr"
    assert build(value)["results"][0]["comparison"] == "unavailable"
    value = apply(example())
    value["research_criteria"]["results"][0]["comparison"] = "met"
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    with pytest.raises(ValueError, match="criterion_result_mismatch"):
        validate_snapshot(value)


def test_archived_source_binding_and_old_snapshot_not_upgraded():
    value = example()
    value["egs_pages"] = [{"id": "EGS-ONH-2019", "source_ref": {"checksum": "synthetic"}}]
    output = apply(value)
    sources = output["research_criteria"]["source_bindings"]
    assert sources["EGS-ONH-2019"]["knowledge_source_ref"] == {"checksum": "synthetic"}
    assert sources["PHENE-2019-V2"]["status"] == "external_reading_not_published_claim"
    assert "research_criteria" not in value
    assert apply(value) == output
