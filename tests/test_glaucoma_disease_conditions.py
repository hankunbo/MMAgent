import base64
import copy

import pytest

from apps import glaucoma_model_selection as selection
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_criteria_v1_3 import apply, build, policy
from apps.glaucoma_evidence_plan import checksum as document_checksum
from apps.glaucoma_rim_radial import radial_profile
from packages.dataset_adapters.reading import grayscale_image
from packages.object_store import checksum
from tests.test_glaucoma_inspection import fixture as previous_fixture


def mask_bodies():
    shape = (24, 24)
    disc = bytes(255 if 7 <= x < 17 and 7 <= y < 17 else 0 for y in range(24) for x in range(24))
    cup = bytes(255 if 9 <= x < 14 and 9 <= y < 14 else 0 for y in range(24) for x in range(24))
    return {"disc": grayscale_image(shape, disc), "cup": grayscale_image(shape, cup)}


def fixture(monkeypatch):
    value = previous_fixture()
    value.pop("model_selection")
    c = value["case_evidence"]
    c["input"]["dimensions"] = [24, 24]
    hashes = dict(selection.MASKS)
    for name, body in mask_bodies().items():
        kind = "ARTIFACT:SLO_SEGMAN_" + name.upper()
        hashes[kind] = checksum(body)
        for a in c["artifacts"]:
            if a["artifact_type_code"] == kind:
                a["checksum"] = checksum(body)
        c["images"]["segman_" + name] = "data:image/png;base64," + base64.b64encode(body).decode()
    monkeypatch.setattr(selection, "MASKS", hashes)
    return selection.apply(value)


def test_only_sourced_explicit_conditions_in_section5(monkeypatch):
    v = fixture(monkeypatch)
    before = copy.deepcopy(v)
    out = apply(v)
    assert v == before
    validate_snapshot(out)
    assert apply(out) == out
    r = out["research_criteria"]
    assert len(r["results"]) == 10
    assert len(r["descriptive_results"]) == 16
    for rule in r["policy"]["criteria"]:
        assert rule["sources"] and rule["condition"]
        if rule["kind"] == "numeric":
            assert "threshold" in rule and "operator" in rule
        else:
            assert rule["kind"] == "phenomenon"
    assert not {"CUP_PIXELS", "CUP_AREA_CONTEXT", "VESSEL_DENSITY", "CLASSIFICATION"} & {
        rule["id"] for rule in r["policy"]["criteria"]
    }
    rows = {x["criterion_id"]: x for x in r["results"]}
    assert "不能说明本例有青光眼" in rows["VCDR_GE_07"]["assessment"]
    assert rows["RIM_RADIAL_LT_01"]["value"] >= 0.1
    assert rows["RIM_NOTCH"]["comparison"] == "not_observed"
    assert rows["IOP"]["comparison"] == "unavailable"
    assert rows["AGE"]["relation_role"] == "risk_context"
    assert rows["IOP"]["relation_role"] == "referral_trigger"
    assert r["node_summaries"]["DENSITY"]["criterion_ids"] == []
    assert r["descriptive_node_summaries"]["DENSITY"]["criterion_ids"] == ["VESSEL_DENSITY"]
    area = next(x for x in r["descriptive_results"] if x["criterion_id"] == "CUP_AREA_CONTEXT")
    assert area["comparison"] == "observed" and "distance_to_reference" not in area
    assert all(x["medical_effect"] == "not_assigned" for x in r["results"])
    assert all(
        n["patient_evidence"]["status"] not in {"已有观察", "已有观察·判读不确定"}
        for n in out["nodes"]
    )


def test_threshold_definitions_keep_endpoints_and_dont_square_cdr():
    rules = {r["id"]: r for r in policy()["criteria"]}
    assert rules["VCDR_GE_07"]["threshold"] == 0.7
    assert rules["RIM_RADIAL_LT_01"]["threshold"] == 0.1
    assert rules["IOP"]["required_method"] == "Goldmann-type applanation"
    assert all(r.get("threshold") != 0.49 for r in rules.values())


@pytest.mark.parametrize("corruption", ["mask_bytes", "shape", "forged_ratio", "forged_sources"])
def test_rim_measurement_and_source_rules_revalidated(monkeypatch, corruption):
    v = fixture(monkeypatch)
    if corruption == "mask_bytes":
        v["case_evidence"]["images"]["segman_disc"] = v["case_evidence"]["images"]["segman_cup"]
        with pytest.raises(ValueError, match="rim_embedded_mask_binding"):
            build(v)
    elif corruption == "shape":
        v["case_evidence"]["input"]["dimensions"] = [30, 30]
        with pytest.raises(ValueError, match="rim_input_shape_mismatch"):
            build(v)
    else:
        v = apply(v)
        if corruption == "forged_ratio":
            v["research_criteria"]["radial_rim_measurement"]["minimum"]["ratio"] = 0.01
        else:
            v["research_criteria"]["policy"]["criteria"][0]["sources"] = []
        v["checksum"] = document_checksum({k: x for k, x in v.items() if k != "checksum"})
        with pytest.raises(ValueError, match="criterion_result_mismatch"):
            validate_snapshot(v)


def test_radial_computation_is_ratio_of_same_axis_not_bbox():
    n = 41
    disc = bytes(
        255 if (x - 20) ** 2 + (y - 20) ** 2 <= 100 else 0 for y in range(n) for x in range(n)
    )
    cup = bytes(
        255 if (x - 20) ** 2 + (y - 20) ** 2 <= 25 else 0 for y in range(n) for x in range(n)
    )
    r = radial_profile((n, n), disc, cup)
    assert r["sampled_rays"] == 360 and r["rays_below_0_1"] == 0
    assert r["minimum"]["ratio"] == pytest.approx(0.25, abs=0.05)
    with pytest.raises(ValueError, match="nested"):
        radial_profile((n, n), cup, disc)
    with pytest.raises(ValueError, match="binary"):
        radial_profile((n, n), bytes([1]) * (n * n), cup)
