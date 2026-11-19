"""Synthetic extension tests; no patient files, models, or production review writes."""

import base64
import copy
import json

import pytest

from apps import glaucoma_case_completion as completion
from apps import glaucoma_model_selection as selection
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_criteria_v1_3 import apply as criteria
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_evidence_structure import apply as organize
from packages.dataset_adapters.reading import grayscale_image
from packages.object_store import checksum as byte_checksum
from tests.test_glaucoma_disease_conditions import fixture as old_fixture
from tests.test_glaucoma_structure_ui import browser


def fixture(monkeypatch):
    v = old_fixture(monkeypatch)
    body = grayscale_image((24, 24), bytes([80]) * 576)
    sha = byte_checksum(body)
    v = json.loads(json.dumps(v).replace(v["case_evidence"]["input"]["checksum"], sha))
    c = v["case_evidence"]
    c["images"]["input"] = "data:image/png;base64," + base64.b64encode(body).decode()
    for a in c["artifacts"]:
        a["derived_from"] = [c["input"]["asset_code"]]
    monkeypatch.setattr(selection, "INPUT", sha)
    monkeypatch.setattr(completion, "IMAGE", sha)
    monkeypatch.setattr(completion, "CASE", c["case_code"])
    return organize(criteria(v))


def test_review_is_complete_but_not_a_made_up_point_or_diagnosis(monkeypatch):
    v = fixture(monkeypatch)
    before = copy.deepcopy(v)
    n = completion.apply(v)
    validate_snapshot(n)
    assert v == before and completion.apply(n) == n
    r = n["case_completion"]
    assert r["fovea"]["review_completed"] is True
    assert r["fovea"]["selected_coordinate_xy"] is None
    assert r["fovea"]["isnt"] is None
    assert len(r["visual_observations"]) == 7
    assert all(not o["deployed_vlm_executed"] for o in r["visual_observations"])
    assert len(n["nodes"]) == 52 and len(n["research_criteria"]["results"]) == 10
    assert n["case_evidence"] == v["case_evidence"]
    assert n["research_criteria"] == v["research_criteria"]
    assert r["summary"]["clinical_diagnosis"] is None
    assert r["summary"]["calibrated_disease_probability"] is None
    assert r["geometry"]["rim"]["record"]["measurement"]["valid_rays"] == 360
    assert r["geometry"]["ellipse"]["record"]["measurement"]["minor_over_major"] == 1


@pytest.mark.parametrize(
    "change", ["point", "vlm", "geometry", "scope", "mask", "projection", "image"]
)
def test_tampering_not_accepted_by_rehash(monkeypatch, change):
    v = completion.apply(fixture(monkeypatch))
    if change == "point":
        v["case_completion"]["fovea"]["selected_coordinate_xy"] = [3, 4]
    elif change == "vlm":
        v["case_completion"]["visual_observations"][0]["deployed_vlm_executed"] = True
    elif change == "geometry":
        v["case_completion"]["geometry"]["ellipse"]["record"]["measurement"]["minor_over_major"] = (
            0.1
        )
    elif change == "scope":
        v["case_completion"]["case_code"] = "CASE:ANON_OTHER"
    elif change == "mask":
        v["case_evidence"]["images"]["segman_disc"] = v["case_evidence"]["images"]["segman_cup"]
    elif change == "image":
        v["case_evidence"]["images"]["input"] = v["case_evidence"]["images"]["segman_cup"]
    else:
        next(n for n in v["nodes"] if n["key"] == "FOVEA_LOCATION")["patient_evidence"][
            "summary"
        ] = "已准确定位"
    v["checksum"] = checksum({k: x for k, x in v.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate_snapshot(v)


def test_mask_must_derive_from_this_image(monkeypatch):
    v = fixture(monkeypatch)
    for a in v["case_evidence"]["artifacts"]:
        a["derived_from"] = ["ASSET:OID_OTHER"]
    with pytest.raises(ValueError, match="completion_mask_scope"):
        completion.apply(v)


def test_new_html_has_all_nodes_measurements_and_no_accepted_border_point(monkeypatch):
    n = completion.apply(fixture(monkeypatch))
    browser(
        n,
        r"""
check(text(elements.detail).includes('逐节点完整清单（52项）'),'missing full list');
check(text(elements.detail).includes('本次未能定位'),'missing visual outcome');
run("show('CUP_RIM_MEASURE')");
check(text(elements.detail).includes('盘沿方向宽度与分布'),'missing rim');
check(text(elements.detail).includes('图像左方盘沿宽度中位数'),'missing sectors');
run("show('ALTERNATIVES')");
check(text(elements.detail).includes('视盘椭圆度与方向'),'missing ellipse');
check(text(elements.detail).includes('盘周形态与其他解释'),'missing visual review');
run("show('FOVEA_LOCATION')");
check(text(elements.detail).includes('本图未能定位'),'false fovea success');
const tables=all(elements.detail).filter(x=>x.tag==='table').map(text).join(' ');
check(!tables.includes('中央凹(x,y)'),'old point rendered as current result');
check(text(elements.detail).includes('历史模型候选（本次不采用）'),'old point lost');
run('for(const n of D.nodes)show(n.key)');
""",
    )
