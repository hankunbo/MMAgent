"""V4 synthetic mechanism tests; not validation of the manual case ratings."""

import copy

import pytest

from apps.evidence_graph_scores import attach, unscored
from packages.evidence_details.models import digest
from packages.evidence_scoring import continuous_policy, participation_policy
from packages.evidence_scoring.api import validate as dispatch_validate
from packages.evidence_scoring.interpretation import build, validate
from packages.evidence_scoring.interpretation_models import InterpretationScores
from tests.test_evidence_scoring_participation import graded  # noqa: F401
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def interpreted(request):
    base, previous = request.getfixturevalue("graded")
    return base, previous, build(base)


def test_every_record_has_specific_decision_and_old_graph_preserved(interpreted):
    base, previous, s = interpreted
    assert len(s["reviews"]) == len(s["adjudications"]) == 46
    assert len(s["nodes"]) == 63 and len(s["removals"]) == 53
    assert all(a["review_completed"] for a in s["adjudications"])
    assert sum(a["state"] == "direction_reviewed" for a in s["adjudications"]) == 7
    assert len(s["readings"]) == 8 and len(s["image_reviews"]) == 2
    assert InterpretationScores.model_validate(s).model_dump(mode="json") == s
    assert unscored(attach(base, s)) == base
    assert dispatch_validate(base, previous) == previous
    assert s["disease_probability"] is None and not s["clinical_acceptance"]


def test_new_observations_are_limited_and_correlated_not_full_negative_tests(interpreted):
    _, previous, s = interpreted
    rs = {r["id"]: r for r in s["reviews"]}
    for key in ("visual:rnfl_gross_pattern", "visual:vessel_gross_course"):
        r = rs[key]
        assert r["judgment"]["lower"] < r["judgment"]["value"] < 0 < r["judgment"]["upper"]
        assert r["refutation"] > 0 and r["dependency_paths"] == [["image"]]
        assert not r["criterion_checked"]
    assert rs["criterion:RNFL_DEFECT"]["judgment"]["value"] is None
    assert rs["association:DISC_VESSEL_COURSE"]["judgment"]["value"] is None
    assert (
        rs["visual:vessel_gross_course"]["correlation_group"]
        == rs["criterion:VCDR_GE_07"]["correlation_group"]
    )
    e = next(e for e in s["evaluations"] if e["assessment_id"] == "visual:vessel_gross_course")
    assert e["marginal_direction_delta"] == 0
    assert s["groups"] == previous["groups"]  # partial views do not acquire the full original slots
    assert (
        s["nodes"]["ROOT"]["combined"]["refutation"]
        > previous["nodes"]["ROOT"]["combined"]["refutation"]
    )


def test_concrete_method_mismatches_and_missing_inputs_not_disease_votes(interpreted):
    _, _, s = interpreted
    aa = {a["assessment_id"]: a for a in s["adjudications"]}
    rr = {r["id"]: r for r in s["reviews"]}
    for key in (
        "ARTERIOLAR_CALIBER",
        "AV_TORTUOSITY",
        "VASCULAR_SKELETON_DENSITY",
        "VASCULAR_FRACTAL",
    ):
        a = aa["association:" + key]
        assert a["state"] == "method_mismatch" and a["source_ids"]
        assert rr[a["assessment_id"]]["judgment"]["value"] is None
    assert "90°" in aa["association:VENULAR_BRANCH_ANGLE"]["unresolved"]
    assert aa["association:DISC_SHAPE_CONTEXT"]["state"] == "context_only"
    assert "1.387" in aa["association:DISC_SHAPE_CONTEXT"]["comparison"]
    assert "beta" in aa["association:PERIPAPILLARY_APPEARANCE"]["unresolved"]
    for a in aa.values():
        if a["state"] != "direction_reviewed":
            r = rr[a["assessment_id"]]
            assert r["support"] == r["refutation"] == 0
    assert aa["criterion:IOP"]["state"] == "input_missing"


def test_source_reading_scope_and_removed_image_invalidate_visual_contribution(interpreted):
    _, _, s = interpreted
    assert (
        next(r for r in s["readings"] if r["id"] == "density_methods")["reading_scope"]
        == "full_text_methods_excerpt_not_full_results"
    )
    assert all(v["input_checksum"] == s["input_checksum"] for v in s["image_reviews"])
    removed = next(r for r in s["removals"] if r["id"] == "resource:image")
    assert all(v["assessment_id"] in removed["affected_assessments"] for v in s["image_reviews"])
    assert removed["after"]["confidence"]["basis"] == ["risk"]
    for r in s["removals"]:
        for scope, a in r["after"]["adequacy"].items():
            assert (
                a["total_importance"] == s["nodes"]["ROOT"]["adequacy"][scope]["total_importance"]
            )


@pytest.mark.parametrize(
    "mutation", ["image", "region", "source", "state", "omit", "comparison", "construct"]
)
def test_forged_or_rehashed_decision_rejected(interpreted, mutation):
    base, _, old = interpreted
    s = copy.deepcopy(old)
    if mutation == "image":
        s["image_reviews"][0]["input_checksum"] = "sha256:" + "a" * 64
    elif mutation == "region":
        s["image_reviews"][0]["region_xyxy"] = [0, 0, 999, 999]
    elif mutation == "source":
        s["readings"][0]["material_checksum"] = "sha256:" + "a" * 64
    elif mutation == "state":
        next(a for a in s["adjudications"] if a["state"] == "method_mismatch")["state"] = (
            "direction_reviewed"
        )
    elif mutation == "omit":
        s["adjudications"].pop()
    elif mutation == "comparison":
        s["adjudications"][0]["comparison"] = "caller invented normal range"
    else:
        s = InterpretationScores.model_construct(**{**s, "clinical_acceptance": True}).model_dump()
    s["checksum"] = digest({k: v for k, v in s.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, s)


def test_unknown_case_has_no_case_specific_interpretations(interpreted, monkeypatch):
    base, _, _ = interpreted
    for policy, name in [
        (participation_policy, "REVIEW_SCOPE"),
        (continuous_policy, "REVIEW_BUNDLE"),
    ]:
        value = copy.deepcopy(getattr(policy, name))
        value["case_code"] = "CASE:OTHER"
        monkeypatch.setattr(policy, name, value)
    s = build(base)
    assert not s["readings"] and not s["image_reviews"]
    assert all(
        a["state"] == "not_reviewed" and not a["review_completed"] for a in s["adjudications"]
    )
    assert all(r["judgment"]["value"] is None for r in s["reviews"])


def test_browser_specific_decisions_and_six_sections(interpreted):
    base, _, s = interpreted
    browser(
        attach(base, s),
        r"""
check(text(elements.detail).includes('逐项判读补全'),'overview absent');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const headings=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(headings.length===6,'six sections');
}
run("show('RNFL_REFLECTANCE')");
check(fifth().includes('粗大RNFL')&&fifth().includes('-0.0400'),'visual absent');
run("show('VASCULAR_SKELETON_DENSITY')");
check(fifth().includes('骨架')&&fifth().includes('方法需对齐'),'method detail');
""",
    )
