"""Synthetic v3 mechanisms, not clinical validation of the working ratings."""

import copy

import pytest

from apps.evidence_graph_details import attach as attach_details
from apps.evidence_graph_scores import attach, unscored
from apps.glaucoma_evidence_plan import checksum
from packages.evidence_details.models import digest
from packages.evidence_details.projection import build as details
from packages.evidence_scoring import continuous_policy
from packages.evidence_scoring import participation_policy as policy
from packages.evidence_scoring.api import validate as dispatch_validate
from packages.evidence_scoring.continuous import build as v2
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.evidence_scoring.participation import adequacy, build, groups_for, validate
from packages.evidence_scoring.participation_models import Adequacy, ParticipationScores
from tests.test_evidence_details import example
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def graded():
    with pytest.MonkeyPatch.context() as patch:
        base, profiles, _ = example(patch)
        # Explicit software fixture overlay, never a production visual observation.
        original = next(
            a
            for a in base["case_evidence"]["artifacts"]
            if a["artifact_type_code"] == "ARTIFACT:SLO_SEGMAN_DISC"
        )
        overlay = copy.deepcopy(original)
        overlay.update(
            asset_code="ASSET:OID_SYNTHETIC_OVERLAY",
            artifact_type_code="ARTIFACT:SLO_SEGMAN_OVERLAY",
            checksum=digest({"fixture": "distinct synthetic overlay bytes"}),
        )
        base["case_evidence"]["artifacts"].append(overlay)
        base["checksum"] = checksum({k: v for k, v in base.items() if k != "checksum"})
        ledger = details(base, profiles)
        view = attach_details(base, profiles, ledger)
        scope = {k: ledger[k] for k in ("case_code", "input_checksum")}
        scope["trace_checksum"] = ledger["checksum"]
        bundle = copy.deepcopy(continuous_policy.REVIEW_BUNDLE)
        bundle.update(scope)
        patch.setattr(continuous_policy, "REVIEW_BUNDLE", bundle)
        patch.setattr(policy, "REVIEW_SCOPE", scope)
        visual = copy.deepcopy(policy.VISUAL_REVIEW)
        visual.update(input_checksum=scope["input_checksum"], overlay_checksum=overlay["checksum"])
        patch.setattr(policy, "VISUAL_REVIEW", visual)
        yield view, build(view)


def test_all_records_nodes_and_scopes_are_complete(graded):
    base, s = graded
    assert len(s["reviews"]) == len(s["evaluations"]) == 44
    assert len(s["nodes"]) == len(s["node_participation"]) == 63
    assert len(s["groups"]) == 15 and len(s["removals"]) == 51
    assert ParticipationScores.model_validate(s).model_dump(mode="json") == s
    root = s["nodes"]["ROOT"]
    for scope, a in root["adequacy"].items():
        assert 0 < a["value"] < 1
        assert a["adjusted"] == 0.5 + a["value"] * (root["confidence"]["value"] - 0.5)
        assert a["adjusted_lower"] <= a["adjusted"] <= a["adjusted_upper"]
        assert a["scope"] == scope
    assert (
        root["adequacy"]["comprehensive_research"]["value"]
        < root["adequacy"]["slo_image_research"]["value"]
    )
    assert s["disease_probability"] is None and not s["clinical_acceptance"]
    assert unscored(attach(base, s)) == base


def test_coarse_negative_and_unassessable_fine_notch_are_distinct(graded):
    _, s = graded
    rs = {r["id"]: r for r in s["reviews"]}
    assert rs["criterion:RIM_NOTCH"]["judgment"]["value"] is None
    assert rs["visual:rim_continuity"]["judgment"]["value"] < 0
    assert rs["visual:rim_continuity"]["refutation"] > 0
    assert (
        rs["visual:rim_continuity"]["correlation_group"]
        == rs["criterion:VCDR_GE_07"]["correlation_group"]
    )
    # A weak additional same-family observation can have zero marginal impact.
    e = next(e for e in s["evaluations"] if e["assessment_id"] == "visual:rim_continuity")
    assert e["effective_refutation"] == 0 and e["marginal_direction_delta"] == 0
    assert e["provenance"] == "assistant_direct_image_inspection"
    assert "细微" in e["unresolved"]


def test_missing_not_negative_and_numeric_mismatches_not_acquired(graded):
    _, s = graded
    rows = {r["id"]: r for r in s["reviews"]}
    groups = {g["id"]: g for g in s["groups"]}
    for key in ("criterion:FIELD", "criterion:IOP", "descriptive:OCT_REFERENCE"):
        assert rows[key]["judgment"]["value"] is None
        assert rows[key]["support"] == rows[key]["refutation"] == 0
    assert groups["field"]["coverage"]["value"] == 0
    assert groups["rnfl_thickness"]["importance"]["slo_image_research"] == 0
    assert groups["vascular"]["coverage"]["numerator"] == 1
    assert not next(
        g
        for g in groups["vascular"]["slots"]
        if "association:VASCULAR_SKELETON_DENSITY" in g["assessment_ids"]
    )["acquired"]
    for m in s["nodes"]["ROOT"]["missing_impacts"]:
        assert m["missing_contribution"] > 0
        assert m["disease_direction_impact"] == "not_quantified"
        assert m["scenario_kind"] == "coverage_only_holds_direction_fixed"


def test_scopes_fixed_denominators_and_removal_recalculate_coverage(graded):
    _, s = graded
    root = s["nodes"]["ROOT"]
    for r in s["removals"]:
        for scope, a in r["after"]["adequacy"].items():
            assert a["total_importance"] == root["adequacy"][scope]["total_importance"]
            assert a["value"] <= root["adequacy"][scope]["value"]
    image = next(r for r in s["removals"] if r["id"] == "resource:image")
    assert image["after"]["confidence"]["basis"] == ["risk"]
    assert image["adequacy_delta"]["comprehensive_research"] < 0
    for profile in root["adequacy"].values():
        impacts = [m for m in root["missing_impacts"] if m["scope"] == profile["scope"]]
        assert sum(m["missing_contribution"] for m in impacts) == pytest.approx(profile["missing"])


def test_quality_technical_and_organization_are_not_disease_votes(graded):
    _, s = graded
    disc = s["node_participation"]["DISC_MASK"]
    assert disc["role"] == "evidence_quality" and disc["technical_coverage"]["value"] == 1
    assert s["nodes"]["DISC_MASK"]["confidence"]["value"] is None
    assert s["node_participation"]["STRUCTURE"]["role"] == "organization"
    q = next(e for e in s["evaluations"] if e["assessment_id"] == "descriptive:IMAGE_QUALITY")
    assert q["quality_review"]["value"] == 0.4 and q["effective_support"] == 0
    assert s["nodes"]["CUP_RIM_MEASURE"]["combined"] == s["nodes"]["CUPDISC"]["combined"]


def test_group_order_and_duplicate_facts_do_not_change_adequacy(graded):
    _, s = graded
    rows = [ContinuousReview.model_validate(r) for r in s["reviews"]]
    g = groups_for(rows, policy.GROUPS)
    assert g == groups_for(list(reversed(rows)) + rows, list(reversed(policy.GROUPS)))
    c = s["nodes"]["ROOT"]["confidence"]
    assert adequacy(g, c, "comprehensive_research") == adequacy(
        list(reversed(g)), c, "comprehensive_research"
    )
    with pytest.raises(ValueError, match="duplicate_group"):
        adequacy(g + [g[0]], c, "comprehensive_research")


@pytest.mark.parametrize("direction", [0.2, 0.5, 0.8])
def test_missing_shrinks_both_directions_without_turning_into_disease_evidence(graded, direction):
    groups = copy.deepcopy(graded[1]["groups"])
    confidence = {"value": direction, "lower": direction, "upper": direction}
    full = adequacy(groups, confidence, "comprehensive_research")
    for g in groups:
        g["coverage"]["value"] = 0.0
    empty = adequacy(groups, confidence, "comprehensive_research")
    assert empty["adjusted"] == 0.5
    assert abs(empty["adjusted"] - 0.5) <= abs(full["adjusted"] - 0.5)
    missing_c = adequacy(
        groups, {"value": None, "lower": None, "upper": None}, "comprehensive_research"
    )
    assert missing_c["adjusted"] is None


@pytest.mark.parametrize(
    "mutation",
    ["score", "quality", "coverage", "scenario", "image", "case", "policy", "target", "role"],
)
def test_forged_rehashed_envelope_rejected(graded, mutation):
    base, old = graded
    s = copy.deepcopy(old)
    if mutation == "score":
        s["reviews"][0]["judgment"]["value"] = 0.95
    elif mutation == "quality":
        next(e for e in s["evaluations"] if e["quality_review"])["quality_review"]["value"] = 0.99
    elif mutation == "coverage":
        s["nodes"]["ROOT"]["adequacy"]["comprehensive_research"]["value"] = 1.0
    elif mutation == "scenario":
        s["nodes"]["ROOT"]["missing_impacts"][0]["adjusted_delta"] = 0.5
    elif mutation == "image":
        s["visual_review"]["overlay_checksum"] = "sha256:" + "f" * 64
    elif mutation == "case":
        s["case_code"] = "CASE:ANON_OTHER"
    elif mutation == "policy":
        s["groups"][0]["importance"]["comprehensive_research"] = 8.0
    elif mutation == "target":
        s["evaluations"][0]["target"] = "evidence_quality"
    else:
        s["node_participation"]["DISC_MASK"]["role"] = "disease_evidence"
    s["checksum"] = digest({k: v for k, v in s.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, s)


def test_legacy_readback_scope_drift_and_unknown_case(graded, monkeypatch):
    base, _ = graded
    old = v2(base)
    assert dispatch_validate(base, old) == old
    assert (
        next(r for r in old["reviews"] if r["id"] == "criterion:RIM_NOTCH")["judgment"]["value"]
        == 0
    )
    scope = copy.deepcopy(policy.REVIEW_SCOPE)
    scope["input_checksum"] = "sha256:" + "0" * 64
    monkeypatch.setattr(policy, "REVIEW_SCOPE", scope)
    with pytest.raises(ValueError, match="scope_or_source"):
        build(base)
    scope["case_code"] = "CASE:OTHER"
    bundle = copy.deepcopy(continuous_policy.REVIEW_BUNDLE)
    bundle["case_code"] = "CASE:OTHER"
    monkeypatch.setattr(continuous_policy, "REVIEW_BUNDLE", bundle)
    s = build(base)
    assert s["visual_review"] is None and not s["groups"]
    assert s["nodes"]["ROOT"]["adequacy"]["comprehensive_research"]["status"] == "not_applicable"
    assert all(e["quality_review"] is None for e in s["evaluations"])


def test_constructed_ratio_and_browser_six_sections(graded):
    base, s = graded
    a = Adequacy.model_validate(s["nodes"]["ROOT"]["adequacy"]["comprehensive_research"])
    bad = a.model_copy(update={"value": float("nan")})
    with pytest.raises(ValueError):
        Adequacy.model_validate(bad.model_dump())
    browser(
        attach(base, s),
        r"""
check(text(elements.detail).includes('充分性调整分'),'adjusted absent');
check(text(elements.detail).includes('SLO影像研究'),'image scope absent');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const sections=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(sections.length===6,'six sections');
 check(fifth().includes('本节点计算角色'),'node role');
}
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('粗大盘沿连续性')&&fifth().includes('细微'),'split absence missing');
check(fifth().includes('边际影响'),'marginal impact missing');
run("show('FIELD_VALUES')");
check(fifth().includes('保持疾病方向不变')&&fifth().includes('不能量化'),'missing scenario truth');
run("show('DISC_MASK')");
check(fifth().includes('技术资源到位'),'technical coverage');
""",
    )
