"""Synthetic grading tests: these values do not validate medical diagnosis."""

import copy
from functools import partial

import pytest

from apps.evidence_graph_details import attach as attach_details
from apps.evidence_graph_scores import attach, unscored
from packages.evidence_details.models import digest
from packages.evidence_scoring import continuous_policy
from packages.evidence_scoring.api import validate as dispatch_validate
from packages.evidence_scoring.continuous import build, channel, summarize, validate
from packages.evidence_scoring.continuous_models import ContinuousReview, ContinuousScores, Judgment
from packages.evidence_scoring.engine import build as old_build
from tests.test_evidence_details import example
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def graded():
    with pytest.MonkeyPatch.context() as patch:
        base, profiles, ledger = example(patch)
        view = attach_details(base, profiles, ledger)
        # Controlled synthetic review owner, not caller-provided deployment data.
        bundle = copy.deepcopy(continuous_policy.REVIEW_BUNDLE)
        bundle.update({key: ledger[key] for key in ("case_code", "input_checksum")})
        bundle["trace_checksum"] = ledger["checksum"]
        patch.setattr(continuous_policy, "REVIEW_BUNDLE", bundle)
        yield view, build(view)


def make_row(raw, *, effect, identifier="synthetic-positive", family="family", block="image"):
    row = copy.deepcopy(raw)
    row.update(id=identifier, correlation_group=family, dependence_block=block, role="diagnostic")
    for field in ("observation", "relation", "applicability"):
        row[field] = {"value": 0.8, "reason": "synthetic fixture", "missing_reason": None}
    row["weight"] = 0.8 * 0.8 * 0.8
    row["judgment"] = {
        "value": effect,
        "lower": effect,
        "upper": effect,
        "reason": "synthetic",
        "missing_reason": None,
        "method": "manual_source_bound_review",
    }
    row["direction"] = "support" if effect > 0 else "refute" if effect < 0 else "neutral"
    row["support"] = row["weight"] * max(effect, 0.0)
    row["refutation"] = row["weight"] * max(-effect, 0.0)
    return ContinuousReview.model_validate(row)


def test_complete_bipolar_score_preserves_original_measurements_and_v1(graded):
    base, scores = graded
    assert len(scores["reviews"]) == 43 and len(scores["nodes"]) == 63
    assert len(scores["removals"]) == 50
    root = scores["nodes"]["ROOT"]
    assert root["diagnostic"]["refutation"] > 0
    assert root["risk"]["support"] > 0 and root["model"]["support"] > 0
    assert root["combined"]["conflict"] > 0
    c = root["confidence"]
    assert c["value"] == 0.5 + 0.5 * root["combined"]["net"]
    assert c["lower"] <= c["value"] <= c["upper"]
    assert scores["disease_probability"] is None and not scores["clinical_acceptance"]
    assert root["judgment_coverage"]["numerator"] == 5
    combined = attach(base, scores)
    assert unscored(combined) == base
    old = old_build(base)
    assert dispatch_validate(base, old) == old
    assert old["schema_version"].endswith("1.0.0")
    assert old["nodes"]["ROOT"]["diagnostic"]["refutation"] == 0


def test_continuous_not_binary_and_ratings_not_votes(graded):
    _, scores = graded
    rows = {r["id"]: r for r in scores["reviews"]}
    cup = rows["criterion:VCDR_GE_07"]
    assert -1 < cup["judgment"]["value"] < 0
    assert cup["refutation"] == cup["weight"] * -cup["judgment"]["value"]
    assert cup["judgment"]["lower"] < 0 < cup["judgment"]["upper"]
    assert rows["criterion:RIM_NOTCH"]["judgment"]["value"] == 0
    assert rows["criterion:RNFL_DEFECT"]["judgment"]["value"] is None
    assert rows["association:VENULAR_BRANCH_ANGLE"]["judgment"]["value"] is None
    model = rows["descriptive:CLASSIFICATION"]
    assert 0 < model["support"] < model["weight"]
    assert model["judgment"]["upper"] <= 1.0


def test_duplicates_correlated_complementary_and_independent_information(graded):
    raw = next(r for r in graded[1]["reviews"] if r["id"] == "criterion:AGE")
    make = partial(make_row, raw, effect=0.5)
    one = make()
    assert channel([one] * 20) == channel([one])
    duplicate_fact = make(identifier="same-fact-second-reference")
    assert channel([one, duplicate_fact]) == channel([one])
    complement = make(identifier="complement", family="another-measurement")
    independent = make(identifier="independent", family="another-measurement", block="other")
    assert channel([one])["support"] < channel([one, complement])["support"]
    assert channel([one, complement])["support"] < channel([one, independent])["support"]
    assert channel([one, complement]) == channel([complement, one])
    conflict = make_row(raw, effect=-0.5, identifier="negative", family="opposing")
    c = summarize([one, conflict])
    assert c["confidence"]["value"] == 0.5 and c["combined"]["conflict"] > 0
    neutral = summarize([make_row(raw, effect=0.0)])
    assert neutral["confidence"]["value"] == 0.5 and neutral["combined"]["conflict"] == 0
    empty = summarize([])
    assert empty["confidence"]["value"] is None
    assert empty["judgment_coverage"]["status"] == "not_applicable"


def test_monotonic_strength_quality_and_bounds(graded):
    raw = next(r for r in graded[1]["reviews"] if r["id"] == "criterion:AGE")
    positive = make_row(raw, effect=0.2)
    stronger = make_row(raw, effect=0.8)
    assert (
        summarize([positive])["confidence"]["value"] < summarize([stronger])["confidence"]["value"]
    )
    negative = make_row(raw, effect=-0.8)
    assert summarize([negative])["confidence"]["value"] < 0.5
    for r in [positive, stronger, negative]:
        c = summarize([r])["confidence"]
        assert c["lower"] <= c["value"] <= c["upper"]
    forged = positive.model_copy(update={"support": 0.9})
    with pytest.raises(ValueError, match="contribution"):
        channel([forged])
    with pytest.raises(ValueError, match="conflicting_duplicate"):
        channel([positive, stronger])


def test_removal_and_no_double_count_at_parent_and_fixed_denominator(graded):
    _, scores = graded
    for node in ("CUP_RIM_MEASURE", "CUPDISC", "STRUCTURE", "ROOT"):
        assert scores["nodes"][node]["assessment_ids"].count("criterion:VCDR_GE_07") == 1
    assert scores["nodes"]["DISC_MASK"]["confidence"]["value"] is None
    removals = {r["id"]: r for r in scores["removals"]}
    no_age = removals["review:criterion:AGE"]
    assert no_age["confidence_delta"] < 0
    assert no_age["after"]["confidence"]["value"] < scores["nodes"]["ROOT"]["confidence"]["value"]
    no_disc = removals["resource:segman_disc"]
    assert no_disc["confidence_delta"] > 0
    assert no_disc["risk_support_delta"] == 0
    assert (
        no_disc["after"]["judgment_coverage"]["denominator"]
        == scores["nodes"]["ROOT"]["judgment_coverage"]["denominator"]
    )
    no_image = removals["resource:image"]
    assert no_image["after"]["confidence"]["basis"] == ["risk"]


@pytest.mark.parametrize("mutation", ["nan", "bounds", "missing", "method", "positive_missing"])
def test_judgment_invalid_construction(mutation):
    raw = {
        "value": 0.2,
        "lower": 0.1,
        "upper": 0.3,
        "reason": "test",
        "missing_reason": None,
        "method": "manual_source_bound_review",
    }
    raw.update(
        {
            "nan": {"value": float("nan")},
            "bounds": {"lower": 0.9},
            "missing": {"value": None},
            "method": {"method": "not_assessed"},
            "positive_missing": {"missing_reason": "missing"},
        }[mutation]
    )
    with pytest.raises(ValueError):
        Judgment.model_validate(raw)


@pytest.mark.parametrize(
    "mutation", ["judgment", "source", "case", "aggregate", "removal", "policy", "fake_probability"]
)
def test_rehash_does_not_accept_forged_review_or_total(graded, mutation):
    base, scores = graded
    raw = copy.deepcopy(scores)
    if mutation == "judgment":
        raw["reviews"][0]["judgment"]["value"] = 0.9
    elif mutation == "source":
        raw["reviews"][0]["source_binding_checksum"] = "forged"
    elif mutation == "case":
        raw["case_code"] = "CASE:FORGED"
    elif mutation == "aggregate":
        raw["nodes"]["ROOT"]["combined"]["support"] = 0.9
    elif mutation == "removal":
        raw["removals"][0]["confidence_delta"] = 0.9
    elif mutation == "policy":
        raw["review_bundle_checksum"] = "forged"
    else:
        raw["disease_probability"] = 0.8
    raw["checksum"] = digest({k: v for k, v in raw.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, raw)


def test_unknown_case_not_given_manual_case_scores_and_changed_inputs_rejected(graded, monkeypatch):
    base, _ = graded
    bundle = copy.deepcopy(continuous_policy.REVIEW_BUNDLE)
    bundle["case_code"] = "CASE:DIFFERENT_REVIEW"
    monkeypatch.setattr(continuous_policy, "REVIEW_BUNDLE", bundle)
    scores = build(base)
    assert scores["review_status"] == "not_reviewed"
    assert scores["nodes"]["ROOT"]["confidence"]["value"] is None
    bundle["case_code"] = base["case_evidence"]["case_code"]
    bundle["input_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="scope_or_source"):
        build(base)


def test_snapshot_model_and_browser_six_sections(graded):
    base, scores = graded
    assert ContinuousScores.model_validate(scores).model_dump(mode="json") == scores
    browser(
        attach(base, scores),
        r"""
check(text(elements.detail).includes('综合证据置信分'),'total missing');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const sections=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(sections.length===6,'sections');
 check(fifth().includes('本节点研究评分'),'node score');
}
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('本例连续判读')&&fifth().includes('实际反驳贡献'),'graded contribution');
run("show('ROOT')");
check(fifth().includes('移除后综合分')&&fifth().includes('连续判读覆盖'),'removal/coverage');
check(fifth().includes('参数敏感范围'),'sensitivity');
""",
    )
