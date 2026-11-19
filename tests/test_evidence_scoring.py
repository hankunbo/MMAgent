"""Synthetic tests: numeric review is separate from source medical evidence."""

import copy

import pytest

from apps.evidence_graph_details import attach as attach_details
from apps.evidence_graph_scores import attach, unscored
from packages.evidence_details.models import digest
from packages.evidence_scoring.engine import build, channel, coverage, path_available, validate
from packages.evidence_scoring.models import Rating, Review, Scores
from tests.test_evidence_details import example
from tests.test_glaucoma_structure_ui import browser


@pytest.fixture(scope="module")
def scored():
    with pytest.MonkeyPatch.context() as patch:
        base, profiles, ledger = example(patch)
        view = attach_details(base, profiles, ledger)
        yield view, build(view)


def test_complete_inventory_old_bytes_and_deduplicated_hierarchy(scored):
    base, scores = scored
    assert len(scores["reviews"]) == 43 and len(scores["nodes"]) == 63
    assert len(scores["removals"]) == 50
    assert scores["disease_probability"] is None and not scores["clinical_acceptance"]
    for key, summary in scores["nodes"].items():
        assert len(summary["assessment_ids"]) == len(set(summary["assessment_ids"]))
        if key in {"CUP_RIM_MEASURE", "CUPDISC", "STRUCTURE", "ROOT"}:
            assert summary["assessment_ids"].count("criterion:VCDR_GE_07") == 1
    assert scores["nodes"]["DISC_MASK"]["assessment_ids"] == []
    assert scores["nodes"]["ROOT"]["criterion_coverage"]["denominator"] == 10
    value = attach(base, scores)
    assert unscored(value) == base
    assert value["case_evidence"]["fairclip"] == base["case_evidence"]["fairclip"]
    assert all(
        a["confidence_score"] is None for a in value["judgment_details"]["ledger"]["assessments"]
    )


def test_negative_observation_missing_and_risk_do_not_become_diagnosis(scored):
    _, scores = scored
    rows = {r["id"]: r for r in scores["reviews"]}
    for key in ["VCDR_GE_07", "RIM_RADIAL_LT_01", "RIM_NOTCH"]:
        r = rows["criterion:" + key]
        assert r["direction"] == "neutral" and r["refutation"] == 0
    assert rows["criterion:FIELD"]["observation"]["value"] is None
    assert rows["association:VENULAR_BRANCH_ANGLE"]["applicability"]["value"] is None
    assert rows["descriptive:CLASSIFICATION"]["support"] == 0
    assert rows["criterion:AGE"]["role"] == "risk"
    assert scores["nodes"]["ROOT"]["diagnostic"]["support"] == 0
    assert scores["nodes"]["ROOT"]["diagnostic"]["refutation"] == 0


def test_removal_respects_and_independent_age_and_frozen_denominator(scored):
    _, scores = scored
    scenarios = {s["id"]: s for s in scores["removals"]}
    disc = scenarios["resource:segman_disc"]
    assert {
        "criterion:VCDR_GE_07",
        "criterion:RIM_RADIAL_LT_01",
        "association:VENULAR_BRANCH_ANGLE",
    } <= set(disc["affected_assessments"])
    assert "criterion:AGE" not in disc["affected_assessments"]
    assert disc["risk_support_delta"] == 0
    assert disc["after"]["criterion_coverage"]["denominator"] == 10
    assert scenarios["resource:demographics"]["after"]["risk"]["support"] == 0
    assert (
        "association:VENULAR_BRANCH_ANGLE" in scenarios["resource:slo_vein"]["affected_assessments"]
    )
    assert "criterion:VCDR_GE_07" not in scenarios["resource:slo_vein"]["affected_assessments"]
    assert path_available([["a", "b"], ["c"]], {"b"})
    assert not path_available([["a", "b"], ["c"]], {"b", "c"})
    assert not path_available([], set()) and not path_available([[]], set())


def test_bipolar_group_dedup_permutation_and_conflict(scored):
    _, scores = scored
    raw = copy.deepcopy(next(r for r in scores["reviews"] if r["id"] == "criterion:AGE"))
    # Synthetic positive/negative rules, NOT assertions about this patient.
    for name in ["observation", "relation", "applicability"]:
        raw[name] = {"value": 0.8, "reason": "synthetic", "missing_reason": None}
    w = 0.8 * 0.8 * 0.8
    raw.update(
        role="diagnostic",
        direction="support",
        weight=w,
        support=w,
        observation_available=True,
        correlation_group="synthetic-group",
    )
    positive = Review.model_validate(raw)
    negative = Review.model_validate(
        {**raw, "id": "synthetic-negative", "direction": "refute", "support": 0.0, "refutation": w}
    )
    assert channel([positive] * 20, "diagnostic") == channel([positive], "diagnostic")
    both = channel([positive, negative], "diagnostic")
    assert both == channel([negative, positive], "diagnostic")
    assert both["conflict"] == pytest.approx(w) and both["net"] == 0
    assert coverage(0, 0)["value"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_illegal_numeric_ratings(value):
    with pytest.raises(ValueError):
        Rating(value=value, reason="synthetic")


def test_unknown_is_not_zero_and_constructed_models_are_rechecked():
    with pytest.raises(ValueError):
        Rating(value=None, reason="synthetic")
    with pytest.raises(ValueError):
        Rating(value=0.5, reason="synthetic", missing_reason="unknown")
    value = Rating(value=0.7, reason="synthetic").model_copy(update={"value": 2.0})
    with pytest.raises(ValueError):
        Rating.model_validate(value.model_dump())


@pytest.mark.parametrize(
    "mutation", ["case", "input", "policy", "rating", "dependency", "aggregate", "removal"]
)
def test_rehash_cannot_authorize_mutated_scores(scored, mutation):
    base, scores = scored
    raw = copy.deepcopy(scores)
    if mutation in {"case", "input", "policy"}:
        field = {"case": "case_code", "input": "input_checksum", "policy": "policy_checksum"}[
            mutation
        ]
        raw[field] = "forged"
    elif mutation == "rating":
        raw["reviews"][0]["observation"]["value"] = 0.99
    elif mutation == "dependency":
        raw["reviews"][0]["dependency_paths"] = [["demographics"]]
    elif mutation == "aggregate":
        raw["nodes"]["ROOT"]["diagnostic"]["support"] = 0.9
    else:
        raw["removals"][0]["affected_assessments"] = []
    raw["checksum"] = digest({k: v for k, v in raw.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, raw)


def test_duplicate_reviews_and_fake_probability_rejected(scored):
    _, scores = scored
    for field, new in [
        ("reviews", [*scores["reviews"], scores["reviews"][0]]),
        ("disease_probability", 0.5),
    ]:
        raw = {**scores, field: new}
        raw["checksum"] = digest({k: v for k, v in raw.items() if k != "checksum"})
        with pytest.raises(ValueError):
            Scores.model_validate(raw)


def test_new_ui_six_sections_numeric_reasons_and_counterfactuals(scored):
    base, scores = scored
    browser(
        attach(base, scores),
        r"""
check(text(elements.detail).includes('第一版评分'),'overview scoring missing');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const numbered=elements.detail.children.filter(x=>x.tag==='h3'&&/^\d ·/.test(x.textContent));
 check(numbered.length===6,'six sections changed');
 check(fifth().includes('本节点研究评分'),'node scoring missing');
}
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('观察可靠性')&&fifth().includes('本例适用性'),'numeric dimensions missing');
check(fifth().includes('未达条件不排除'),'reason lost');
run("show('ROOT')");
check(fifth().includes('移除SegMAN视盘掩膜'),'dependency removal missing');
check(!fifth().includes('本项尚未设置信心权重'),'stale score placeholder');
""",
    )
