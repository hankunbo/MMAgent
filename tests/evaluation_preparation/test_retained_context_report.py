"""Retained original-owner observations are visible but do not gain a vote."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest
from packages.multimodal_research.report import attach_retained_context, present_retained_context
from tests.evaluation_preparation.test_retained_review_context import projection_fixture


def fixture(tmp_path):
    phase, _, omitted = projection_fixture(tmp_path)
    phase.base.review_context_scope = "current-owner-prior-visible/1.0.0"
    parent = phase.current_projection()
    selected = {p.id for p in parent.policy.items}
    by_id = phase.base.observations
    rows = [
        dict(
            id=r.id,
            observation=by_id[r.id].model_dump(mode="json") if r.id in selected else None,
            sources=[v.model_dump(mode="json") for v in r.definition.source_refs],
            status="fixture",
        )
        for r in parent.policy.requirements
    ]
    campaign = SimpleNamespace(
        base=phase.base,
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.38.0",
            review_context_scope=phase.base.review_context_scope,
            reuse_assessment_ids=[next(iter(selected))],
        ),
        accepted_observations=lambda: {},
    )
    report = dict(
        result=dict(case_code=parent.material.scope.case_code),
        requirements=rows,
        scores=dict(reviews=[dict(id=k) for k in sorted(selected)]),
        counts=dict(acquired=len(selected)),
        acquired_count=len(selected),
        completion=dict(all_obtained_reviewed=True),
        task_conclusion=dict(limitation="Not clinical validation."),
    )
    return campaign, report, parent, omitted


def test_retained_fact_count_and_six_sections_without_new_vote(tmp_path):
    campaign, report, parent, omitted = fixture(tmp_path)
    original_score = digest(report["scores"])
    original_parent = digest(parent)
    attach_retained_context(campaign, report, parent)
    rendered = produce(report, parent)
    present_retained_context(rendered)
    row = next(r for r in rendered["requirements"] if r["id"] == omitted)
    actual = campaign.base.observations[omitted].model_dump(mode="json")
    assert row["observation"] is None
    assert row["unscored_observation"] == row["presentation_sections"]["observation"] == actual
    assert row["presentation_sections"]["score_pointer"] is None
    assert row["execution_binding"]["observation_reference"] == actual["reference"]
    assert row["execution_binding"]["unscored"] is True
    assert rendered["counts"]["acquired"] == len(parent.policy.items) + 1
    assert rendered["counts"]["retained_unscored_observations"] == 1
    assert rendered["counts"]["review_scope_observations"] == len(parent.policy.items)
    assert rendered["completion"]["selected_increment_reviewed"] is True
    assert rendered["completion"]["all_obtained_reviewed"] is False
    assert rendered["retained_review_context"]["extra_scoring_votes"] == 0
    assert digest(rendered["scores"]) == original_score
    assert digest(parent) == original_parent


def test_old_report_unchanged(tmp_path):
    campaign, report, parent, _ = fixture(tmp_path)
    campaign.protocol = dict(schema_version="multimodal-campaign-renewal/1.33.0")
    before = deepcopy(report)
    attach_retained_context(campaign, report, parent)
    assert report == before


@pytest.mark.parametrize("change", ["scope", "fact", "target", "scored"])
def test_report_context_binding_rejects_inconsistency(tmp_path, change):
    campaign, report, parent, omitted = fixture(tmp_path)
    if change == "scope":
        campaign.protocol["review_context_scope"] = "caller-trust-me"
    elif change == "fact":
        old = campaign.base.observations[omitted]
        campaign.base.observations[omitted] = old.model_copy(
            update={"value": old.value + " changed"}
        )
    elif change == "target":
        report["requirements"] = [r for r in report["requirements"] if r["id"] != omitted]
    else:
        row = next(r for r in report["requirements"] if r["id"] == omitted)
        row["observation"] = campaign.base.observations[omitted].model_dump(mode="json")
    with pytest.raises(ValueError, match="report_retained_context"):
        attach_retained_context(campaign, report, parent)
