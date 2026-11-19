"""New selective-acquisition context policy, with exact legacy dispatch rules."""

import pytest

from packages.multimodal_research.campaign_renewal import (
    acquisition_review_ids,
    validate_exact_acquisition_budget,
)
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration, valid_plan

VERSION = "multimodal-campaign-renewal/1.37.0"
SCOPE = "current-owner-prior-visible/1.0.0"


def test_retained_version_keeps_exact_request_budget():
    plan = dict(valid_plan(), schema_version=VERSION, review_context_scope=SCOPE)
    validate_exact_acquisition_budget(plan, configuration())
    plan["max_new_requests"] = 2
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        validate_exact_acquisition_budget(plan, configuration())


@pytest.mark.parametrize("scope", [None, "caller_validated"])
def test_retained_version_cannot_omit_or_replace_scope(scope):
    plan = dict(valid_plan(), schema_version=VERSION, review_context_scope=scope)
    with pytest.raises(ValueError, match="renewal_retained_context_scope"):
        validate_exact_acquisition_budget(plan, configuration())


def test_old_version_cannot_silently_opt_in():
    plan = dict(valid_plan(), review_context_scope=SCOPE)
    with pytest.raises(ValueError, match="renewal_retained_context_scope"):
        validate_exact_acquisition_budget(plan, configuration())


def test_new_context_scope_keeps_explicit_clarification_and_disjoint_ids():
    plan = dict(
        valid_plan(),
        schema_version=VERSION,
        review_context_scope=SCOPE,
        review_clarification=dict(assessment_ids=["old-clarified"]),
        reuse_assessment_ids=["old-reused"],
        new_observation_execution_allowed=True,
    )
    validate_exact_acquisition_budget(plan, configuration())
    assert acquisition_review_ids(plan) == ("a", "b", "old-clarified", "old-reused")
    plan["reuse_assessment_ids"] = ["old-clarified"]
    with pytest.raises(ValueError, match="renewal_acquisition_review_scope"):
        acquisition_review_ids(plan)
