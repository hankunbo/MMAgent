"""A zero-wire retained-context repair is text-only and keeps cumulative limits."""

from types import SimpleNamespace

import pytest


def test_recovery_bounds_spend_only_unused_original_allowance():
    from packages.multimodal_research.acquired_review import retained_projection_bounds

    config = SimpleNamespace(
        review_batching=SimpleNamespace(batch_size=1), transport=SimpleNamespace(max_retries=0)
    )
    proof = dict(
        new_assessment_ids=["new-a", "new-b", "new-c"], unused_original_request_allowance=5
    )
    assert retained_projection_bounds(proof, config) == 4
    assert retained_projection_bounds(proof, config, 3) == 3
    for cap in (0, 2, 6, True):
        with pytest.raises(ValueError, match="retained_review_projection_request_limit"):
            retained_projection_bounds(proof, config, cap)
    with pytest.raises(ValueError, match="retained_review_projection_request_limit"):
        retained_projection_bounds(dict(proof, unused_original_request_allowance=2), config)


def test_recovery_constructor_is_new_read_context_not_parent_mutation(tmp_path, monkeypatch):
    from packages.multimodal_research import acquired_review
    from packages.multimodal_research import campaign_renewal as renewal
    from packages.multimodal_research.campaign import MultimodalCampaign

    reader = object()
    previous = SimpleNamespace()
    parent = SimpleNamespace(
        base=SimpleNamespace(review_target_ids=("old-only",)),
        parent=previous,
        target=None,
        config=SimpleNamespace(),
        protocol=dict(
            tip=dict(run_id="original-accepted"),
            review_clarification=dict(assessment_ids=["clarified"]),
        ),
    )
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.38.0",
        mode="review",
        review_acquisition_gap=dict(original_failure_preserved=True),
        review_target_ids=["old", "new", "clarified"],
        reuse_assessment_ids=["old"],
    )
    validated = []
    monkeypatch.setattr(
        acquired_review, "validate_projection_repair", lambda p, v: validated.append((p, v))
    )
    monkeypatch.setattr(renewal, "accepted_reader", lambda *_a, **_k: reader)
    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self, base, *_a, **_k: setattr(self, "base", base)
    )
    value = renewal.RenewedCampaign(parent, tmp_path, plan)
    assert validated == [(parent, plan)]
    assert value.base is not parent.base
    assert parent.base.review_target_ids == ("old-only",)
    assert value.base.review_context_scope == "current-owner-prior-visible/1.0.0"
    assert value.base.confirmed_review_reader is reader
    assert value.base.review_clarification == parent.protocol["review_clarification"]
    assert value.targets == value.initial_targets == ()
    assert value.config is parent.config


def test_context_recovery_exact_cap_allows_local_finish_not_extra_request(tmp_path, monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from tests.evaluation_preparation.test_exact_acquisition_budget import campaign

    value, calls = campaign(tmp_path, version="1.38.0", used=3, previous=37)
    value.protocol["mode"] = "review"
    value.guard()
    assert calls == ["current-source"]
    monkeypatch.setattr(
        MultimodalCampaign,
        "budget_factory",
        lambda *_a, **_k: SimpleNamespace(before_reserve=lambda _: None),
    )
    with pytest.raises(ProviderError, match="budget"):
        value.budget_factory().before_reserve("extra-not-allowed")


@pytest.mark.parametrize(
    "fault",
    ["unknown", "accepted", "wrong_key", "wrong_run", "missing_observation", "wrong_version"],
)
def test_retained_repair_is_not_generic_retry_authorization(fault):
    from packages.multimodal_research.acquired_review import retained_projection_repair_proof

    def forbidden():
        pytest.fail("invalid origin reached current-owner reads")

    parent = SimpleNamespace(
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.32.0",
            mode="acquisition",
            review_partition="owner_bound_reused_and_new/1.0.0",
            acquisition_targets=["new"],
            previous_attempts=[],
        ),
        config=SimpleNamespace(transport=SimpleNamespace(max_retries=0)),
        base=SimpleNamespace(check=forbidden),
    )
    phase = dict(
        target=None,
        accepted=False,
        acceptance_key=None,
        error="accepted_review_reuse_visible_context_changed",
        run_id="failed-review",
    )
    result = dict(
        complete=False, previous_attempts=[], phases=[dict(target="new", accepted=True), phase]
    )
    if fault == "unknown":
        phase["error"] = "timeout"
    elif fault == "accepted":
        phase["accepted"] = True
    elif fault == "wrong_key":
        phase["acceptance_key"] = "some-other-acceptance"
    elif fault == "wrong_run":
        phase["run_id"] = "another-run"
    elif fault == "missing_observation":
        result["phases"][0]["accepted"] = False
    else:
        parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.26.0"
    with pytest.raises(ValueError, match="retained_review_projection_origin"):
        retained_projection_repair_proof(parent, result, dict(run_id="failed-review"))
