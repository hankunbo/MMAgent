"""Versioned dispatch budget; local completion does not consume a request."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.campaign_renewal import (
    EXACT_ACQUISITION_VERSION,
    RenewedCampaign,
    validate_exact_acquisition_budget,
)


def campaign(tmp_path, *, version="1.32.0", used=3, previous=37):
    value = object.__new__(RenewedCampaign)
    value.output = tmp_path
    calls = []
    value.base = SimpleNamespace(check=lambda: calls.append("current-source"), detailed=False)
    history = [dict(requests=previous)]
    value.protocol = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="acquisition",
        max_new_requests=3,
        max_model_dispatches=3,
        max_cumulative_requests=40,
        previous_attempts=history,
        deadline=(datetime.now(UTC) + timedelta(seconds=500)).isoformat(),
    )
    value.previous_attempts = lambda: history
    value.usage = lambda: used
    return value, calls


def test_exact_cap_local_checks_keep_current_owner(tmp_path):
    value, calls = campaign(tmp_path)
    value.guard()
    assert calls == ["current-source"]


def test_exact_cap_refuses_next_reservation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        MultimodalCampaign,
        "budget_factory",
        lambda *_a, **_k: SimpleNamespace(before_reserve=lambda sha: calls.append(sha)),
    )
    value, _ = campaign(tmp_path, used=2)
    budget = value.budget_factory()
    budget.before_reserve("last-allowed")
    value.usage = lambda: 3
    with pytest.raises(ProviderError, match="budget"):
        budget.before_reserve("not-allowed")
    assert calls == ["last-allowed"]


def test_old_protocol_keeps_original_budget_behavior(tmp_path):
    value, _ = campaign(tmp_path, version="1.26.0")
    with pytest.raises(ProviderError, match="budget"):
        value.guard()


@pytest.mark.parametrize("used,previous", [(4, 37), (3, 38)])
def test_over_bound_still_rejected(tmp_path, used, previous):
    value, _ = campaign(tmp_path, used=used, previous=previous)
    with pytest.raises(ProviderError, match="budget"):
        value.guard()


@pytest.mark.parametrize("fault", ["cancel", "deadline", "source"])
def test_local_completion_is_not_current_authorization_bypass(tmp_path, fault):
    value, _ = campaign(tmp_path)
    error, message = ProviderError, "cancelled"
    if fault == "cancel":
        (tmp_path / "cancel").touch()
    elif fault == "deadline":
        value.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        message = "deadline"
    else:

        def withdrawn():
            raise ValueError("current_source_withdrawn")

        value.base.check = withdrawn
        error, message = ValueError, "current_source_withdrawn"
    with pytest.raises(error, match=message):
        value.guard()


def valid_plan():
    return dict(
        schema_version=EXACT_ACQUISITION_VERSION,
        mode="acquisition",
        acquisition_targets=["a", "b"],
        max_new_requests=3,
        max_model_dispatches=3,
        max_cumulative_requests=40,
        previous_attempts=[dict(requests=37)],
        exact_request_budget="dispatch_only_local_completion_zero_request/1.0",
        review_partition="owner_bound_reused_and_new/1.0.0",
        acquisition_selection=dict(schema_version="bounded-acquisition-selection/1.1.0"),
    )


def configuration():
    return SimpleNamespace(
        review_batching=SimpleNamespace(batch_size=3), transport=SimpleNamespace(max_retries=0)
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="review"),
        dict(max_new_requests=2),
        dict(max_new_requests=4),
        dict(max_model_dispatches=4),
        dict(max_cumulative_requests=41),
        dict(exact_request_budget=None),
        dict(acquisition_increment_only=True),
        dict(continue_acquisition=True),
        dict(review_partition=None),
        dict(acquisition_targets=[]),
        dict(acquisition_selection={}),
        dict(schema_version="multimodal-campaign-renewal/1.26.0"),
    ],
)
def test_changed_exact_plan_rejected(change):
    plan = valid_plan()
    validate_exact_acquisition_budget(plan, configuration())
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        validate_exact_acquisition_budget(dict(plan, **change), configuration())


def test_real_retry_and_batch_count_consumes_budget():
    config = configuration()
    config.transport.max_retries = 1
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        validate_exact_acquisition_budget(valid_plan(), config)
    config.transport.max_retries = 0
    config.review_batching.batch_size = 1
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        validate_exact_acquisition_budget(valid_plan(), config)


def test_reserve_counts_actual_dispatch_not_local_closure():
    from packages.multimodal_research.active_continuation import reserve_current_review

    args = dict(
        seconds_left=3000,
        assessment_ids=["one"],
        batch_size=3,
        timeout_seconds=120,
        max_retries=0,
        requests_used=1,
        request_limit=3,
        policy=dict(
            schema_version="review-work-reserve/1.0.0",
            local_seconds_per_batch=60,
            close_seconds=300,
        ),
    )
    old = reserve_current_review(**args)
    new = reserve_current_review(**args, local_request_headroom=0)
    assert old["stop_acquisition"] and not new["stop_acquisition"]
    assert old["review_reserve_seconds"] == new["review_reserve_seconds"]
    assert new["review_reserve_requests"] == 1
    assert "request_budget_semantics" not in old


def test_invalid_cli_scope_rejected_before_owner_assembly(monkeypatch):
    from packages.multimodal_research import campaign_renewal

    def forbidden(*args, **kwargs):
        pytest.fail("invalid scope assembled an owner")

    monkeypatch.setattr(campaign_renewal, "completed_parent", forbidden)
    with pytest.raises(ValueError, match="renewal_exact_request_budget_scope"):
        campaign_renewal.prepare(
            facts="unused",
            previous="unused",
            output="unused",
            mode="review",
            authorization="test",
            exact_request_budget=True,
        )
