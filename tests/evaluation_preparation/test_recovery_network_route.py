"""New text recovery uses an explicit route without rewriting old facts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.campaign_renewal import (
    RETAINED_REVIEW_VERSION,
    RenewedCampaign,
    prepare,
    recovery_configuration,
)
from tests.evaluation_preparation.test_unstarted_pending_window import fixture


@pytest.fixture
def route(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8888")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}).model_dump(mode="json")


def origin():
    return configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )


@pytest.mark.parametrize(
    "version",
    [
        RETAINED_REVIEW_VERSION,
        "multimodal-campaign-renewal/1.39.0",
        "multimodal-campaign-renewal/1.42.0",
    ],
)
def test_new_route_only_changes_transport_envelope(version, route):
    old = origin()
    before = old.model_dump(mode="json")
    plan = dict(schema_version=version, mode="review", recovery_network_route=route)
    new = recovery_configuration(old, plan)
    after = new.model_dump(mode="json")
    assert after.pop("network_route") == route
    after["schema_version"] = before["schema_version"]
    assert after == before
    assert old.model_dump(mode="json") == before
    assert digest(new) != digest(old)
    assert recovery_configuration(new, plan) is new
    assert recovery_configuration(old, {}) is old


@pytest.mark.parametrize("change", ["version", "mode", "null", "environment", "bypass"])
def test_route_cannot_broaden_scope_or_silently_change_network(change, route, monkeypatch):
    plan = dict(schema_version=RETAINED_REVIEW_VERSION, mode="review", recovery_network_route=route)
    if change == "version":
        plan["schema_version"] = "multimodal-campaign-renewal/1.33.0"
    elif change == "mode":
        plan["mode"] = "acquisition"
    elif change == "null":
        plan["recovery_network_route"] = None
    elif change == "environment":
        monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8890")
    else:
        monkeypatch.setenv("NO_PROXY", "*")
    with pytest.raises(ValueError, match="recovery_network_route|network_proxy"):
        recovery_configuration(origin(), plan)


def test_pending_new_window_binds_new_provider_but_preserves_old_budget(route):
    from packages.multimodal_research.pending_review import validate_unstarted_window

    parent, plan = fixture()
    parent.config = origin()
    history = deepcopy(plan["previous_attempts"])
    plan["recovery_network_route"] = route
    plan["provider"] = digest(recovery_configuration(parent.config, plan))
    validate_unstarted_window(plan, parent)
    assert plan["previous_attempts"] == history
    assert plan["max_cumulative_requests"] == 40
    assert plan["pending_review"]["quarantined_batches"][0]["assessment_ids"] == ["old:unknown"]
    plan["provider"] = digest(parent.config)
    with pytest.raises(ValueError, match="pending_unstarted_window_binding"):
        validate_unstarted_window(plan, parent)


def test_old_constructor_cannot_smuggle_new_route(tmp_path, route):
    with pytest.raises(ValueError, match="recovery_network_route_scope"):
        RenewedCampaign(
            SimpleNamespace(config=origin()),
            tmp_path,
            dict(
                schema_version="multimodal-campaign-renewal/1.23.0",
                mode="review",
                recovery_network_route=route,
            ),
        )


def test_prepare_rejects_other_modes_before_reading_history(tmp_path):
    with pytest.raises(ValueError, match="recovery_network_route_scope"):
        prepare(
            facts="not-read",
            previous="not-read",
            output=tmp_path,
            mode="acquisition",
            authorization="synthetic",
            recovery_network_route="not-read",
        )


def test_cli_exposes_explicit_route_to_original_prepare(monkeypatch):
    import sys

    from apps import multimodal_case
    from packages.multimodal_research import campaign_renewal

    seen = {}
    monkeypatch.setattr(campaign_renewal, "prepare", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "synthetic",
            "--pending-review",
            "--pending-new-window",
            "--recovery-network-route",
            "configs/agents/environment_https_route_v1.json",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen["recovery_network_route"] == "configs/agents/environment_https_route_v1.json"
    assert seen["pending_new_window"] is True
