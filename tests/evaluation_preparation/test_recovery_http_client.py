"""Explicit original renewal client transition, never implicit history rewriting."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.client_selection import apply_selection, load_selection
from packages.llm_provider.config import with_network_route
from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research.campaign_renewal import (
    RETAINED_ACQUISITION_VERSION,
    RenewedCampaign,
    prepare,
    recovery_configuration,
)
from tests.evaluation_preparation.test_recovery_network_route import origin

PATH = "configs/agents/system_curl_client_v1.json"


def pair():
    old = with_network_route(
        origin(), prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"})
    )
    binding = load_selection(PATH)
    new = apply_selection(old, binding)
    plan = dict(
        schema_version=RETAINED_ACQUISITION_VERSION,
        mode="acquisition",
        recovery_http_client=binding,
        provider=digest(new),
        previous_attempts=[dict(requests=9)],
        max_cumulative_requests=40,
    )
    return old, new, plan


def test_client_changes_only_version_and_explicit_transport():
    old, expected, plan = pair()
    before, frozen = old.model_dump(mode="json"), deepcopy(plan)
    actual = recovery_configuration(old, plan)
    assert actual == expected
    after = actual.model_dump(mode="json")
    assert after.pop("http_client") == plan["recovery_http_client"]["client"]
    after["schema_version"] = before["schema_version"]
    assert after == before
    assert old.model_dump(mode="json") == before
    assert plan == frozen
    assert recovery_configuration(old, {}) is old


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "mode",
        "null",
        "binding",
        "provider",
        "combined_route",
        "unknown",
        "clarification",
    ],
)
def test_client_does_not_authorize_unrelated_renewal_scope(fault):
    old, _, plan = pair()
    if fault == "version":
        plan["schema_version"] = "multimodal-campaign-renewal/1.32.0"
    elif fault == "mode":
        plan["mode"] = "review"
    elif fault == "null":
        plan["recovery_http_client"] = None
    elif fault == "binding":
        plan["recovery_http_client"]["client_checksum"] = digest("changed")
    elif fault == "provider":
        plan["provider"] = digest(old)
    elif fault == "combined_route":
        plan["recovery_network_route"] = old.network_route.model_dump(mode="json")
    elif fault == "unknown":
        plan["unknown_acquisition"] = dict(selected=["unknown"])
    else:
        plan["review_clarification"] = dict(assessment_ids=["old"])
    with pytest.raises(ValueError, match="recovery_http_client|client_selection"):
        recovery_configuration(old, plan)


def test_constructor_rejects_smuggled_client_before_original_assembly(tmp_path):
    old, _, plan = pair()
    plan["schema_version"] = "multimodal-campaign-renewal/1.23.0"
    with pytest.raises(ValueError, match="recovery_http_client"):
        RenewedCampaign(SimpleNamespace(config=old), tmp_path, plan)


@pytest.mark.parametrize(
    "mode,exact,targets",
    [
        ("review", False, ()),
        ("definitions", False, ()),
        ("acquisition", False, ("active:new",)),
        ("acquisition", True, ()),
    ],
)
def test_prepare_rejects_before_history_or_credentials(tmp_path, mode, exact, targets):
    with pytest.raises(ValueError, match="recovery_http_client_scope"):
        prepare(
            facts="not-read",
            previous="not-read",
            output=tmp_path,
            mode=mode,
            authorization="synthetic",
            exact_request_budget=exact,
            acquisition_target=targets,
            recovery_http_client=PATH,
        )


def test_cli_explicit_client_routes_to_original_prepare(monkeypatch):
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
            "acquisition",
            "--authorization",
            "synthetic",
            "--exact-request-budget",
            "--acquisition-target",
            "active:new",
            "--request-limit",
            "2",
            "--recovery-http-client",
            PATH,
        ],
    )
    assert multimodal_case.main() == 0
    assert seen["recovery_http_client"] == PATH
    assert seen["acquisition_target"] == ["active:new"]
