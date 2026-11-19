"""Reading a prior proof never authorizes a changed route for new work."""

from types import SimpleNamespace

import pytest

from packages.llm_provider.network_route import prepare_route
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unknown_acquisition as owner
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration


def fixture(monkeypatch):
    config = configuration()
    config.transport.base_url = "https://example.invalid/v1"
    config.network_route = prepare_route(environ={"HTTPS_PROXY": "http://127.0.0.1:7890"})
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7891")
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    proof = dict(selected=["target"], attempts=[dict(request_id="old")])
    parent = SimpleNamespace(config=config, accepted_observations=lambda: {})
    monkeypatch.setattr(owner, "selection", lambda *_: proof)
    plan = dict(
        schema_version=owner.VERSION,
        unknown_acquisition=proof,
        mode="acquisition",
        seconds=2700,
        new_public_search_allowed=False,
        acquisition_targets=["target"],
        previous_attempts=[],
        started_at="2026-01-01T00:00:00+00:00",
        deadline="2026-01-01T00:45:00+00:00",
        max_new_requests=4,
    )
    return parent, plan


def test_history_proof_does_not_depend_on_current_proxy(monkeypatch):
    parent, plan = fixture(monkeypatch)
    assert owner.check(parent, plan) == ["target"]
    with pytest.raises(ValueError, match="unknown_acquisition_plan_binding"):
        owner.check(parent, dict(plan, acquisition_targets=["wrong"]))
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        owner.check_network(parent.config)


def test_current_campaign_guard_still_checks_route(monkeypatch, tmp_path):
    parent, plan = fixture(monkeypatch)
    campaign = object.__new__(renewal.RenewedCampaign)
    campaign.parent, campaign.protocol, campaign.output = parent, plan, tmp_path
    monkeypatch.setattr(renewal.RenewedCampaign, "usage", lambda self: 0)
    monkeypatch.setattr(renewal.MultimodalCampaign, "guard", lambda *_, **__: None)
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        campaign.guard(dispatch=True)


def test_public_run_refuses_route_before_execute(monkeypatch):
    parent, _ = fixture(monkeypatch)
    executed = []
    parent.execute = lambda: executed.append(True)
    monkeypatch.setattr(renewal, "assemble", lambda _: parent)
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        renewal.run("synthetic")
    assert executed == []
