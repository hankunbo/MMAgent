"""Fresh aggregate acquisitions use the declared original Runtime exclusion."""

from contextlib import nullcontext

import pytest

from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.evaluation_preparation.test_execution_lease import aggregate_fixture


def test_declared_aggregate_profile_reaches_runtime_not_filesystem(tmp_path, monkeypatch):
    fixture = aggregate_fixture(tmp_path)
    campaign = object.__new__(RenewedCampaign)
    campaign.__dict__.update(vars(fixture))
    campaign.protocol["execution_lease_profile"] = lease.PROFILE
    seen = []
    monkeypatch.setattr(lease, "session", lambda owner: seen.append(owner) or nullcontext())

    def filesystem(_):
        raise AssertionError("declared aggregate profile fell through to filesystem lock")

    monkeypatch.setattr(MultimodalCampaign, "execution_session", filesystem)
    with campaign.execution_session():
        pass
    assert seen == [campaign]


def test_fresh_declaration_binds_owner_and_preserves_historical_plan(tmp_path):
    campaign = aggregate_fixture(tmp_path)
    historical = dict(campaign.protocol)
    lease.declare_aggregate(campaign.protocol, campaign.parent)
    assert campaign.protocol["execution_lease_profile"] == lease.PROFILE
    assert "execution_lease_profile" not in historical
    with pytest.raises(ValueError, match="aggregate"):
        lease.declare_aggregate(campaign.protocol, campaign.parent)


@pytest.mark.parametrize("change", ["origin", "state", "review", "initial", "profile", "mode"])
def test_declared_profile_revalidates_scope_before_runtime(tmp_path, monkeypatch, change):
    fixture = aggregate_fixture(tmp_path)
    campaign = object.__new__(RenewedCampaign)
    campaign.__dict__.update(vars(fixture))
    campaign.protocol["execution_lease_profile"] = lease.PROFILE
    if change == "origin":
        campaign.parent.aggregation_baseline.manifest["origin"] = "changed"
    elif change == "state":
        campaign.parent.aggregation_baseline.state["state"] = "changed"
    elif change == "review":
        campaign.protocol["review_target_ids"] = ["other"]
    elif change == "initial":
        campaign.initial_targets = ("other",)
    elif change == "profile":
        campaign.protocol["execution_lease_profile"] = "unknown"
    else:
        campaign.protocol["mode"] = "review"
    entered = []
    monkeypatch.setattr(lease, "session", lambda owner: entered.append(owner) or nullcontext())
    with pytest.raises(ValueError, match="aggregate"):
        campaign.execution_session()
    assert not entered


def test_historical_aggregate_without_profile_keeps_original_lock(tmp_path, monkeypatch):
    fixture = aggregate_fixture(tmp_path)
    campaign = object.__new__(RenewedCampaign)
    campaign.__dict__.update(vars(fixture))
    seen = []
    monkeypatch.setattr(
        MultimodalCampaign, "execution_session", lambda owner: seen.append(owner) or nullcontext()
    )
    with campaign.execution_session():
        pass
    assert seen == [campaign]
