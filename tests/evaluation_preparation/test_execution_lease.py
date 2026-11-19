"""Lease repair changes exclusion only; it cannot authorize existing phase replay."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.campaign_renewal import RETAINED_ACQUISITION_VERSION


def fixture(tmp_path):
    return SimpleNamespace(
        output=tmp_path,
        protocol=dict(schema_version=RETAINED_ACQUISITION_VERSION, mode="acquisition"),
        initial_targets=("a", "b"),
        request_key=lambda target: str(target),
        phase_path=lambda target: tmp_path / str(target),
        contexts=SimpleNamespace(read_research_run=lambda _, *, missing_ok: None),
    )


def test_empty_original_context_inventory_is_required(tmp_path):
    c = fixture(tmp_path)
    lease.unstarted(c)
    c.contexts.read_research_run = lambda _, *, missing_ok: object()
    with pytest.raises(ValueError, match="phase_already_registered"):
        lease.unstarted(c)


def aggregate_fixture(tmp_path):
    from packages.grounding.models import digest

    campaign = fixture(tmp_path)
    baseline = SimpleNamespace(state={"state": "original"}, manifest={"origin": "original"})
    campaign.parent = SimpleNamespace(aggregation_baseline=baseline)
    campaign.protocol.update(
        schema_version="multimodal-campaign-renewal/1.26.0",
        acquisition_increment_only=True,
        acquisition_targets=["a", "b"],
        review_target_ids=["a", "b"],
        accepted_state_checksum=digest(baseline.state),
        accepted_origin_checksum=digest(baseline.manifest),
    )
    return campaign


def test_never_started_aggregate_increment_can_request_explicit_lease(tmp_path):
    campaign = aggregate_fixture(tmp_path)
    lease.unstarted(campaign)
    campaign.contexts.read_research_run = lambda _, *, missing_ok: object()
    with pytest.raises(ValueError, match="phase_already_registered"):
        lease.unstarted(campaign)


@pytest.mark.parametrize(
    "change",
    ["origin", "state", "no_baseline", "plain_acquisition", "review_scope", "target_scope"],
)
def test_aggregate_lease_cannot_substitute_for_original_owner_binding(tmp_path, change):
    campaign = aggregate_fixture(tmp_path)
    if change == "origin":
        campaign.parent.aggregation_baseline.manifest["origin"] = "changed"
    elif change == "state":
        campaign.parent.aggregation_baseline.state["state"] = "changed"
    elif change == "no_baseline":
        campaign.parent.aggregation_baseline = None
    elif change == "plain_acquisition":
        campaign.protocol["acquisition_increment_only"] = False
    elif change == "review_scope":
        campaign.protocol["review_target_ids"] = ["a"]
    else:
        campaign.initial_targets = ("a", "other")
    with pytest.raises(ValueError, match="unstarted_acquisition_scope"):
        lease.unstarted(campaign)


@pytest.mark.parametrize("name", ["result.json", "stage-failure.json", "prewire-repair.json"])
def test_finished_failed_or_other_amendment_cannot_be_replayed(tmp_path, name):
    c = fixture(tmp_path)
    (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="started_or_other_repair"):
        lease.unstarted(c)


def test_even_unconfirmed_local_request_mirror_refuses_repair(tmp_path):
    c = fixture(tmp_path)
    path = c.phase_path("b")
    path.mkdir()
    (path / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="phase_mirror_exists"):
        lease.unstarted(c)


@pytest.mark.parametrize(
    "version,mode", [("old", "acquisition"), (RETAINED_ACQUISITION_VERSION, "review")]
)
def test_other_windows_do_not_implicitly_change_lock_semantics(tmp_path, version, mode):
    c = fixture(tmp_path)
    c.protocol.update(schema_version=version, mode=mode)
    with pytest.raises(ValueError, match="unstarted_acquisition_scope"):
        lease.unstarted(c)


def test_cli_explicit_repair_is_separate_from_run(monkeypatch, capsys):
    from apps.multimodal_case import main

    seen = []
    monkeypatch.setattr(lease, "authorize", lambda *args: seen.append(args) or dict(requests=0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "case",
            "renewal-execution-lease",
            "--input",
            "prepared",
            "--proof",
            "tests.json",
            "--authorization",
            "explicit",
        ],
    )
    assert main() == 0
    assert seen == [("prepared", "tests.json", "explicit")]
    assert '"requests": 0' in capsys.readouterr().out


def test_new_selected_plan_declares_runtime_without_rewriting_old_plan():
    old = dict(schema_version=RETAINED_ACQUISITION_VERSION, mode="acquisition")
    new = dict(old)
    lease.declare_selected(new)
    assert new["execution_lease_profile"] == lease.PROFILE
    assert "execution_lease_profile" not in old
    with pytest.raises(ValueError, match="selected_scope"):
        lease.declare_selected(new)


@pytest.mark.parametrize(
    "version,mode", [("unsupported", "acquisition"), (RETAINED_ACQUISITION_VERSION, "review")]
)
def test_new_selected_lease_cannot_be_applied_to_other_windows(version, mode):
    with pytest.raises(ValueError, match="execution_lease_selected"):
        lease.declare_selected(dict(schema_version=version, mode=mode))


def test_legacy_selected_plan_keeps_original_file_lock(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    campaign = object.__new__(RenewedCampaign)
    campaign.output = tmp_path
    campaign.protocol = dict(schema_version=RETAINED_ACQUISITION_VERSION, mode="acquisition")
    seen = []
    monkeypatch.setattr(
        MultimodalCampaign, "execution_session", lambda self: seen.append("legacy") or nullcontext()
    )
    with campaign.execution_session():
        pass
    assert seen == ["legacy"]
    campaign.protocol["execution_lease_profile"] = "invalid"
    with pytest.raises(ValueError, match="execution_lease_profile"):
        campaign.execution_session()
