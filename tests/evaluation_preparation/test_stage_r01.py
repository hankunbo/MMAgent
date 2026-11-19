"""Review reserve is shared by guards and dispatch, never by acquisition."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import stages
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from packages.multimodal_research.implementation import freeze


def campaign_at(tmp_path, monkeypatch, seconds):
    campaign = RenewedCampaign.__new__(RenewedCampaign)
    campaign.output = tmp_path
    campaign.synthetic = True
    campaign.base = SimpleNamespace(detailed=True, check=lambda: None)
    campaign.protocol = dict(
        schema_version="multimodal-campaign-renewal/1.5.0",
        deadline=(datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(),
        max_new_requests=3,
        previous_attempts=[],
        implementation=freeze(),
    )
    campaign.usage = lambda: 0
    campaign.previous_attempts = lambda: []
    campaign.freeze = lambda: campaign.protocol
    campaign.contexts = SimpleNamespace(read_research_run=lambda *a, **kw: None)
    campaign.request_key = lambda target=None: "synthetic:" + str(target)
    campaign.targets = ()
    (tmp_path / "protocol.json").write_text(json.dumps(campaign.protocol))
    stages.prepare(tmp_path)

    def original_phase(self, target=None):
        assert target is None
        self.guard()
        return dict(accepted=True, evidence_limited=False, run_id="SYNTHETIC_REVIEW")

    monkeypatch.setattr(MultimodalCampaign, "execute_phase", original_phase)
    monkeypatch.setattr(stages, "durable_receipt", lambda *a: None)
    monkeypatch.setattr(stages, "verify", lambda *a: dict(stage="acquisition"))
    monkeypatch.setattr(stages, "seal", lambda *a: dict(stage=a[1], payload=a[3]))
    campaign.expand_definitions = lambda **kwargs: None
    campaign.phase_path = lambda target=None: tmp_path / "review"

    @contextmanager
    def no_credentials(camp, needed):
        # Exercise the real guard at the credential boundary without reading a key.
        if needed:
            camp.guard()
        yield

    monkeypatch.setattr(stages, "credentials", no_credentials)
    return campaign


@pytest.mark.parametrize("seconds", [600, 240])
def test_review_reserve_includes_outer_and_credential_guard(tmp_path, monkeypatch, seconds):
    campaign = campaign_at(tmp_path, monkeypatch, seconds)
    assert campaign.execute_phase()["accepted"]  # Original owner positive control.
    result = stages.run(campaign, "review")
    assert result["stage"] == "review"
    assert campaign._final_review_active is False
    if seconds == 240:
        with pytest.raises(ProviderError, match="deadline"):
            campaign.guard()  # Acquisition still cannot spend the Review reserve.


@pytest.mark.parametrize("reason", ["deadline", "cancelled", "budget"])
def test_review_still_refuses_closed_or_exhausted_window(tmp_path, monkeypatch, reason):
    campaign = campaign_at(tmp_path, monkeypatch, 30 if reason == "deadline" else 240)
    if reason == "cancelled":
        (tmp_path / "cancel").touch()
    if reason == "budget":
        campaign.usage = lambda: 3
    with pytest.raises((ProviderError, ValueError), match=reason):
        stages.run(campaign, "review")
    assert not getattr(campaign, "_final_review_active", False)


def test_phase_context_nested_and_exception_restore(tmp_path, monkeypatch):
    campaign = campaign_at(tmp_path, monkeypatch, 240)
    with campaign.phase_context(None):
        assert campaign._final_review_active
        assert campaign.execute_phase()["accepted"]
        assert campaign._final_review_active
        with pytest.raises(RuntimeError, match="injected"):
            with campaign.phase_context("observation"):
                assert not campaign._final_review_active
                raise RuntimeError("injected")
        assert campaign._final_review_active
    assert not campaign._final_review_active


@pytest.mark.parametrize("stop", ["expired", "cancelled", "budget"])
def test_confirmed_review_ack_is_local_not_new_dispatch(tmp_path, monkeypatch, stop):
    campaign = campaign_at(tmp_path, monkeypatch, 30 if stop == "expired" else 240)
    if stop == "cancelled":
        (tmp_path / "cancel").touch()
    if stop == "budget":
        campaign.usage = lambda: 3
    phase = dict(accepted=True, evidence_limited=False, run_id="SYNTHETIC_REVIEW")
    monkeypatch.setattr(stages, "confirmed_phase", lambda *a: phase)
    checked = []
    monkeypatch.setattr(stages, "local_current", lambda *a: checked.append("current"))
    monkeypatch.setattr(stages, "credentials", lambda *a: pytest.fail("credential assembly"))
    campaign.execute_phase = lambda *a: pytest.fail("phase replay")
    assert stages.run(campaign, "review")["payload"] == phase
    assert checked == ["current"]
    assert not campaign._final_review_active


@pytest.mark.parametrize("change", ["target", "protocol", "mirror", "none"])
def test_confirmed_phase_binds_original_owner_and_mirror(tmp_path, change):
    from db.runtime.repository.case_pipeline import identity

    rid = identity("RUN", "task-research:synthetic")
    protocol = dict(identity="fixed")
    value = dict(run_id=rid, target=None, accepted=True)
    owner = {
        "dynamic-campaign": (protocol,),
        "dynamic-phase-result": (value,),
    }
    if change == "target":
        value["target"] = "other"
    if change == "protocol":
        owner["dynamic-campaign"] = (dict(identity="changed"),)
    if change == "mirror":
        (tmp_path / "result.json").write_text(json.dumps(dict(value, accepted=False)))
    campaign = SimpleNamespace(
        contexts=SimpleNamespace(read_research_run=lambda *a, **kw: object()),
        request_key=lambda target: "synthetic",
        documents=lambda run: SimpleNamespace(read=owner.get),
        protocol=protocol,
        phase_path=lambda target: tmp_path,
    )
    if change == "none":
        assert stages.confirmed_phase(campaign, None) == value
    else:
        with pytest.raises(ValueError, match="stage_phase_binding"):
            stages.confirmed_phase(campaign, None)


def test_source_without_observation_stays_unscored(tmp_path):
    from packages.multimodal_research.stage_failure import render, snapshot

    campaign = SimpleNamespace(
        output=tmp_path,
        targets=(),
        campaign_id="source-only",
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum="synthetic"),
            task=SimpleNamespace(target_code="GOAL:TEST"),
        ),
        protocol=dict(deadline="2000-01-01T00:00:00+00:00", max_new_requests=3),
        usage=lambda: 0,
        phase_path=lambda t: tmp_path / "not-created",
    )
    value = snapshot(campaign, ValueError("configuration"), "sources")
    assert value["schema_version"] == "multimodal-stage-failure/1.0.0"
    assert value["scores"] is None and value["original_review"] is None
    assert value["control"] == dict(exists=False, phase=None, pending=None)
    assert "本轮未评分" in render(value)


@pytest.mark.parametrize("stage", ["sources", "acquisition"])
def test_local_export_does_not_reopen_earlier_unknown_failure(tmp_path, stage):
    protocol = {"task": "original"}
    (tmp_path / "stage-failure.json").write_text(
        json.dumps(
            dict(
                stage=stage,
                protocol_checksum=stages.digest(protocol),
                run_id=None,
                campaign_id="original",
            )
        )
    )
    with pytest.raises(ValueError, match="stage_previous_failure_read_only"):
        stages.recover_export(
            SimpleNamespace(output=tmp_path, protocol=protocol, campaign_id="original")
        )


def test_local_restore_installs_read_only_reader_before_input_check(monkeypatch):
    events = []
    active = True

    def check():
        events.append("inputs")
        if active:
            raise ProviderError("stage_previous_failure_read_only")

    def restore(**kwargs):
        nonlocal active
        assert kwargs == dict(restore_only=True, report_only=True)
        events.append("read_only_sources")
        active = False

    campaign = SimpleNamespace(
        base=SimpleNamespace(check=check),
        expand_definitions=restore,
        accepted_observations=lambda: events.append("facts"),
    )
    monkeypatch.setattr(stages, "review_for_export", lambda *a: None)
    stages.local_current(campaign, None)
    assert events == ["read_only_sources", "inputs", "facts"]
