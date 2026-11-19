"""Fresh experiment scope, no private launcher/network or native inference."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import definition_trial as trial


def envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "freeze", lambda: {"frozen": "same"})
    spec = dict(schema_version="multimodal-live-entry/1.13.0", implementation=trial.freeze())
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    return trial.prepare(
        tmp_path,
        expansion_policy="configs/research/multimodal_definition_gate_v5.json",
        authorization="explicit independent synthetic experiment",
    )


def test_offline_envelope_does_not_start_a_clock_or_run(tmp_path, monkeypatch):
    value = envelope(tmp_path, monkeypatch)
    assert trial.read(tmp_path) == value
    assert set(p.name for p in tmp_path.iterdir()) == {"entry.json", "definition-trial.json"}
    with pytest.raises(ValueError, match="initialize_required"):
        trial.cold(tmp_path)


@pytest.mark.parametrize("change", ["entry", "implementation", "cap", "policy", "history"])
def test_envelope_drift_or_scope_rejected(tmp_path, monkeypatch, change):
    value = envelope(tmp_path, monkeypatch)
    spec = json.loads((tmp_path / "entry.json").read_bytes())
    if change == "entry":
        spec["sample"] = "other"
    elif change == "implementation":
        monkeypatch.setattr(trial, "freeze", lambda: {"frozen": "changed"})
    elif change == "cap":
        value["max_new_requests"] = 40
    elif change == "policy":
        value["expansion_policy"]["schema_version"] = "campaign-definition-expansion/1.2.0"
    else:
        spec["previous"] = "old-unknown"
        value["entry_checksum"] = digest(spec)
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    (tmp_path / "definition-trial.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        trial.read(tmp_path)


def scoped(tmp_path, seconds=240, used=0):
    c = trial.DefinitionTrialCampaign.__new__(trial.DefinitionTrialCampaign)
    c.trial = dict(schema_version="multimodal-definition-trial/1.0.0", max_new_requests=12)
    c.output = tmp_path
    c.base = SimpleNamespace(check=lambda: None, detailed=True)
    c.protocol = dict(
        deadline=(datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(),
        max_new_requests=12,
        previous_attempts=[],
    )
    c.usage = lambda: used
    c.previous_attempts = lambda: []
    return c


def test_review_reserve_and_restore_are_shared_by_outer_guard(tmp_path):
    c = scoped(tmp_path)
    with pytest.raises(ProviderError, match="deadline"):
        c.guard()
    with c.phase_context(None):
        c.guard()
        with c.phase_context("image"):
            with pytest.raises(ProviderError, match="deadline"):
                c.guard()
        c.guard()
    assert not c._final_review_active


@pytest.mark.parametrize("reason", ["cancelled", "deadline", "budget"])
def test_scoped_review_never_bypasses_stop(tmp_path, reason):
    c = scoped(
        tmp_path, seconds=30 if reason == "deadline" else 240, used=12 if reason == "budget" else 0
    )
    if reason == "cancelled":
        (tmp_path / "cancel").touch()
    with c.phase_context(None), pytest.raises(ProviderError, match=reason):
        c.guard()


def test_definition_reserve_uses_trial_cap_not_legacy_forty(tmp_path):
    from packages.multimodal_research.definition_expansion import (
        DefinitionExpansion,
        read_expansion_policy,
    )

    c = scoped(tmp_path, seconds=900, used=9)
    e = DefinitionExpansion.__new__(DefinitionExpansion)
    e.campaign = c
    e.control = lambda: None
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v5.json").read_bytes())
    )
    assert e.policy.reserve_review_requests == 3
    with pytest.raises(ProviderError, match="definition_review_reserve"):
        e.guard()


@pytest.mark.parametrize("used,stop", [(8, False), (9, True)])
def test_acquisition_reserves_review_with_trial_limit(tmp_path, used, stop):
    c = scoped(tmp_path, seconds=3600, used=used)
    c.trial["expansion_policy"] = dict(reserve_review_requests=3)
    c.base.active_schedule_v19 = True
    c.base.manifest = SimpleNamespace(checksum="input")
    c.base.observations = {}
    c.accepted_observations = lambda: {}
    c.config = SimpleNamespace(review_batching=SimpleNamespace(batch_size=2))
    assert c.active_review_reserved() is stop
    if stop:
        record = json.loads((tmp_path / "acquisition-budget-stop.json").read_bytes())
        assert record["review_reserve_requests"] == 3
        assert record["protocol_checksum"] == digest(c.protocol)
        c.usage = lambda: 0
        assert c.active_review_reserved()  # Closed acquisition never reopens.


def test_each_revision_install_narrows_schedule_and_review(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = scoped(tmp_path)
    revision = dict(added=["new"], replaced=[dict(id="changed")], rows=[dict(id="old")])

    def install(self, **kwargs):
        self.targets = ("old", "new", "changed")
        return revision

    monkeypatch.setattr(MultimodalCampaign, "expand_definitions", install)
    for _ in range(2):
        assert c.expand_definitions(restore_only=True) == revision
        assert c.targets == ("new",)
        assert c.base.review_target_ids == ("changed", "new")
    assert revision["rows"] == [dict(id="old")]


def test_stage_cold_dispatch_uses_explicit_version(monkeypatch, tmp_path):
    from packages.multimodal_research import stages

    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(schema_version="multimodal-live-campaign/1.5.0"))
    )
    sentinel = object()
    monkeypatch.setattr(trial, "cold", lambda p: sentinel)
    assert stages.assemble_for_stage(tmp_path) is sentinel


def test_native_is_deferred_and_cold_install_only_reads(tmp_path, monkeypatch):
    from packages.multimodal_research import active_acquisition as active

    base = SimpleNamespace(
        output=tmp_path,
        manifest=SimpleNamespace(checksum="fixed", components=[]),
        sources=SimpleNamespace(initial={}),
        task=object(),
        prepared=SimpleNamespace(payload_bytes=[]),
    )
    rows = [dict(id="native", component="image", acquisition_method="disc_cup_measurement")]
    plan = dict(rows=[dict(id="native")])
    monkeypatch.setattr(active, "project", lambda *a: (rows, plan))
    monkeypatch.setattr(active.ActiveOwner, "check", lambda self: None)
    profile = dict(definition_expansion={})

    def forbidden(*a, **kw):
        pytest.fail("native or network dispatch")

    active.install(base, profile, native_runner=forbidden, discoverer=forbidden, defer_native=True)
    saved = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(active, "write_once", forbidden)
    active.install(
        base,
        profile,
        native_runner=forbidden,
        discoverer=forbidden,
        defer_native=True,
        read_only=True,
    )
    assert saved == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert base.ledger == rows and base.observations == {} and base.active_owner.native is None
    (tmp_path / "active-plan.json").unlink()
    with pytest.raises(FileNotFoundError):
        active.install(base, profile, defer_native=True, read_only=True)


def test_invalid_envelope_is_rejected_before_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "freeze", lambda: {})
    (tmp_path / "entry.json").write_text(
        json.dumps(dict(schema_version="multimodal-live-entry/1.13.0", implementation={}))
    )
    policy = json.loads(Path("configs/research/multimodal_definition_gate_v5.json").read_bytes())
    policy["reserve_review_requests"] = 12
    config = tmp_path / "policy.json"
    config.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="definition_trial_policy"):
        trial.prepare(tmp_path, expansion_policy=config, authorization="synthetic invalid budget")
    assert not (tmp_path / "definition-trial.json").exists()
    policy["reserve_review_requests"] = 3
    config.write_text(json.dumps(policy))
    assert trial.prepare(tmp_path, expansion_policy=config, authorization="synthetic valid budget")


@pytest.mark.parametrize("change", ["deadline", "duration", "none"])
def test_frozen_trial_clock_bound_to_initial_lease(tmp_path, monkeypatch, change):
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = scoped(tmp_path, seconds=3600)
    start = datetime.now(UTC)
    lease = dict(
        started_at=start.isoformat(), deadline=(start + timedelta(seconds=3600)).isoformat()
    )
    c.base.active_start = lease
    c.trial["seconds"] = 3600
    c.protocol.update(lease)
    if change == "deadline":
        c.protocol["deadline"] = (start + timedelta(days=1)).isoformat()
    elif change == "duration":
        c.trial["seconds"] = 1200
    monkeypatch.setattr(MultimodalCampaign, "freeze", lambda self: self.protocol)
    if change == "none":
        assert c.freeze() == c.protocol
    else:
        with pytest.raises(ValueError, match="lifetime_binding"):
            c.freeze()


@pytest.mark.parametrize("changed", [False, True])
def test_initialization_proof_is_original_owner_not_local_checksum(tmp_path, changed):
    c = scoped(tmp_path, seconds=3600)
    c.request_key = lambda target: "synthetic-init"
    c.contexts = SimpleNamespace(read_research_run=lambda *a, **kw: object())
    frozen = dict(c.protocol)
    c.documents = lambda run: SimpleNamespace(read=lambda key: (frozen,))
    if changed:
        c.protocol["deadline"] = (datetime.now(UTC) + timedelta(days=2)).isoformat()
        with pytest.raises(ValueError, match="initialization_binding"):
            c.verify_initialization()
    else:
        c.verify_initialization()
