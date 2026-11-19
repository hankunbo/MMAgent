"""Current public entry recovery keeps old windows and real facts immutable."""

from packages.multimodal_research.campaign import configuration
from packages.multimodal_research.campaign_renewal import confirmed_facts_configuration


def test_text_window_is_explicit_single_item_not_old_config_mutation():
    old = configuration(resilient_vision=True, directional=True, exact_primary_status=True)
    before = old.model_dump(mode="json")
    new = confirmed_facts_configuration()
    from packages.evidence_details.live_permission import validate_live_config

    assert validate_live_config(new) == new
    assert new.transport.model == old.transport.model
    assert new.transport.timeout_seconds == 300
    assert new.transport.max_retries == 0
    assert new.review_batching.batch_size == 1
    assert new.review_correction_policy == old.review_correction_policy
    assert old.model_dump(mode="json") == before
    assert [(r.role, r.prompt_version) for r in new.roles] == [
        (r.role, r.prompt_version) for r in old.roles
    ]


def test_new_window_does_not_select_acquisition_targets():
    from types import SimpleNamespace

    from packages.grounding.models import digest
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    c = object.__new__(RenewedCampaign)
    facts = {"observation": "saved-owner-value"}
    c.parent = SimpleNamespace(accepted_observations=lambda: facts)
    c.protocol = {"confirmed_observations_checksum": digest(facts)}
    c.targets = ()
    assert c.accepted_observations() == facts


def test_live_owner_still_rejects_more_than_one_retry():
    import pytest

    from packages.evidence_details.live_permission import validate_live_config
    from packages.llm_provider.config import read_provider_config

    raw = confirmed_facts_configuration().model_dump(mode="json")
    raw["transport"]["max_retries"] = 2
    with pytest.raises(ValueError, match="bounded_live_configuration_required"):
        validate_live_config(read_provider_config(raw))


def test_prewire_repair_rejects_any_existing_control(monkeypatch, tmp_path):
    import fcntl
    import json
    from types import SimpleNamespace

    import pytest

    from packages.multimodal_research import campaign_renewal as module

    request = {"scope": {"case_code": "CASE:SYNTHETIC"}}
    (tmp_path / "request.json").write_text(json.dumps(request))
    ctx = SimpleNamespace(request=SimpleNamespace(model_dump=lambda **kw: request))
    docs = SimpleNamespace(run_id="RUN:SYNTHETIC")
    c = SimpleNamespace(
        output=tmp_path,
        protocol={"confirmed_facts_review": True},
        guard=lambda: None,
        phase_path=lambda target: tmp_path,
        request_key=lambda target: "key",
        contexts=SimpleNamespace(read_research_run=lambda rid: ctx),
        base=SimpleNamespace(contexts={}),
        documents=lambda rid: docs,
        sessions=SimpleNamespace(read=lambda rid: object()),
    )
    monkeypatch.setattr(module, "assemble", lambda output: c)
    monkeypatch.setattr(module, "RuntimeAgentJournal", lambda d: SimpleNamespace(read=lambda: []))
    # This unit test isolates the existing-Control gate, not filesystem locks.
    # The original Runtime lease exclusion has its own owned-PG integration test.
    monkeypatch.setattr(fcntl, "flock", lambda *args: None)
    with pytest.raises(ValueError, match="renewal_prewire_not_unstarted"):
        module.authorize_prewire_repair(tmp_path, "unused", "explicit-user")
    assert not (tmp_path / "prewire-repair.json").exists()
