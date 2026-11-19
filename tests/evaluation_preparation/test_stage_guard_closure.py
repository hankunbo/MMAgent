"""A refused dispatch still leaves a truthful immutable technical snapshot."""

import json
from types import SimpleNamespace

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import stages
from tests.evaluation_preparation.test_stage_r01 import campaign_at


@pytest.mark.parametrize("reason", ["cancelled", "deadline", "budget"])
def test_guard_refusal_keeps_error_and_records_no_new_dispatch(tmp_path, monkeypatch, reason):
    campaign = campaign_at(tmp_path, monkeypatch, 30 if reason == "deadline" else 240)
    campaign.campaign_id = "guard-test"
    campaign.base.manifest = SimpleNamespace(checksum="test-input")
    campaign.base.task = SimpleNamespace(target_code="GOAL:TEST")
    if reason == "cancelled":
        (tmp_path / "cancel").touch()
    if reason == "budget":
        campaign.usage = lambda: 3
    monkeypatch.setattr(stages, "credentials", lambda *a: pytest.fail("no credentials"))
    with pytest.raises((ProviderError, ValueError), match=reason):
        stages.run(campaign, "review")

    saved = json.loads((tmp_path / "stage-failure.json").read_bytes())
    assert saved["error_code"] == reason
    assert saved["scores"] is None
    assert saved["automatic_replay_authorized"] is False
    assert saved["control"]["exists"] is False
    assert not campaign._final_review_active


def test_cold_definition_readback_uses_policy_defaults_without_creating_budget(tmp_path):
    from db.runtime.repository.case_pipeline import identity
    from packages.multimodal_research.stage_failure import saved_definition_state

    protocol = dict(deadline="2026-01-01T00:00:00+00:00")
    rid = identity("RUN", "task-research:definition-expansion")
    reads = []

    def read(key):
        reads.append(key)
        return (protocol,) if key == "dynamic-campaign" else None

    docs = SimpleNamespace(run_id=rid, read=read)
    campaign = SimpleNamespace(
        contexts=SimpleNamespace(read_research_run=lambda *a, **kw: object()),
        request_key=lambda target: target,
        documents=lambda run: docs if run == rid else pytest.fail("wrong run"),
        protocol=protocol,
        base=SimpleNamespace(
            definition_expansion_policy=dict(index_keys=[], public_queries=["Glaucoma"])
        ),
    )
    restored, budget, operations = saved_definition_state(campaign)
    assert restored is docs and budget is None and operations == ()
    assert "definition:operation:7" in reads
    assert "definition:operation:8" not in reads


def test_closure_storage_failure_preserves_original_guard_error(tmp_path, monkeypatch):
    from packages.multimodal_research import stage_failure

    campaign = campaign_at(tmp_path, monkeypatch, 240)
    campaign.campaign_id = "closure-primary"
    campaign.base.manifest = SimpleNamespace(checksum="test-input")
    campaign.base.task = SimpleNamespace(target_code="GOAL:TEST")
    (tmp_path / "cancel").touch()

    def unavailable(*args, **kwargs):
        raise OSError("local diagnostic publication unavailable")

    monkeypatch.setattr(stage_failure, "publish_json", unavailable)
    monkeypatch.setattr(stages, "credentials", lambda *a: pytest.fail("no credentials"))
    with pytest.raises(ValueError, match="^cancelled$") as caught:
        stages.run(campaign, "review")
    assert isinstance(caught.value.__cause__, OSError)
    assert not (tmp_path / "stage-failure.json").exists()
