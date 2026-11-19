"""Capacity recovery keeps original cancel/deadline/budget and zero-Vision scope."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.multimodal_research.review_capacity_recovery import origin_proof, plan_for


def test_factory_installs_capacity_reader_on_private_base_and_original_config(
    tmp_path, monkeypatch
):
    from packages.multimodal_research import campaign_renewal as renewal
    from packages.multimodal_research import review_capacity_recovery as recovery
    from packages.multimodal_research.campaign import MultimodalCampaign

    config = renewal.confirmed_facts_configuration()
    previous = SimpleNamespace()
    parent = SimpleNamespace(
        base=SimpleNamespace(review_target_ids=("old",)),
        parent=previous,
        target=None,
        config=config,
        protocol=dict(tip=dict(run_id="accepted")),
    )
    plan = dict(
        schema_version=renewal.CAPACITY_REVIEW_VERSION,
        mode="review",
        review_target_ids=["old", "new"],
        reuse_assessment_ids=["old"],
        review_capacity=dict(required_batches=18),
    )
    calls = []
    reader = object()
    monkeypatch.setattr(recovery, "check", lambda p, v: calls.append((p, v)))

    def installed(p, rid, **options):
        assert p is previous and rid == "accepted"
        assert options["capacity_config"] == recovery.configuration(config, 18)
        return reader

    monkeypatch.setattr(renewal, "accepted_reader", installed)
    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self, base, *_a, **_k: setattr(self, "base", base)
    )
    value = renewal.RenewedCampaign(parent, tmp_path, plan)
    assert calls == [(parent, plan)]
    assert value.base is not parent.base
    assert parent.base.review_target_ids == ("old",)
    assert value.base.confirmed_review_reader is reader
    assert value.config == recovery.configuration(config, 18)
    assert value.targets == value.initial_targets == ()


@pytest.mark.parametrize("fault", ["unknown", "accepted", "key", "run", "observation", "version"])
def test_invalid_terminal_origin_rejected_before_owner_reads(fault):
    def forbidden():
        pytest.fail("invalid origin reached owner")

    parent = SimpleNamespace(
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.37.0",
            mode="acquisition",
            acquisition_targets=["new"],
            previous_attempts=[],
        ),
        config=SimpleNamespace(transport=SimpleNamespace(max_retries=0)),
        base=SimpleNamespace(check=forbidden),
    )
    phase = dict(
        target=None, accepted=False, acceptance_key=None, error="batch_partition", run_id="review"
    )
    result = dict(
        complete=False, previous_attempts=[], phases=[dict(target="new", accepted=True), phase]
    )
    if fault == "unknown":
        phase["error"] = "timeout"
    elif fault == "accepted":
        phase["accepted"] = True
    elif fault == "key":
        phase["acceptance_key"] = "wrong"
    elif fault == "run":
        phase["run_id"] = "wrong"
    elif fault == "observation":
        result["phases"][0]["accepted"] = False
    else:
        parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.26.0"
    with pytest.raises(ValueError, match="review_capacity_original_failure"):
        origin_proof(parent, result, dict(run_id="review"))


def test_exact_budget_no_new_dispatch_after_three_but_local_close_allowed(tmp_path, monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from tests.evaluation_preparation.test_exact_acquisition_budget import campaign

    value, calls = campaign(tmp_path, version="1.43.0", used=3, previous=34)
    value.protocol["mode"] = "review"
    value.guard()
    assert calls == ["current-source"]
    monkeypatch.setattr(
        MultimodalCampaign,
        "budget_factory",
        lambda *_a, **_k: SimpleNamespace(before_reserve=lambda _: None),
    )
    with pytest.raises(ProviderError, match="budget"):
        value.budget_factory().before_reserve("extra")


def test_plan_inherits_deadline_and_only_three_unspent_requests(tmp_path):
    from packages.multimodal_research.campaign_renewal import confirmed_facts_configuration

    now = datetime.now(UTC)
    parent = SimpleNamespace(
        protocol=dict(deadline=(now + timedelta(seconds=1200)).isoformat()),
        config=confirmed_facts_configuration(),
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum="input"), task=SimpleNamespace(checksum="task")
        ),
        accepted_observations=lambda: {},
    )
    proof = dict(
        required_batches=18,
        new_dispatches=3,
        review_target_ids=["new"],
        reuse_assessment_ids=["old"],
    )
    tip = dict(path=str(tmp_path), previous_attempts=[dict(requests=34)])
    plan = plan_for(
        parent, {}, tip, proof, output=tmp_path / "new", authorization="bounded", started=now
    )
    assert plan["deadline"] == parent.protocol["deadline"]
    assert plan["max_new_requests"] == plan["max_model_dispatches"] == 3
    assert plan["new_vision_requests"] == 0 and plan["acquisition_targets"] == []
    with pytest.raises(ValueError, match="remaining_time"):
        plan_for(
            parent,
            {},
            tip,
            proof,
            output=tmp_path / "late",
            authorization="bounded",
            started=now + timedelta(seconds=1000),
        )
