"""Exact rejected text request permits one changed contract, not a retry cascade."""

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.acquired_review import contract_repair_tip
from packages.multimodal_research.campaign import configuration
from tests.evaluation_preparation.test_acquired_review import Observation
from tests.evaluation_preparation.test_declined_acquisition import fixture, persist


def setup(path, monkeypatch):
    f = fixture(path)
    observations = {"new1": Observation()}
    f.parent.config = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=True,
    )
    f.parent.parent = object()
    f.parent.accepted_observations = lambda: observations
    original = dict(
        path=str(path / "original"),
        acquired_review=dict(
            selected_ids=["new1"],
            remaining_requests=5,
            deadline=f.protocol["deadline"],
            historical_review_ids=["old"],
        ),
    )
    f.protocol.update(
        schema_version="multimodal-campaign-renewal/1.19.0",
        review_acquired_only=True,
        tip=original,
        review_target_ids=["new1"],
        confirmed_observations_checksum=digest(observations),
    )
    f.result["phases"][0]["target"] = None
    f.events[1]["record"].update(role="review", error="configuration")
    f.events[-1]["value"] = dict(http_status=400, category="request_rejected")
    monkeypatch.setattr(
        "packages.multimodal_research.campaign_renewal.stopped_tip", lambda *a, **k: original
    )
    return f


def test_one_contract_successor_preserves_original_deadline_and_cumulative_charge(
    tmp_path, monkeypatch
):
    f = setup(tmp_path, monkeypatch)
    persist(f)
    before = digest(f.docs.saved), len(f.docs.audit)
    tip = contract_repair_tip(f.parent, tmp_path, f.protocol, f.result)
    assert tip["deadline"] == f.protocol["deadline"]
    assert tip["acquired_review"]["remaining_requests"] == 4
    assert tip["acquired_review"]["selected_ids"] == ["new1"]
    assert tip["previous_attempts"][-1]["requests"] == 1
    assert tip["contract_repair"]["http_400_cause_not_proven"]
    assert not tip["contract_repair"]["further_contract_successors_allowed"]
    assert (digest(f.docs.saved), len(f.docs.audit)) == before


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "cancelled",
        "different_request",
        "wrong_input",
        "retry",
        "accepted",
        "second",
        "changed_observation",
        "not400",
    ],
)
def test_invalid_attempt_cannot_become_contract_repair(tmp_path, monkeypatch, bad):
    f = setup(tmp_path, monkeypatch)
    if bad == "unknown":
        f.events[2]["confirmed"] = None
    elif bad == "cancelled":
        f.control.cancelled = True
    elif bad == "different_request":
        f.events[1]["record"]["request_id"] = "request:other"
    elif bad == "wrong_input":
        f.protocol["input"] = digest("other")
    elif bad == "retry":
        f.events[1]["record"]["retry_index"] = 1
    elif bad == "accepted":
        f.result["phases"][0]["accepted"] = True
    elif bad == "second":
        f.protocol["schema_version"] = "multimodal-campaign-renewal/1.20.0"
    elif bad == "changed_observation":
        f.protocol["confirmed_observations_checksum"] = digest("changed")
    elif bad == "not400":
        f.events[-1]["value"]["http_status"] = 429
    persist(f)
    with pytest.raises(ValueError, match="acquired_contract_repair_"):
        contract_repair_tip(f.parent, tmp_path, f.protocol, f.result)
