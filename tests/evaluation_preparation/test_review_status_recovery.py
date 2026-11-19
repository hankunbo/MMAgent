"""Only confirmed, original-owner rejected status errors permit this one successor."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.review_status_recovery import check_failure
from tests.evaluation_preparation.test_directional_review import explanation, fixture, reply


def prepared(tmp_path, monkeypatch):
    import json

    from packages.evidence_details.review_batches import batch_payload, make_plan
    from packages.multimodal_research import review_status_recovery
    from packages.object_store import checksum

    f = fixture()
    state = f.owner.current()
    plan = make_plan(state, f.config.review_batching, key="event:A")
    payload = batch_payload(state, plan, plan.batches[0])
    raw = reply(dict(messages=[{}, dict(content=[dict(text=json.dumps(payload))])]))
    item = next(i for i in raw["items"] if i["status"] == "assessed")
    item["status"] = "observation_ungradable"
    item["judgment"].update(
        value=None,
        lower=None,
        upper=None,
        method="not_assessed",
        missing_reason="Layer cannot be resolved",
    )
    item["direction_assessment"] = explanation(None)
    item["direction_assessment"]["schema_version"] = "direction-assessment/1.0.0"
    text = json.dumps(raw)
    record = dict(
        request_id="request:test",
        role="review",
        error="scope",
        api_requests=1,
        simulated=False,
        input_checksum=digest(payload),
        config_checksum=digest(f.config),
    )
    response = dict(
        raw_model_text=text,
        raw_checksum=checksum(text.encode()),
        provider_config=f.config.model_dump(mode="json"),
        record_checksum=digest(record),
        finish_reason="stop",
    )
    events = [
        dict(phase="reserved", request_id="request:test"),
        dict(phase="completed", record=record),
        dict(phase="response_confirmation", request_id="request:test", confirmed=True),
    ]
    result = dict(complete=False, phases=[dict(target=None, accepted=False, run_id="run:test")])
    values = {
        "live-stage-intent:review": (dict(event=dict(code="event")),),
        "review-batches:event:A:plan": (plan.model_dump(mode="json"),),
        "review-batches:event:A:batch:000:attempt:request:test": (response,),
        "dynamic-phase-input": (state.model_dump(mode="json"),),
        "multimodal-case-result": (result,),
    }
    docs = SimpleNamespace(read=values.get)
    # digest requires serializable domain values, not a raw SimpleNamespace.

    class Control(dict):
        __getattr__ = dict.__getitem__

    control = Control(phase="stopped", pending=None)
    campaign = SimpleNamespace(
        documents=lambda _: docs, sessions=SimpleNamespace(read=lambda _: control), output=tmp_path
    )
    monkeypatch.setattr(
        review_status_recovery,
        "RuntimeAgentJournal",
        lambda _: SimpleNamespace(read=lambda: events),
    )
    return campaign, result, events


def test_only_confirmed_status_conflict_can_continue(tmp_path, monkeypatch):
    campaign, result, events = prepared(tmp_path, monkeypatch)
    assert check_failure(campaign, result)["result"] == digest(result)
    events[-1]["confirmed"] = None
    with pytest.raises(ValueError, match="confirmed_failure_required"):
        check_failure(campaign, result)


def test_cancel_marker_blocks_successor(tmp_path, monkeypatch):
    campaign, result, _ = prepared(tmp_path, monkeypatch)
    (tmp_path / "cancel").touch()
    with pytest.raises(ValueError, match="confirmed_failure_required"):
        check_failure(campaign, result)


@pytest.mark.parametrize("change", ["transport", "simulated", "unknown"])
def test_other_failure_or_unknown_cannot_use_status_successor(tmp_path, monkeypatch, change):
    campaign, result, events = prepared(tmp_path, monkeypatch)
    if change == "transport":
        events[1]["record"]["error"] = "transport"
    elif change == "simulated":
        events[1]["record"]["simulated"] = True
    else:
        events.pop()
    with pytest.raises(ValueError, match="confirmed_failure_required"):
        check_failure(campaign, result)


def test_successor_three_request_cap_includes_reserved_dispatch(monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.review_status_recovery import StatusReviewCampaign

    calls = []
    monkeypatch.setattr(MultimodalCampaign, "guard", lambda self, **kw: calls.append(kw))
    c = object.__new__(StatusReviewCampaign)
    c.usage = lambda: 3
    with pytest.raises(ProviderError, match="budget"):
        c.guard()
    c.guard(dispatch=True)  # The third already-reserved request, not a fourth reservation.
    assert calls == [dict(dispatch=True)]
    c.usage = lambda: 4
    with pytest.raises(ProviderError, match="budget"):
        c.guard(dispatch=True)


def test_time_gate_includes_observed_local_owner_cost_and_final_close():
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.review_status_recovery import check_window, required_window

    assert required_window(5, 3) == 1500
    now = datetime.now(UTC)
    # The actual failed successor had less than this; a ten-minute window is not enough.
    with pytest.raises(ValueError, match="insufficient_original_time"):
        check_window((now + timedelta(minutes=10)).isoformat(), 5, 3)
    assert check_window((now + timedelta(minutes=26)).isoformat(), 5, 3) == 1500
    with pytest.raises(ValueError, match="insufficient_original_time"):
        check_window((now - timedelta(seconds=1)).isoformat(), 5, 3)
