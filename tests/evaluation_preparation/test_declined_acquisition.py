"""A confirmed refusal may have one explicit successor, not a renewed deadline."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign_renewal import (
    RenewedCampaign,
    declined_acquisition_tip,
)
from tests.agent_provider.support import Documents


def fixture(path):
    state, manifest = {"state": "current"}, {"accepted": "original"}
    task = {"code": "GOAL:TEST"}
    protocol = dict(
        schema_version="multimodal-campaign-renewal/1.17.0",
        mode="acquisition",
        acquisition_increment_only=True,
        facts=str(path / "facts"),
        accepted_state_checksum=digest(state),
        accepted_origin_checksum=digest(manifest),
        input=digest("input"),
        task=digest(task),
        max_new_requests=8,
        deadline=(datetime.now(UTC) + timedelta(seconds=3600)).isoformat(),
        previous_attempts=[],
        acquisition_targets=["new1", "new2", "new3"],
    )
    phase = dict(target="new1", run_id="RUN:DECLINED", accepted=False, acceptance_key=None)
    result = dict(
        case_code="CASE:TEST",
        task=task,
        complete=False,
        previous_attempts=[],
        new_requests=1,
        phases=[phase],
    )
    record = dict(
        error="rate_limit",
        role="vision",
        simulated=False,
        api_requests=1,
        retry_index=0,
        output_checksum=None,
        request_id="request:declined",
        scope=dict(run_id="RUN:DECLINED", case_code="CASE:TEST", input_checksum=digest("input")),
    )
    events = [
        dict(phase="reserved", request_id=record["request_id"]),
        dict(phase="completed", record=record),
        dict(phase="response_confirmation", request_id=record["request_id"], confirmed=True),
        dict(
            schema_version="agent-transport-diagnostic/1.0.0",
            request_id=record["request_id"],
            value=dict(http_status=429, category="unclassified_429"),
        ),
    ]
    docs = Documents()
    docs.run_id = phase["run_id"]
    control = SimpleNamespace(phase="stopped", cancelled=False, pending=None)
    parent = SimpleNamespace(
        aggregation_baseline=SimpleNamespace(path=path / "facts", state=state, manifest=manifest),
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest(task), model_dump=lambda **_: task),
            binding=dict(case_code="CASE:TEST"),
        ),
        documents=lambda _: docs,
        sessions=SimpleNamespace(read=lambda _: control),
    )
    return SimpleNamespace(
        parent=parent,
        protocol=protocol,
        result=result,
        events=events,
        docs=docs,
        control=control,
        path=path,
    )


def persist(f):
    for key, value in (
        ("dynamic-campaign", f.protocol),
        ("dynamic-phase-result", f.result["phases"][0]),
        ("multimodal-case-result", f.result),
    ):
        f.docs.write(key, (value,))
    for index, event in enumerate(f.events):
        f.docs.write("journal:" + str(index), (event,))


def test_declined_successor_keeps_budget_deadline_history_and_has_no_writes(tmp_path):
    f = fixture(tmp_path)
    persist(f)
    before = digest(f.docs.saved), len(f.docs.audit)
    tip = declined_acquisition_tip(f.parent, f.path, f.protocol, f.result)
    assert tip["deadline"] == f.protocol["deadline"]
    assert tip["declined_acquisition"]["remaining_requests"] == 7
    assert tip["previous_attempts"][-1]["requests"] == 1
    assert tip["declined_acquisition"]["targets"] == ["new1", "new2", "new3"]
    assert not tip["declined_acquisition"]["further_decline_successors_allowed"]
    assert declined_acquisition_tip(f.parent, f.path, f.protocol, f.result) == tip
    assert (digest(f.docs.saved), len(f.docs.audit)) == before


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "incomplete",
        "timeout",
        "wrong_run",
        "quota",
        "cancelled",
        "accepted",
        "second_successor",
        "wrong_input",
    ],
)
def test_declined_successor_rejects_other_states_without_write(tmp_path, bad):
    f = fixture(tmp_path)
    if bad == "unknown":
        f.events[2]["confirmed"] = None
    elif bad == "incomplete":
        f.events.pop(1)
    elif bad == "timeout":
        f.events[1]["record"]["error"] = "timeout"
    elif bad == "wrong_run":
        f.events[1]["record"]["scope"]["run_id"] = "RUN:OTHER"
    elif bad == "quota":
        f.events[3]["value"]["category"] = "insufficient_quota"
    elif bad == "cancelled":
        f.control.cancelled = True
    elif bad == "accepted":
        f.result["phases"][0]["accepted"] = True
    elif bad == "second_successor":
        f.protocol["schema_version"] = "multimodal-campaign-renewal/1.18.0"
    elif bad == "wrong_input":
        f.protocol["input"] = digest("different")
    persist(f)
    before = digest(f.docs.saved), len(f.docs.audit)
    with pytest.raises(ValueError, match="renewal_declined_acquisition_"):
        declined_acquisition_tip(f.parent, f.path, f.protocol, f.result)
    assert (digest(f.docs.saved), len(f.docs.audit)) == before


def test_successor_uses_shortened_original_window_and_preserves_closure_headroom(monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign

    observed = []

    def budget(_self, **kwargs):
        assert kwargs == dict(bounded_phase=True)
        return SimpleNamespace(before_reserve=lambda value: observed.append(value))

    monkeypatch.setattr(MultimodalCampaign, "budget_factory", budget)
    campaign = RenewedCampaign.__new__(RenewedCampaign)
    campaign.protocol = dict(
        schema_version="multimodal-campaign-renewal/1.18.0", max_model_dispatches=6
    )
    campaign.usage = lambda: 5
    b = campaign.budget_factory()
    b.before_reserve("last-permitted")
    campaign.usage = lambda: 6
    with pytest.raises(ProviderError, match="budget"):
        b.before_reserve("cannot-spend-closure")
    assert observed == ["last-permitted"]
