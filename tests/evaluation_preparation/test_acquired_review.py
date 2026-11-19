"""Terminal original facts, no acquisition retry, no implicit new time window."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import Value, digest
from packages.multimodal_research.acquired_review import bounds, terminal_inventory
from packages.multimodal_research.campaign_renewal import RenewedCampaign
from tests.agent_provider.support import Documents
from tests.evaluation_preparation.test_declined_acquisition import fixture


class Observation(Value):
    status: str = "obtained"


def setup(path, bad=None):
    f = fixture(path)
    f.protocol.update(
        schema_version="multimodal-campaign-renewal/1.18.0",
        reuse_assessment_ids=["old"],
        max_new_requests=7,
    )
    accepted = dict(target="new1", run_id="RUN:ACQUIRED", accepted=True, acceptance_key="accepted")
    f.result["phases"][0]["target"] = "new2"
    f.result["phases"].insert(0, accepted)
    f.result["new_requests"] = 2
    good = deepcopy(f.events[:3])
    for event in good:
        if "request_id" in event:
            event["request_id"] = "request:accepted"
    good[1]["record"].update(
        request_id="request:accepted", error=None, output_checksum=digest("body")
    )
    good[1]["record"]["scope"]["run_id"] = "RUN:ACQUIRED"
    values = {"new1": Observation()}
    if bad == "unknown":
        f.events[2]["confirmed"] = None
    elif bad == "cancelled":
        f.control.cancelled = True
    elif bad == "pending":
        f.control.pending = "unresolved"
    elif bad == "input":
        f.protocol["input"] = digest("other")
    elif bad == "scope":
        good[1]["record"]["scope"]["run_id"] = "RUN:OTHER"
    elif bad == "count":
        f.result["new_requests"] = 1
    elif bad == "old":
        f.protocol["reuse_assessment_ids"] = ["new1"]
    elif bad == "observation":
        values["new1"] = Observation(status="unknown")
    elif bad == "receipt":
        accepted["acceptance_key"] = None
    elif bad == "timeout":
        f.events[1]["record"]["error"] = "timeout"
    elif bad == "new_run":
        f.protocol["schema_version"] = "multimodal-campaign-renewal/1.19.0"
    stores = {}
    for p, events in zip(f.result["phases"], (good, f.events), strict=True):
        d = Documents()
        d.run_id = p["run_id"]
        d.write("dynamic-campaign", (f.protocol,))
        d.write("dynamic-phase-result", (p,))
        d.write("multimodal-case-result", (f.result,))
        for index, event in enumerate(events):
            d.write("journal:" + str(index), (event,))
        stores[d.run_id] = d
    f.parent.documents = stores.__getitem__
    f.parent.accepted_observations = lambda: values
    f.stores = stores
    return f


def test_read_only_inventory_is_new_observations_only_and_keeps_accounting(tmp_path):
    f = setup(tmp_path)
    before = {k: (digest(d.saved), len(d.audit)) for k, d in f.stores.items()}
    tip = terminal_inventory(f.parent, tmp_path, f.protocol, f.result)
    assert tip["acquired_review"]["selected_ids"] == ["new1"]
    assert tip["acquired_review"]["historical_review_ids"] == ["old"]
    assert tip["acquired_review"]["remaining_requests"] == 5
    assert tip["deadline"] == f.protocol["deadline"]
    assert tip["previous_attempts"][-1]["requests"] == 2
    assert terminal_inventory(f.parent, tmp_path, f.protocol, f.result) == tip
    assert before == {k: (digest(d.saved), len(d.audit)) for k, d in f.stores.items()}


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "cancelled",
        "pending",
        "input",
        "scope",
        "count",
        "old",
        "observation",
        "receipt",
        "timeout",
        "new_run",
    ],
)
def test_invalid_terminal_or_original_binding_cannot_authorize_text_review(tmp_path, bad):
    f = setup(tmp_path, bad)
    with pytest.raises(ValueError, match="acquired_review_"):
        terminal_inventory(f.parent, tmp_path, f.protocol, f.result)


@pytest.mark.parametrize(
    "seconds,remaining,retries,valid",
    [
        (3600, 5, 0, True),
        (299, 5, 0, False),
        (-1, 5, 0, False),
        (3600, 1, 0, False),
        (3600, 5, 1, False),
    ],
)
def test_no_new_window_retry_or_local_closure_budget_spend(seconds, remaining, retries, valid):
    now = datetime.now(UTC)
    end = now + timedelta(seconds=seconds)
    tip = dict(
        acquired_review=dict(
            deadline=end.isoformat(), remaining_requests=remaining, selected_ids=["new"]
        )
    )
    config = SimpleNamespace(
        review_batching=SimpleNamespace(batch_size=1),
        transport=SimpleNamespace(max_retries=retries),
    )
    if not valid:
        with pytest.raises(ValueError, match="acquired_review_"):
            bounds(tip, config, 2, now)
    else:
        assert bounds(tip, config, 2, now) == (end, seconds, 2, 1)


def test_dispatch_cap_is_one_text_request_not_two_with_closure_headroom(monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign

    calls = []

    def factory(_self, **kw):
        assert kw == dict(bounded_phase=True)
        return SimpleNamespace(before_reserve=lambda sha: calls.append(sha))

    monkeypatch.setattr(MultimodalCampaign, "budget_factory", factory)
    c = RenewedCampaign.__new__(RenewedCampaign)
    c.protocol = dict(schema_version="multimodal-campaign-renewal/1.19.0", max_model_dispatches=1)
    c.usage = lambda: 0
    b = c.budget_factory()
    b.before_reserve("first")
    c.usage = lambda: 1
    with pytest.raises(ProviderError, match="budget"):
        b.before_reserve("repeat")
    assert calls == ["first"]
