import threading
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
from packages.knowledge_connectors import ConnectorError, FakeClock, SafeHTTPClient
from packages.knowledge_search.policy import (
    ProviderGovernor,
    RunBudget,
    SearchStopped,
    SourceBudget,
)
from tests.knowledge_connectors.test_http_04b1 import ScriptedTransport, response
from tests.knowledge_search.support import config, policy


class Expansion:
    def __init__(self, clock, delay, deadline):
        self.clock, self.delay, self.absolute = clock, delay, deadline
        self.policy = SimpleNamespace(max_search_requests=3)
        self.campaign = SimpleNamespace(
            protocol={"deadline": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}
        )
        self.calls = []
        self.saved = []
        self.cancelled = False
        self.documents = SimpleNamespace(
            session=lambda: nullcontext(),
            write=lambda key, values: self.saved.append((key, values)),
        )

    def fast_control(self):
        from packages.llm_provider.provider import ProviderError

        if self.cancelled:
            raise ProviderError("cancelled")
        if self.clock.monotonic() >= self.absolute:
            raise ProviderError("deadline")

    def control(self):
        self.fast_control()

    def reserve_wire(self, provider, *, kind):
        self.fast_control()
        self.calls.append((provider, kind))
        self.clock.sleep(self.delay)
        return "wire:" + str(len(self.calls))


def build(delay=55, absolute=120, responses=None):
    cfg = config()
    clock = FakeClock()
    expansion = Expansion(clock, delay, absolute)
    guard = LiteratureDefinitionConnector(None, expansion, {}).governor(kind="search")
    governor = ProviderGovernor((cfg,), policy(), clock, request_guard=guard)
    transport = ScriptedTransport(*(responses or [response()]))
    client = SafeHTTPClient(
        cfg, transport, clock=clock, governor=governor, resolver=lambda host: ("93.184.216.34",)
    )
    budget = SourceBudget(RunBudget(absolute, 3, 10, threading.Event()), absolute, 3, max_retries=0)
    return client, transport, expansion, governor, budget


def request(client):
    return client.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=synthetic",
        operation="search",
        expected_mime=("application/json",),
    )


def test_slow_original_admission_leaves_bounded_network_time():
    client, transport, e, g, b = build()
    with g.bind(b):
        result = request(client)
    assert result.metadata.elapsed_ms >= 55000
    assert len(transport.requests) == len(e.calls) == b.requests == 1
    assert 0 < transport.requests[0][0].timeout_seconds <= 10
    assert len(e.saved) == 1 and e.saved[0][1][0].http_status == 200


def test_absolute_search_deadline_still_stops_before_transport():
    client, transport, e, g, b = build(55, 40)
    with g.bind(b), pytest.raises((SearchStopped, ConnectorError)):
        request(client)
    assert not transport.requests


def test_failed_admission_cannot_become_dispatch():
    client, transport, e, g, b = build()
    e.cancelled = True
    with g.bind(b), pytest.raises(SearchStopped):
        request(client)
    assert not transport.requests and not e.calls


def test_redirect_rechecks_owner_and_keeps_one_network_budget():
    client, transport, e, g, b = build(
        40, 120, [response(302, headers=(("location", "/next"),)), response()]
    )
    with g.bind(b):
        result = request(client)
    assert len(e.calls) == len(transport.requests) == b.requests == 2
    assert len(e.saved) == 2 and result.metadata.elapsed_ms >= 80000
    assert all(0 < r[0].timeout_seconds <= 10 for r in transport.requests)


def test_owner_delay_does_not_hide_cumulative_transport_deadline():
    client, transport, e, g, b = build(
        40, 200, [response(302, headers=(("location", "/next"),)), response()]
    )
    send = transport.send

    def delayed(*args):
        result = send(*args)
        client.clock.sleep(16)
        return result

    transport.send = delayed
    with g.bind(b), pytest.raises(ConnectorError):
        request(client)
    assert len(e.calls) == len(transport.requests) == 2
    # 80s owner work + 32s network exceeds the unchanged 30s network window.
    assert client.clock.monotonic() >= 112
    assert len(e.saved) == 2


def test_rate_wait_is_not_credited_as_local_owner_work():
    client, transport, e, g, b = build(5, 200)
    g.states[client.config.provider_code].next_at = 40
    with g.bind(b), pytest.raises(ConnectorError):
        request(client)
    assert len(e.calls) == 1 and not transport.requests


def test_cancel_during_owner_gate_stops_without_transport():
    client, transport, e, g, b = build()
    original = e.reserve_wire

    def cancel(*args, **kwargs):
        key = original(*args, **kwargs)
        e.cancelled = True
        return key

    e.reserve_wire = cancel
    with g.bind(b), pytest.raises(SearchStopped):
        request(client)
    assert len(e.calls) == 1 and not transport.requests


def test_exhausted_budget_does_not_reserve_owner_wire():
    client, transport, e, g, b = build()
    b.requests = b.max_requests
    with g.bind(b), pytest.raises(SearchStopped):
        request(client)
    assert not transport.requests and not e.calls


def test_direct_fetch_governor_uses_same_preparation_and_absolute_limit():
    client, transport, e, g, b = build()
    # Full-text retrieval also consumes the original direct definition governor.
    client.governor = g.request_guard
    result = request(client)
    assert result.metadata.elapsed_ms >= 55000
    assert len(transport.requests) == len(e.calls) == len(e.saved) == 1
    assert client.governor.prepared_provider is None


def test_confirmed_wire_owner_storage_does_not_erase_valid_response():
    client, transport, e, g, b = build(5, 160)

    def persist(key, values):
        client.clock.sleep(55)
        e.saved.append((key, values))

    e.documents.write = persist
    with g.bind(b):
        result = request(client)
    assert len(transport.requests) == len(e.calls) == len(e.saved) == 1
    assert result.metadata.elapsed_ms >= 60000


def test_receipt_storage_cannot_extend_absolute_campaign():
    client, transport, e, g, b = build(5, 40)

    def persist(key, values):
        client.clock.sleep(55)
        e.saved.append((key, values))

    e.documents.write = persist
    with g.bind(b), pytest.raises(SearchStopped, match="timeout"):
        request(client)
    # Original confirmed outcome remains saved even though the absolute run expired.
    assert len(transport.requests) == len(e.calls) == len(e.saved) == 1
