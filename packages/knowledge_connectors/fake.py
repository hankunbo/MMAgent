"""Deterministic fixtures replace only clock, DNS and byte transport, not adapter logic."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field

from .configuration import ProviderConfig
from .connectors import CONNECTOR_TYPES
from .http import SafeHTTPClient, WireRequest, WireResponse
from .ports import (
    Capabilities,
    ErrorCategory,
    FetchedDocument,
    HealthResult,
    SearchPage,
    SourceLocator,
    StrictModel,
    StructuredQuery,
    Text,
    Version,
    checked,
)


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def now(self) -> datetime:
        return datetime(2026, 9, 5, tzinfo=UTC) + timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class FixtureExchange(StrictModel):
    host: Text
    path: Text
    query: tuple[tuple[str, str], ...] = ()
    status: int = Field(default=200, ge=100, le=599)
    headers: tuple[tuple[str, str], ...] = (("content-type", "application/json"),)
    body_text: str = Field(default="", max_length=4 * 1024 * 1024, repr=False)
    error: ErrorCategory | None = None


class FixtureTape(StrictModel):
    fixture_version: Version
    synthetic: bool
    exchanges: tuple[FixtureExchange, ...] = Field(max_length=100)


class RecordedTransport:
    # This transport consumes local bytes only and never falls back to live HTTP.
    contact_mode = "synthetic"

    def __init__(self, exchanges: tuple[FixtureExchange, ...]):
        self._exchanges = deque(checked(FixtureExchange, exchange) for exchange in exchanges)
        self.calls = 0

    def send(self, request: WireRequest, ip: str, max_bytes: int) -> WireResponse:
        self.calls += 1
        if not self._exchanges:
            raise AssertionError("fixture tape exhausted; no live fallback")
        exchange = self._exchanges.popleft()
        parsed = urlsplit(request.url)
        if (
            parsed.hostname != exchange.host
            or parsed.path != exchange.path
            or sorted(parse_qsl(parsed.query)) != sorted(exchange.query)
        ):
            raise AssertionError("request does not match the fixture tape")
        if exchange.error is ErrorCategory.TIMEOUT:
            raise TimeoutError
        if exchange.error is ErrorCategory.NETWORK:
            raise OSError("synthetic transport failure")
        if exchange.error is not None:
            raise ValueError("fixture transport supports only network/timeout failure injection")
        return WireResponse(exchange.status, exchange.headers, exchange.body_text.encode())

    def assert_consumed(self) -> None:
        if self._exchanges:
            raise AssertionError("unused fixture exchanges")


def load_fixture(path: Path) -> FixtureTape:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("fixture tape exceeds size bound")
    return FixtureTape.model_validate_json(path.read_bytes())


class DeterministicFakeConnector:
    def __init__(self, config: ProviderConfig, exchanges: tuple[FixtureExchange, ...]):
        self.transport = RecordedTransport(exchanges)
        self.clock = FakeClock()
        client = SafeHTTPClient(
            config,
            self.transport,
            resolver=lambda host: ("93.184.216.34",),
            clock=self.clock,
            environment=lambda name: (
                "fixture@example.invalid" if name == config.contact_email_env else None
            ),
        )
        self._connector = CONNECTOR_TYPES[config.adapter](config, client)

    def search(self, query: StructuredQuery) -> SearchPage:
        return self._connector.search(query)

    def fetch(self, locator: SourceLocator) -> FetchedDocument:
        return self._connector.fetch(locator)

    def healthcheck(self) -> HealthResult:
        return self._connector.healthcheck()

    def capabilities(self) -> Capabilities:
        return self._connector.capabilities()
