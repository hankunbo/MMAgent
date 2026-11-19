"""Provider-wide admission runs before identity resolution, once per wire attempt."""

from __future__ import annotations

import math
import threading
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from packages.knowledge_connectors import ProviderConfig
from packages.knowledge_connectors.http import Clock, SystemClock
from packages.knowledge_connectors.ports import StrictModel, Version, checked


class RatePolicy(StrictModel):
    policy_version: Version
    deployment_workers: int = Field(ge=1, le=1)
    ncbi_without_key_per_second: float = Field(gt=0)
    ncbi_with_key_per_second: float = Field(gt=0)
    crossref_pool: Literal["public", "polite"]
    crossref_public_per_second: float = Field(gt=0)
    crossref_polite_per_second: float = Field(gt=0)
    crossref_max_concurrency: int = Field(ge=1, le=1)
    penalty_factor: float = Field(ge=1)
    default_cooldown_ms: int = Field(ge=1, le=300000)
    max_header_delay_ms: int = Field(ge=1000, le=86400000)
    poll_ms: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def finite_numbers(self):
        for v in (
            self.ncbi_without_key_per_second,
            self.ncbi_with_key_per_second,
            self.crossref_public_per_second,
            self.crossref_polite_per_second,
            self.penalty_factor,
        ):
            if not math.isfinite(v):
                raise ValueError("finite policy value required")
        return self


def load_rate_policy(path: Path) -> RatePolicy:
    if path.stat().st_size > 16384:
        raise ValueError("policy file exceeds bound")
    return RatePolicy.model_validate(tomllib.loads(path.read_text()))


class SearchStopped(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass
class RunBudget:
    deadline: float
    max_requests: int
    max_hits: int
    cancel: threading.Event
    requests: int = 0
    hits: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class SourceBudget:
    run: RunBudget
    deadline: float
    max_requests: int
    max_retries: int = 2
    requests: int = 0
    metadata: list = field(default_factory=list)
    wire_receipts: list = field(default_factory=list)


@dataclass
class RateState:
    interval: float
    next_at: float = 0.0
    blocked: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ProviderGovernor:
    def __init__(
        self,
        configs: tuple[ProviderConfig, ...],
        policy: RatePolicy,
        clock: Clock | None = None,
        *,
        request_guard=None,
    ):
        self.policy = checked(RatePolicy, policy)
        self.clock = clock or SystemClock()
        self.request_guard = request_guard
        self.local = threading.local()
        self.states = {}
        for config in configs:
            cap = config.http.rate_limit_per_minute / 60
            if config.adapter == "pubmed":
                cap = min(
                    cap,
                    policy.ncbi_with_key_per_second
                    if config.api_key_env
                    else policy.ncbi_without_key_per_second,
                )
            if config.adapter == "crossref":
                if policy.crossref_pool == "polite" and config.contact_email_env is None:
                    raise ValueError("polite pool requires configured contact environment")
                cap = min(
                    cap,
                    policy.crossref_polite_per_second
                    if policy.crossref_pool == "polite"
                    else policy.crossref_public_per_second,
                )
            self.states[config.provider_code] = RateState(interval=1 / cap)

    @contextmanager
    def bind(self, source: SourceBudget):
        if getattr(self.local, "source", None) is not None:
            raise RuntimeError("nested budget context forbidden")
        self.local.source = source
        try:
            yield
        finally:
            del self.local.source

    def checkpoint(self):
        if self.request_guard is not None and self.request_guard.remaining() <= 0:
            raise SearchStopped("timeout")
        source = self.local.source
        if source.run.cancel.is_set():
            raise SearchStopped("cancelled")
        if self.clock.monotonic() >= min(source.deadline, source.run.deadline):
            raise SearchStopped("timeout")

    def retry_limit(self, configured_limit: int) -> int:
        limit = min(configured_limit, self.local.source.max_retries)
        return self.request_guard.retry_limit(limit) if self.request_guard is not None else limit

    def remaining(self) -> float:
        self.checkpoint()
        source = self.local.source
        seconds = min(source.deadline, source.run.deadline) - self.clock.monotonic()
        return (
            min(seconds, self.request_guard.remaining())
            if self.request_guard is not None
            else seconds
        )

    def wait(self, seconds: float) -> None:
        until = self.clock.monotonic() + seconds
        while self.clock.monotonic() < until:
            self.checkpoint()
            self.clock.sleep(min(until - self.clock.monotonic(), self.policy.poll_ms / 1000))
        self.checkpoint()

    def completed_wire(self, receipt) -> None:
        self.local.source.wire_receipts.append(receipt)
        if self.request_guard is not None:
            self.request_guard.completed_wire(receipt)

    def completed_request(self, metadata) -> None:
        self.local.source.metadata.append(metadata)

    def prepare_send(self, provider_code: str) -> None:
        """Run the explicit external owner gate before the connector wire timer.

        This does not consume this governor's counters or skip rate waiting.
        The unchanged before_send remains the atomic shared budget boundary.
        """
        self.checkpoint()
        source = self.local.source
        with source.run.lock:
            if (
                source.requests >= source.max_requests
                or source.run.requests >= source.run.max_requests
            ):
                raise SearchStopped("request_budget")
        prepare = getattr(self.request_guard, "prepare_send", None)
        if prepare is not None:
            prepare(provider_code)
        self.checkpoint()

    def before_send(self, provider_code: str) -> float:
        """No URL, request, email, key or environment resolver crosses this API."""
        source = self.local.source
        state = self.states[provider_code]
        while True:
            self.checkpoint()
            with state.lock:
                if state.blocked:
                    raise SearchStopped("request_budget")
                delay = state.next_at - self.clock.monotonic()
                if delay <= 0:
                    with source.run.lock:
                        if (
                            source.requests >= source.max_requests
                            or source.run.requests >= source.run.max_requests
                        ):
                            raise SearchStopped("request_budget")
                        source.requests += 1
                        source.run.requests += 1
                    state.next_at = self.clock.monotonic() + state.interval
                    seconds = min(source.deadline, source.run.deadline) - self.clock.monotonic()
                    if self.request_guard is not None:
                        seconds = min(seconds, self.request_guard.before_send(provider_code))
                    return seconds
            self.clock.sleep(min(delay, self.policy.poll_ms / 1000))

    def after_send(
        self, provider_code: str, status: int, headers: tuple[tuple[str, str], ...]
    ) -> None:
        # SafeHTTPClient only sends the explicit limit-header allowlist, never wire
        # requests or general headers. Values are interpreted and never retained.
        state = self.states[provider_code]
        values = dict(headers)
        with state.lock:
            now = self.clock.monotonic()
            if status == 429:
                state.interval *= self.policy.penalty_factor
                state.next_at = max(state.next_at, now + self.policy.default_cooldown_ms / 1000)
            concurrency = values.get("x-concurrency-limit")
            if concurrency is not None:
                try:
                    if int(concurrency) < 1:
                        state.blocked = True
                except ValueError:
                    state.blocked = True
            raw = values.get("retry-after")
            if raw:
                try:
                    delay = (
                        float(raw)
                        if raw.isdecimal()
                        else (parsedate_to_datetime(raw) - self.clock.now()).total_seconds()
                    )
                    if not math.isfinite(delay) or delay < 0:
                        raise ValueError
                    if delay * 1000 > self.policy.max_header_delay_ms:
                        state.blocked = True  # Never truncate a server's long cooldown.
                    else:
                        state.next_at = max(state.next_at, now + delay)
                except (ValueError, TypeError, OverflowError):
                    state.blocked = True
            limit = values.get("x-rate-limit-limit", values.get("x-ratelimit-limit"))
            interval = values.get("x-rate-limit-interval", "1s")
            if limit is not None:
                try:
                    count = float(limit)
                    seconds = float(interval.removesuffix("s"))
                    if count <= 0 or seconds <= 0 or not math.isfinite(count + seconds):
                        raise ValueError
                    state.interval = max(state.interval, seconds / count)
                    state.next_at = max(state.next_at, now + state.interval)
                except (ValueError, OverflowError):
                    state.blocked = True
            remaining = values.get("x-rate-limit-remaining", values.get("x-ratelimit-remaining"))
            if remaining is not None:
                try:
                    if float(remaining) <= 0:
                        reset = values.get("x-ratelimit-reset")
                        if reset is None:
                            state.next_at = max(
                                state.next_at, now + self.policy.default_cooldown_ms / 1000
                            )
                        else:
                            delay = max(0, float(reset) - self.clock.now().timestamp())
                            if (
                                not math.isfinite(delay)
                                or delay * 1000 > self.policy.max_header_delay_ms
                            ):
                                state.blocked = True
                            else:
                                state.next_at = max(state.next_at, now + delay)
                except (ValueError, OverflowError):
                    state.blocked = True
