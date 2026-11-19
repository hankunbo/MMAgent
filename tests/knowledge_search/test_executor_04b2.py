import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from packages.knowledge_connectors import (
    ConnectorConfiguration,
    ErrorCategory,
    FakeClock,
    SafeHTTPClient,
)
from packages.knowledge_search import FederatedSearch, ProviderBudget
from packages.knowledge_search.policy import (
    ProviderGovernor,
    RunBudget,
    SearchStopped,
    SourceBudget,
)
from tests.knowledge_search.support import assembly, config, exchange, plan, policy


def test_single_source_safe_hits_and_empty_results():
    c = config()
    p = plan(c)
    executor, transports = assembly((c,), {c.provider_code: (exchange(c, p),)})
    with executor:
        result = executor.execute(p)
    assert result.status == "succeeded"
    source = result.providers[0]
    assert source.hit_count == 1 and source.wire_requests == 1
    assert source.hits[0].metadata.ranking_semantics == "retrieval_relevance_only"
    assert source.receipts[0].requests[0].config_checksum == c.checksum()
    transports[c.provider_code].assert_consumed()
    executor, _ = assembly((c,), {c.provider_code: (exchange(c, p, ids=()),)})
    with executor:
        assert executor.execute(p).status == "empty"


def test_independent_failure_preserves_success():
    a, b = config(), config(provider_code="SOURCE:PROVIDER:OTHER")
    p = plan(a, b)
    executor, _ = assembly(
        (a, b), {a.provider_code: (exchange(a, p),), b.provider_code: (exchange(b, p, status=401),)}
    )
    with executor:
        result = executor.execute(p)
    assert result.status == "partial"
    assert sum(o.hit_count for o in result.providers) == 1
    assert {o.error_category for o in result.providers} == {None, "authentication"}
    assert next(o for o in result.providers if o.error_category).error_requests


@pytest.mark.parametrize("limit", [0, 1])
def test_every_retry_consumes_total_request_budget(limit):
    c = config()
    p = plan(c, total_max_requests=limit)
    tape = (exchange(c, p, status=500),) if limit else ()
    executor, transports = assembly((c,), {c.provider_code: tape})
    with executor:
        result = executor.execute(p)
    assert result.status == "budget_exhausted"
    assert result.providers[0].wire_requests == limit
    assert transports[c.provider_code].calls == limit


def test_timeout_and_precancel_are_explicit():
    c = config()
    p = plan(c, total_timeout_ms=1)
    executor, transports = assembly(
        (c,), {c.provider_code: (exchange(c, p, error=ErrorCategory.TIMEOUT),)}
    )
    with executor:
        result = executor.execute(p)
    assert result.status == "timeout"
    cancel = threading.Event()
    cancel.set()
    executor, transports = assembly((c,), {})
    with executor:
        result = executor.execute(p, cancel=cancel)
    assert result.status == "cancelled" and transports[c.provider_code].calls == 0


def test_duplicate_page_and_logical_hits_across_pages():
    c = config()
    p = plan(c)
    tape = (
        exchange(c, p, ids=("1", "2"), next_cursor="next"),
        exchange(c, p, ids=("2", "3"), cursor="next", next_cursor="third"),
        exchange(c, p, ids=("3", "2"), cursor="third", next_cursor="fourth"),
    )
    executor, _ = assembly((c,), {c.provider_code: tape})
    with executor:
        result = executor.execute(p)
    source = result.providers[0]
    assert result.status == "partial" and source.stop_reason == "duplicate_page"
    assert source.hit_count == 3 and len(source.receipts) == 3


def test_page_and_hit_budget_keep_discoveries():
    c = config()
    for changes, reason, hits in [
        (
            {
                "providers": (
                    ProviderBudget(provider_code=c.provider_code, page_size=2, max_pages=1),
                )
            },
            "page_limit",
            2,
        ),
        ({"total_max_hits": 1}, "hit_budget", 1),
    ]:
        p = plan(c, **changes)
        executor, _ = assembly(
            (c,), {c.provider_code: (exchange(c, p, ids=("1", "2"), next_cursor="next"),)}
        )
        with executor:
            result = executor.execute(p)
        assert result.status == "partial" and result.providers[0].stop_reason == reason
        assert result.providers[0].hit_count == hits


def test_429_retry_after_slows_all_requests_not_just_current_retry():
    c = config()
    p = plan(c)
    clock = FakeClock()
    tape = (
        exchange(c, p, status=429, headers=(("Retry-After", "3"),)),
        exchange(c, p),
        exchange(c, p),
    )
    executor, transports = assembly((c,), {c.provider_code: tape}, clock=clock)
    with executor:
        first = executor.execute(p)
        assert clock.elapsed >= 3 and first.providers[0].wire_requests == 2
        before = clock.elapsed
        second = executor.execute(p)
        assert second.status == "succeeded" and clock.elapsed >= before + 1
    assert transports[c.provider_code].calls == 3


def test_provider_limit_headers_conservatively_reduce_rate():
    c = config()
    p = plan(c)
    clock = FakeClock()
    executor, _ = assembly(
        (c,),
        {
            c.provider_code: (
                exchange(
                    c, p, headers=(("X-Rate-Limit-Limit", "1"), ("X-Rate-Limit-Interval", "4s"))
                ),
                exchange(c, p),
            )
        },
        clock=clock,
    )
    with executor:
        executor.execute(p)
        executor.execute(p)
        assert clock.elapsed >= 4
        assert executor.governor.states[c.provider_code].interval >= 4


def test_multiple_tasks_share_provider_admission_and_cannot_construct_second_executor():
    c = config()
    p = plan(c)
    executor, transports = assembly(
        (c,), {c.provider_code: tuple(exchange(c, p) for _ in range(4))}
    )
    with executor:
        with pytest.raises(ValueError, match="already belongs"):
            assembly((c,), {})
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(lambda _: executor.execute(p), range(4)))
        assert all(r.status == "succeeded" for r in results)
        assert executor.clock.elapsed >= 1.5
    assert transports[c.provider_code].calls == 4


def test_governor_shared_run_budget_across_parallel_tasks():
    c = config()
    clock = FakeClock()
    governor = ProviderGovernor((c,), policy(), clock)
    run = RunBudget(100, 5, 10, threading.Event())

    def work(_):
        count = 0
        with governor.bind(SourceBudget(run, 100, 5)):
            try:
                while True:
                    governor.before_send(c.provider_code)
                    count += 1
            except SearchStopped:
                return count

    with ThreadPoolExecutor(max_workers=3) as pool:
        counts = tuple(pool.map(work, range(3)))
    assert sum(counts) == 5 and run.requests == 5
    assert clock.elapsed >= 2


def test_factory_cannot_bypass_shared_governor():
    c = config()
    from packages.knowledge_connectors import RecordedTransport

    with pytest.raises(ValueError, match="executor governor"):
        FederatedSearch(
            ConnectorConfiguration(config_version="2.0.0", providers=(c,)),
            policy(),
            lambda cfg, governor: SafeHTTPClient(cfg, RecordedTransport(())),
        )


def test_privacy_rejects_provider_echo_in_title():
    c = config()
    p = plan(c)
    executor, _ = assembly(
        (c,),
        {
            c.provider_code: (
                exchange(c, p, title="https://service.example.org?email=secret@example.invalid"),
            )
        },
    )
    from packages.knowledge_search import PrivacyError

    with executor, pytest.raises(PrivacyError, match="privacy policy"):
        executor.execute(p)


def test_large_or_invalid_retry_after_fails_closed_without_truncation():
    c = config()
    for value in ("999999999", "invalid"):
        p = plan(c)
        executor, transports = assembly(
            (c,),
            {c.provider_code: (exchange(c, p, status=429, headers=(("Retry-After", value),)),)},
        )
        with executor:
            result = executor.execute(p)
        assert result.status == "budget_exhausted"
        assert transports[c.provider_code].calls == 1


def test_successful_empty_provider_is_preserved_as_partial_when_another_fails():
    a, b = config(), config(provider_code="SOURCE:PROVIDER:EMPTY_PEER")
    p = plan(a, b)
    executor, _ = assembly(
        (a, b),
        {
            a.provider_code: (exchange(a, p, ids=()),),
            b.provider_code: (exchange(b, p, status=401),),
        },
    )
    with executor:
        result = executor.execute(p)
    assert result.status == "partial"
    assert {o.stop_reason for o in result.providers} == {"empty", "provider_error"}
