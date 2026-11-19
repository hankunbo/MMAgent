"""Use only synthetic tapes through real SafeHTTPClient and adapter code."""

import json
import threading
from datetime import timedelta
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from packages.knowledge_connectors import FakeClock, FixtureExchange, load_fixture
from packages.knowledge_search import KnowledgeQueryPlan, ProviderBudget
from packages.knowledge_search.planning import provider_query
from tests.knowledge_search.support import assembly, config, exchange, plan, policy


def pubmed_tape(c, p, email="fixture@example.invalid"):
    tape = load_fixture(Path("tests/knowledge_connectors/fixtures/pubmed.json"))
    text = provider_query(p, c, policy()).texts[0]
    first = tape.exchanges[0]
    values = dict(first.query) | {"term": text, "retmax": "2", "email": email}
    first = FixtureExchange.model_validate(
        first.model_copy(
            update={
                "query": tuple(values.items()),
                "body_text": json.dumps({"esearchresult": {"count": "1", "idlist": ["999990001"]}}),
            }
        )
    )
    second = tape.exchanges[1]
    second = FixtureExchange.model_validate(
        second.model_copy(
            update={
                "query": tuple((dict(second.query) | {"email": email}).items()),
            }
        )
    )
    return first, second


@pytest.mark.parametrize("budget", [1, 2])
def test_pubmed_summary_shares_wire_budget_and_retains_parent_request(budget):
    c = config("pubmed")
    p = plan(c, total_max_requests=budget)
    tape = pubmed_tape(c, p)[:budget]
    executor, transports = assembly((c,), {c.provider_code: tape})
    with executor:
        result = executor.execute(p)
    source = result.providers[0]
    assert transports[c.provider_code].calls == budget
    assert len(source.wire_receipts) == budget
    if budget == 1:
        assert source.stop_reason == "request_budget" and source.hit_count == 0
        assert source.error_requests[0].operation == "search"
    else:
        assert result.status == "succeeded"
        search, summary = source.receipts[0].requests
        assert summary.search_request_checksum == search.request_checksum
        assert [r.operation for r in source.wire_receipts] == ["search", "summary"]
    assert "fixture@example.invalid" not in source.model_dump_json()
    assert c.pubmed_tool not in source.model_dump_json()


def test_contact_environment_is_resolved_only_by_send_boundary_once_per_attempt():
    c = config("pubmed")
    p = plan(c)
    calls = []

    def environment(name):
        calls.append(name)
        return "first@example.invalid" if len(calls) == 1 else "second@example.invalid"

    a, _ = pubmed_tape(c, p, "first@example.invalid")
    _, b = pubmed_tape(c, p, "second@example.invalid")
    executor, _ = assembly((c,), {c.provider_code: (a, b)}, environment=environment)
    assert not calls
    with executor:
        executor.queries(p)
        executor.execution_checksum(p)
        assert not calls
        result = executor.execute(p)
    assert result.status == "succeeded" and calls == [c.contact_email_env] * 2
    assert "first@" not in result.model_dump_json() and "second@" not in result.model_dump_json()


@pytest.mark.parametrize("budget", [1, 2])
def test_redirect_is_separately_admitted_and_has_safe_checksums(budget):
    c = config()
    p = plan(c, total_max_requests=budget)
    success = exchange(c, p)
    location = success.path + "?" + urlencode(success.query)
    redirect = FixtureExchange.model_validate(
        success.model_copy(
            update={
                "status": 302,
                "headers": (("location", location),),
                "body_text": "redirect fixture",
            }
        )
    )
    executor, transports = assembly((c,), {c.provider_code: (redirect, success)[:budget]})
    with executor:
        result = executor.execute(p)
    assert transports[c.provider_code].calls == budget
    wire = result.providers[0].wire_receipts
    assert len(wire) == budget and wire[0].http_status == 302
    assert wire[0].request_checksum.startswith("sha256:")
    assert wire[0].response_checksum.startswith("sha256:")
    assert location not in result.model_dump_json()
    assert result.status == ("budget_exhausted" if budget == 1 else "succeeded")


def test_http_date_retry_after_and_no_key_vs_key_policy_tier():
    c = config()
    p = plan(c)
    clock = FakeClock()
    after = format_datetime(clock.now() + timedelta(seconds=4), usegmt=True)
    executor, _ = assembly(
        (c,),
        {
            c.provider_code: (
                exchange(c, p, status=429, headers=(("Retry-After", after),)),
                exchange(c, p),
            )
        },
        clock=clock,
    )
    with executor:
        assert executor.execute(p).status == "succeeded"
        assert clock.elapsed >= 4
    from packages.knowledge_connectors import HTTPPolicy
    from packages.knowledge_search.policy import ProviderGovernor

    for keyed, expected in ((False, 1 / 3), (True, 1 / 10)):
        c = config("pubmed")
        c = config(
            "pubmed",
            api_key_env="NCBI_TEST_KEY" if keyed else None,
            auth_mode="query" if keyed else "none",
            http=HTTPPolicy.model_validate(
                c.http.model_copy(update={"rate_limit_per_minute": 600 if keyed else 180})
            ),
        )
        governor = ProviderGovernor((c,), policy(), FakeClock())
        assert governor.states[c.provider_code].interval == expected


def test_cancel_while_waiting_for_shared_rate_never_sends_another_attempt():
    c = config()
    p = plan(c)
    cancel = threading.Event()

    class CancellingClock(FakeClock):
        def sleep(self, seconds):
            super().sleep(seconds)
            cancel.set()

    executor, transports = assembly(
        (c,), {c.provider_code: (exchange(c, p, status=500),)}, clock=CancellingClock()
    )
    with executor:
        result = executor.execute(p, cancel=cancel)
    assert result.status == "cancelled" and transports[c.provider_code].calls == 1


def test_per_provider_timeout_does_not_cancel_successful_provider():
    a, b = config(), config(provider_code="SOURCE:PROVIDER:SLOW")
    budgets = (
        ProviderBudget(provider_code=a.provider_code, page_size=2),
        ProviderBudget(provider_code=b.provider_code, page_size=2, timeout_ms=1),
    )
    p = plan(a, b, providers=budgets, max_concurrency=1)
    executor, _ = assembly(
        (a, b),
        {
            a.provider_code: (exchange(a, p),),
            b.provider_code: (exchange(b, p, status=500),),
        },
    )
    with executor:
        result = executor.execute(p)
    assert result.status == "partial" and sum(o.hit_count for o in result.providers) == 1
    assert (
        next(
            o for o in result.providers if o.query.identity.provider_code == b.provider_code
        ).stop_reason
        == "timeout"
    )


def test_independent_provider_pagination_and_total_hits_are_bounded():
    a, b = config(), config(provider_code="SOURCE:PROVIDER:PAGED")
    p = plan(a, b, total_max_hits=3, max_concurrency=1)
    executor, _ = assembly(
        (a, b),
        {
            a.provider_code: (exchange(a, p, ids=("1", "2")),),
            b.provider_code: (exchange(b, p, ids=("1", "2")),),
        },
    )
    with executor:
        result = executor.execute(p)
    # Same paper at different providers remains separate discovery paths (04B3 owns dedup).
    assert sum(o.hit_count for o in result.providers) == 3
    assert result.status == "partial"


def test_expansion_trace_cannot_be_rebound_to_changed_population():
    from packages.knowledge_search import ExpansionConfig, expand_plan
    from tests.knowledge_search.test_planning_04b2 import FakeLLM

    p = expand_plan(plan(config()), ExpansionConfig(enabled=True), FakeLLM('{"suggestions":[]}'))
    with pytest.raises(ValueError):
        KnowledgeQueryPlan.model_validate(p.model_copy(update={"population": ("children",)}))
    assert p.expansion.prompt and p.expansion.config.enabled


def test_plan_can_reduce_provider_retries_to_zero_without_changing_config():
    c = config()
    p = plan(
        c, providers=(ProviderBudget(provider_code=c.provider_code, page_size=2, max_retries=0),)
    )
    executor, transports = assembly((c,), {c.provider_code: (exchange(c, p, status=500),)})
    with executor:
        result = executor.execute(p)
    assert result.status == "failed" and transports[c.provider_code].calls == 1
    source = result.providers[0]
    assert len(source.error_requests[0].attempts) == 1
    assert source.query.identity.config_checksum == c.checksum()
