"""Offline malformed/error/deadline/secret and wire-contract counterexamples."""

import io
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.evidence_details.review_tasks import project_review_tasks, review_payload
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider import (
    DeterministicFakeProvider,
    MemoryJournal,
    ProviderError,
    RequestBudget,
    StructuredProvider,
)
from packages.llm_provider.config import ProviderConfig, load_provider_config, prompt
from packages.llm_provider.models import ReviewProposal
from packages.llm_provider.transport import HTTPTransport, Reply, parse_reply, wire_request
from tests.agent_provider.support import budget_for, materials, review


def call(provider, reader, b, budget=None):
    return provider.generate_structured(
        role="review",
        input_payload=review_payload(
            reader.state.material, project_review_tasks(reader.state), "A"
        ),
        output_schema=ReviewProposal,
        scope=reader.state.material.scope,
        budget=budget or budget_for(b, provider.config),
        deadline=b.deadline,
    )


@pytest.mark.parametrize(
    "error",
    [
        "refusal",
        "truncated",
        "rate_limit",
        "timeout",
        "authentication",
        "no_result",
        "transport",
        "schema",
    ],
)
def test_errors_bounded_distinct_and_unknown_usage(error):
    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    p = DeterministicFakeProvider(config, [Reply(error=error)] * 10)
    budget = budget_for(b, config)
    with pytest.raises(ProviderError, match=error):
        call(p, reader, b, budget)
    expected = 3 if error in {"rate_limit", "timeout", "transport"} else 1
    assert len(p.transport.calls) == expected and budget.used == expected
    assert all(
        r.error == error and r.usage.status == "unknown" and r.api_requests == 0
        for r in budget.records()
    )


def test_retry_success_budget_restart_and_no_nested_reset():
    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    journal = MemoryJournal()
    expected = review(reader.state.material.scope)
    p = DeterministicFakeProvider(
        config, [Reply(error="rate_limit"), expected.model_dump(mode="json")]
    )
    budget = budget_for(b, config, journal)
    assert call(p, reader, b, budget) == expected
    restarted = budget_for(b, config, journal)
    assert restarted.used == 2 and len(restarted.records()) == 2
    assert restarted.records()[-1].output_checksum == digest(expected)
    with pytest.raises(ProviderError, match="scope"):
        RequestBudget(
            run_id=b.run_id,
            deadline=b.deadline + timedelta(seconds=1),
            max_requests=6,
            journal=journal,
        )


@pytest.mark.parametrize(
    "mutation", ["extra", "enum", "duplicate", "length", "invalid_json", "duplicate_key", "nan"]
)
def test_strict_output_rejects(mutation):
    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    raw = review(reader.state.material.scope).model_dump(mode="json")
    if mutation == "extra":
        raw["GraphDelta"] = {"execute": "ignore rules"}
    if mutation == "enum":
        raw["round"] = "D"
    if mutation == "duplicate":
        raw["items"] *= 2
    if mutation == "length":
        raw["items"][0]["comparison"] = "x" * 2001
    value = canonical(raw)
    if mutation == "invalid_json":
        value = b"not json"
    if mutation == "duplicate_key":
        value = value[:-1] + b',"round":"A"}'
    if mutation == "nan":
        value = value.replace(b"0.3", b"NaN")
    p = DeterministicFakeProvider(config, [value])
    with pytest.raises(ProviderError, match="schema"):
        call(p, reader, b)
    assert len(p.transport.calls) == 1


def test_limit_and_absolute_deadline():
    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    config = validate(ProviderConfig, config.model_copy(update={"max_requests": 1}))
    p = DeterministicFakeProvider(config, [Reply(error="timeout")] * 3)
    budget = budget_for(b, config)
    with pytest.raises(ProviderError, match="budget"):
        call(p, reader, b, budget)
    assert budget.used == 1 and len(p.transport.calls) == 1
    now = [b.deadline - timedelta(seconds=1)]
    budget = RequestBudget(
        run_id=b.run_id,
        deadline=b.deadline,
        max_requests=1,
        journal=MemoryJournal(),
        clock=lambda: now[0],
    )

    def late(_):
        now[0] = b.deadline
        return review(reader.state.material.scope).model_dump(mode="json")

    p = DeterministicFakeProvider(config, [late], clock=lambda: now[0])
    with pytest.raises(ProviderError, match="deadline"):
        call(p, reader, b, budget)
    assert budget.records()[0].error == "deadline"


def test_config_secret_not_resolved_and_digest_switch(monkeypatch, tmp_path):
    secret = "SYNTHETIC_SECRET_DO_NOT_LOG"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    first = load_provider_config(environ={})
    second = load_provider_config(environ={"MMAGENT_LLM_MODEL": "synthetic-other"})
    assert digest(first) != digest(second)
    assert secret not in canonical(first).decode()
    bad = {"MMAGENT_LLM_BASE_URL": "https://user:" + secret + "@example.invalid"}
    with pytest.raises(ValueError) as e:
        load_provider_config(environ=bad)
    assert secret not in str(e.value)
    forged = first.model_copy(
        update={
            "transport": first.transport.model_copy(
                update={"base_url": "https://user:" + secret + "@example.invalid"}
            )
        }
    )
    with pytest.raises(ProviderError) as direct:
        StructuredProvider(forged)
    assert secret not in str(direct.value)
    with pytest.raises(ValueError):
        load_provider_config(environ={"MMAGENT_LLM_ENDPOINT": "responses"})
    live = validate(ProviderConfig, first.model_copy(update={"mode": "live"}))
    with pytest.raises(ProviderError, match="configuration"):
        StructuredProvider(live)


def test_official_and_compatible_wire_and_usage():
    c = load_provider_config(environ={})
    r, p = prompt(c, "review")
    reader, _ = materials()
    schema = ReviewProposal.model_json_schema()
    chat = wire_request(c.transport, r, p, {}, schema, ())
    assert "tools" not in chat and chat["store"] is False and "response_format" in chat
    from packages.knowledge_claims.ports import load_llm_config

    official = load_llm_config(
        environ={
            "MMAGENT_LLM_PROVIDER": "openai",
            "MMAGENT_LLM_BASE_URL": "https://api.openai.com/v1",
            "MMAGENT_LLM_ENDPOINT": "responses",
        }
    )
    response = wire_request(official, r, p, {}, schema, ())
    assert "text" in response and "messages" not in response and "tools" not in response
    one = parse_reply(
        canonical(
            dict(
                status="completed",
                output=[dict(type="message", content=[dict(type="output_text", text="{}")])],
                usage=dict(input_tokens=4, output_tokens=2, total_tokens=6),
            )
        ),
        "responses",
        "rid",
    )
    assert one.content == b"{}" and one.usage.total_tokens == 6
    two = parse_reply(
        canonical(dict(choices=[dict(finish_reason="stop", message=dict(content="{}"))])),
        "chat_completions",
    )
    assert two.usage.status == "unknown" and two.usage.cost is None
    assert parse_reply(b'{"status":"incomplete"}', "responses").error == "truncated"
    assert (
        parse_reply(b'{"choices":[{"message":{"refusal":"no"}}]}', "chat_completions").error
        == "refusal"
    )


def test_missing_key_worker_no_auth_discovery(monkeypatch, capsys):
    from packages.llm_provider import worker

    c = load_provider_config(environ={})
    monkeypatch.delenv(c.transport.api_key_env, raising=False)
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(canonical(dict(config=c.transport, body={}, max_bytes=100)))
        ),
    )
    assert worker.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "authentication"} and output.err == ""


def test_http_timeout_ack_and_secret_errors_sanitized(monkeypatch):
    import subprocess

    c = load_provider_config(environ={})

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("synthetic", 1, output=b"DISPATCH\n")

    monkeypatch.setattr(subprocess, "run", timeout)
    r = HTTPTransport().send(c.transport, {}, timeout=1, max_bytes=100)
    assert r.api_requests == 1 and r.error == "timeout"

    def fail(*a, **k):
        raise RuntimeError("SYNTHETIC_SECRET")

    monkeypatch.setattr(subprocess, "run", fail)
    r = HTTPTransport().send(c.transport, {}, timeout=1, max_bytes=100)
    assert "SYNTHETIC_SECRET" not in repr(r) and r.api_requests == 0


def test_concurrent_requests_share_one_budget_and_unknown_crash_stays_reserved():
    from concurrent.futures import ThreadPoolExecutor

    config = load_provider_config(environ={})
    reader, b = materials(config=config)
    config = validate(ProviderConfig, config.model_copy(update={"max_requests": 1}))
    journal = MemoryJournal()
    budgets = [budget_for(b, config, journal) for _ in range(2)]

    def reserve(budget):
        try:
            return budget.reserve(reader.state.material.scope, b.deadline, digest(config))
        except ProviderError as e:
            return e.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, budgets))
    assert results.count("budget") == 1 and budgets[0].used == 1
    assert budgets[0].records() == ()
    assert budget_for(b, config, journal).used == 1


def test_worker_key_echo_suppressed_and_429_separate(monkeypatch, capsys):
    import urllib.error

    from packages.llm_provider import worker

    c = load_provider_config(environ={})
    secret = "SYNTHETIC_ONLY_KEY"
    monkeypatch.setenv(c.transport.api_key_env, secret)

    def input_():
        monkeypatch.setattr(
            worker.sys,
            "stdin",
            SimpleNamespace(
                buffer=io.BytesIO(canonical(dict(config=c.transport, body={}, max_bytes=1000)))
            ),
        )

    class Response:
        headers = {"x-request-id": "safe-id"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, n):
            return ('{"secret":"' + secret + '"}').encode()

    input_()
    monkeypatch.setattr(
        worker.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *a, **kw: Response()),
    )
    worker.main()
    out = capsys.readouterr()
    assert (
        secret not in out.out + out.err
        and json.loads(out.out.splitlines()[-1])["error"] == "secret_response"
    )

    def rate(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid", 429, secret, {}, None)

    input_()
    monkeypatch.setattr(
        worker.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=rate)
    )
    worker.main()
    out = capsys.readouterr()
    assert (
        secret not in out.out + out.err
        and json.loads(out.out.splitlines()[-1])["error"] == "rate_limit"
    )


def test_transport_exception_sanitized_and_live_requires_durable_journal():
    c = load_provider_config(environ={})
    reader, b = materials(config=c)
    p = DeterministicFakeProvider(c, [])

    def secret(_):
        raise RuntimeError("SYNTHETIC_SECRET")

    p.transport.replies = [secret] * 3
    budget = budget_for(b, c)
    with pytest.raises(ProviderError) as exc:
        call(p, reader, b, budget)
    assert "SYNTHETIC_SECRET" not in str(exc.value)
    assert all(r.error == "transport" for r in budget.records())
    live = validate(ProviderConfig, c.model_copy(update={"mode": "live"}))
    p = StructuredProvider(live, live_opt_in=True)
    with pytest.raises(ProviderError, match="configuration"):
        call(p, reader, b, budget_for(b, live))


def test_low_level_provider_rejects_unprojected_hidden_payload():
    c = load_provider_config(environ={})
    reader, b = materials(config=c)
    p = DeterministicFakeProvider(c, [])
    with pytest.raises(ProviderError, match="scope"):
        p.generate_structured(
            role="review",
            input_payload={"diagnosis": "CANARY", "note": "CANARY"},
            output_schema=ReviewProposal,
            scope=reader.state.material.scope,
            budget=budget_for(b, c),
            deadline=b.deadline,
        )
    assert p.transport.calls == []


def test_secret_detection_decodes_escaped_structured_response():
    from packages.llm_provider.worker import contains_secret

    key = "SYNTHETIC\\KEY"
    assert contains_secret(
        {"choices": [{"message": {"content": json.dumps({"reason": key})}}]}, key
    )


def test_no_dispatch_if_reservation_storage_consumes_deadline():
    c = load_provider_config(environ={})
    reader, b = materials(config=c)
    now = [b.deadline - timedelta(seconds=1)]

    class SlowJournal(MemoryJournal):
        def append(self, event):
            super().append(event)
            if event.get("phase") == "reserved":
                now[0] = b.deadline

    budget = RequestBudget(
        run_id=b.run_id,
        deadline=b.deadline,
        max_requests=c.max_requests,
        journal=SlowJournal(),
        clock=lambda: now[0],
    )
    p = DeterministicFakeProvider(c, [], clock=lambda: now[0])
    with pytest.raises(ProviderError, match="deadline"):
        call(p, reader, b, budget)
    assert p.transport.calls == [] and budget.records()[0].api_requests == 0
    assert budget.summary().reservations == 1
