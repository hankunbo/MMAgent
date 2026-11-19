"""Injected worker observations; no sockets, patient material, or caller confirmation trust."""

import json
from types import SimpleNamespace

import pytest

from apps.vectorengine_public_diagnostic import public_step
from apps.vectorengine_single_case import load_config
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import canonical, digest
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_probes import PublicProbeMaterials
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.transport import HTTPTransport, Reply
from packages.llm_provider.worker import safe_http_error
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.support import review_from_wire
from tests.agent_provider.test_bounded_live import generate
from tests.agent_provider.test_live_terminal_guard import controlled, stop

CONFIG = "configs/agents/vectorengine_compat_case.json"


def injected_worker(monkeypatch, replies):
    calls = []

    def run(command, *, input, **kwargs):
        body = json.loads(input)["body"]
        calls.append(body)
        value = replies.pop(0)
        if callable(value):
            value = value(body)
        if value == "unknown":
            return SimpleNamespace(returncode=0, stdout=b'DISPATCH\n{"error":"timeout"}')
        if isinstance(value, int):
            value = dict(
                error="authentication"
                if value == 401
                else "rate_limit"
                if value == 429
                else "transport",
                diagnostic=safe_http_error(value, b'{"error":{"code":"server_error"}}'),
            )
        elif "error" not in value:
            value = dict(
                http_status=200,
                request_id="synthetic-public",
                body=dict(
                    model="gpt-5.6-sol",
                    choices=[dict(finish_reason="stop", message=dict(content=json.dumps(value)))],
                    usage=dict(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                ),
            )
        return SimpleNamespace(returncode=0, stdout=b"DISPATCH\n" + canonical(value))

    import subprocess

    monkeypatch.setattr(
        "packages.llm_provider.transport.subprocess",
        SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    return calls


def recovery_fixture(monkeypatch, replies):
    cfg = load_config(CONFIG)
    monkeypatch.setattr("tests.agent_provider.test_bounded_live.load_config", lambda: cfg)
    f = controlled()
    f.permission.budget = f.budget
    f.provider = StructuredProvider(
        cfg, live_opt_in=True, transport=HTTPTransport(), sleep=lambda _: None
    )
    f.calls = injected_worker(monkeypatch, list(replies))
    return f


def test_500_success_two_reservations_one_acceptance_and_recovery(monkeypatch):
    f = recovery_fixture(monkeypatch, [500, review_from_wire])
    sub = generate(f)
    service = ResearchAcceptance(
        owner=f.reader, documents=f.docs, requests=f.budget, live_permission=f.permission
    )
    service.submit(sub, key="retry-candidate")
    receipt = service.accept(key="retry-candidate")
    assert receipt.status == "accepted"
    assert service.accept(key="retry-candidate") == receipt
    assert len(f.calls) == f.budget.used == 2
    assert [r.error for r in f.budget.records()] == ["transport", None]
    assert [r.retry_index for r in f.budget.records()] == [0, 1]
    again = BoundedLiveBudget(
        binding=f.binding, config=f.config, journal=RuntimeAgentJournal(f.docs)
    )
    assert again.used == 2 and again.deadline == f.budget.deadline
    again.before_reserve(digest(f.config))
    assert len(f.calls) == 2 and f.budget.transport_counts()["remote_response_confirmed"] == 2


@pytest.mark.parametrize(
    "replies,count,reason",
    [
        ([500, 500], 2, "transport"),
        (["unknown"], 1, "timeout"),
        ([401], 1, "authentication"),
        ([429], 1, "rate_limit"),
        (
            [
                dict(
                    error="rate_limit",
                    diagnostic=safe_http_error(429, b'{"error":{"code":"insufficient_quota"}}'),
                )
            ],
            1,
            "rate_limit",
        ),
    ],
)
def test_no_unbounded_or_unknown_retry(monkeypatch, replies, count, reason):
    f = recovery_fixture(monkeypatch, replies)
    with pytest.raises(ProviderError, match=reason):
        generate(f)
    assert len(f.calls) == f.budget.used == count
    assert not f.docs.consumed(f.binding.graph)
    if count == 2:
        with pytest.raises(ProviderError, match="transport"):
            f.budget.before_reserve(digest(f.config))


def test_business_scope_keeps_raw_and_usage_zero_retry(monkeypatch):
    def wrong(body):
        value = review_from_wire(body)
        value["scope_checksum"] = digest("foreign")
        return value

    f = recovery_fixture(monkeypatch, [wrong])
    with pytest.raises(ProviderError, match="scope"):
        generate(f)
    r = f.budget.records()[0]
    assert f.budget.used == len(f.calls) == 1 and r.usage.total_tokens == 30
    assert f.docs.read("live-response:" + r.request_id)[0]["validation_error"] == "scope"


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_backoff_owner_control(monkeypatch, mode):
    f = recovery_fixture(monkeypatch, [500, review_from_wire])

    def during_sleep(_):
        if mode == "cancel":
            stop(f, "cancel")
        else:
            f.budget.clock = lambda: f.binding.deadline

    f.provider.sleep = during_sleep
    with pytest.raises(ProviderError, match="cancelled|deadline"):
        generate(f)
    assert len(f.calls) == f.budget.used == 1


def test_caller_confirmed_diagnostic_does_not_authorize_retry(monkeypatch):
    f = recovery_fixture(monkeypatch, [])
    f.provider.transport = f.wire
    f.wire.replies = [
        Reply(
            error="transport",
            api_requests=1,
            response_confirmed=True,
            diagnostic=safe_http_error(500, b"{}"),
        )
    ]
    with pytest.raises(ProviderError, match="transport"):
        generate(f)
    assert len(f.wire.calls) == 1
    with pytest.raises(ProviderError, match="transport"):
        f.budget.before_reserve(digest(f.config))


def test_saved_public_success_no_resend_and_stop_denies_new_step(monkeypatch):
    def vision(body):
        p = body["response_format"]["json_schema"]["schema"]["properties"]
        return {
            **{
                k: p[k]["enum"][0]
                for k in ("scope_checksum", "image_checksum", "preprocessing_checksum", "location")
            },
            "status": "observed",
            "finding": "red synthetic square",
        }

    f = recovery_fixture(monkeypatch, [500, vision])
    materials = PublicProbeMaterials(f.binding.run_id, f.config)
    args = dict(
        name="vision",
        role="vision",
        provider=f.provider,
        budget=f.budget,
        documents=f.docs,
        materials=materials,
    )
    assert public_step(**args)["status"] == "passed"
    assert public_step(**args)["status"] == "passed"
    stop(f)
    assert public_step(**args)["status"] == "passed"
    with pytest.raises(ProviderError, match="stopped"):
        public_step(**dict(args, name="new"))
    assert f.budget.used == len(f.calls) == 2
    assert not f.docs.read("compat:new:claim")


def test_error_redaction_closed_vocabulary_and_backoff():
    from datetime import UTC, datetime

    from packages.llm_provider.http_recovery import retry_delay

    diagnostic = safe_http_error(
        500,
        b'{"error":{"message":"patient secret-text image_url invalid schema",'
        b'"code":"secret-text"}}',
        {"x-request-id": "secret-text", "Retry-After": "2"},
        "secret-text",
    )
    assert "secret-text" not in json.dumps(diagnostic) and "patient" not in json.dumps(diagnostic)
    assert diagnostic["request_id"] is None and diagnostic["error_code"] is None
    assert retry_delay(diagnostic, now=datetime.now(UTC), default=0.5) == 2
