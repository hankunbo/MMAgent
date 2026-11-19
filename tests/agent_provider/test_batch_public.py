"""Real transport parsing with an injected subprocess; no actual HTTP or Key."""

from dataclasses import replace

import pytest

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from apps.vectorengine_single_case import load_config
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError, StructuredProvider
from packages.llm_provider.public_batches import (
    case_estimate,
    check_public_batches,
    public_plan,
    run_public_batches,
)
from packages.llm_provider.public_shape import PublicReviewMaterials
from packages.llm_provider.transport import HTTPTransport
from tests.agent_provider.batch_support import CONFIG, batch_reply, fixture
from tests.agent_provider.test_http_recovery import injected_worker
from tests.agent_provider.test_live_terminal_guard import controlled, stop


def controlled_batch(monkeypatch, replies):
    config = load_config(CONFIG)
    monkeypatch.setattr("tests.agent_provider.test_bounded_live.load_config", lambda: config)
    f = controlled()
    f.calls = injected_worker(monkeypatch, list(replies))
    f.provider = StructuredProvider(
        config, transport=HTTPTransport(), live_opt_in=True, sleep=lambda _: None
    )
    owner = PublicReviewMaterials(f.binding.run_id, config, task_count=3)
    f.batch_owner = owner
    f.agent = StructuredRoleAgent("review", provider=f.provider, budget=f.budget, materials=owner)
    return f


def execute(f):
    return run_batches(
        f.agent,
        state=f.batch_owner.current(),
        documents=f.docs,
        key="public-http-batch",
        round_id="A",
    )


def test_public_1_then_3_synthetic_qualification_cannot_be_real():
    f = fixture(3)
    q = run_public_batches(f.config, f.budget, f.documents, digest("round"), provider=f.provider)
    assert [len(s["candidate"]["items"]) for s in q["submissions"]] == [1, 3]
    assert f.budget.used == 2 and q["execution"] == "synthetic_test"
    assert (
        run_public_batches(f.config, f.budget, f.documents, digest("round"), provider=f.provider)
        == q
    )
    assert f.budget.used == 2
    with pytest.raises(ValueError, match="real_public"):
        check_public_batches(f.config, f.budget, f.documents, q, digest("round"), require_live=True)
    forged = dict(q, execution="live_http")
    with pytest.raises(ValueError, match="real_public_context"):
        check_public_batches(f.config, f.budget, f.documents, forged, digest("round"))


def test_small_plan_reuses_exact_source_vision_and_duration_bound():
    config = load_config(CONFIG)
    p = public_plan("RUN:SYNTHETIC", config)
    assert [s["task_count"] for s in p["steps"]] == [1, 3]
    assert [s["role"] for s in p["source_vision_reuse"]] == ["source", "vision"]
    assert case_estimate(45, config, 2, [1.0, 5.0])["feasible"]
    assert not case_estimate(45, config, 8, [1.0, 5.0])["feasible"]
    assert not case_estimate(45, config, 2, [240.0])["feasible"]
    assert not case_estimate(46, config, 2, [5.0])["feasible"]


def test_confirmed_temporary_retry_and_saved_response_usage(monkeypatch):
    f = controlled_batch(monkeypatch, [500, batch_reply])
    sub = execute(f)
    assert len(f.calls) == f.budget.used == 2 and len(sub.batches) == 1
    assert [r.error for r in f.budget.records()] == ["transport", None]
    assert f.budget.records()[-1].usage.total_tokens == 30
    assert execute(f) == sub and len(f.calls) == 2


@pytest.mark.parametrize("terminal", ["stop", "cancel", "deadline"])
def test_original_control_stops_internal_retry(monkeypatch, terminal):
    f = controlled_batch(monkeypatch, [500, batch_reply])

    def during_sleep(_):
        if terminal == "deadline":
            f.budget.clock = lambda: f.binding.deadline
        else:
            stop(f, terminal)

    f.provider.sleep = during_sleep
    with pytest.raises(ProviderError, match="stopped|cancelled|deadline"):
        execute(f)
    assert f.budget.used == len(f.calls) == 1


def test_unknown_no_retry_no_second_claim(monkeypatch):
    f = controlled_batch(monkeypatch, ["unknown"])
    with pytest.raises(ProviderError, match="timeout"):
        execute(f)
    with pytest.raises(ProviderError, match="timeout|unknown_no_repeat"):
        execute(f)
    assert f.budget.used == len(f.calls) == 1
    assert f.budget.transport_counts()["remote_response_unknown"] == 1


def test_confirmed_response_stop_keeps_raw_and_usage_without_acceptance(monkeypatch):
    f = controlled_batch(monkeypatch, [batch_reply])
    original = f.provider.transport.send

    def send(*args, **kwargs):
        reply = original(*args, **kwargs)
        stop(f)
        return replace(reply, usage=reply.usage)

    f.provider.transport.send = send
    with pytest.raises(ProviderError, match="stopped"):
        execute(f)
    r = f.budget.records()[0]
    assert r.usage.total_tokens == 30
    assert f.docs.read("live-response:" + r.request_id)[0]["raw_model_text"]
    assert not any(k.startswith("accepted:") for k in f.docs.saved)
    assert len(f.calls) == 1


def test_outer_timeout_inner_http_and_absolute_deadline(monkeypatch):
    f = controlled_batch(monkeypatch, [batch_reply])
    envelopes = []
    import json

    from packages.llm_provider import transport

    original = transport.subprocess.run

    def run(command, *, input, **kwargs):
        envelopes.append((json.loads(input), kwargs["timeout"]))
        return original(command, input=input, **kwargs)

    monkeypatch.setattr(transport.subprocess, "run", run)
    execute(f)
    envelope, outer = envelopes[0]
    assert 0 < envelope["effective_http_timeout"] < outer <= 240
    assert f.budget.deadline == f.binding.deadline


def test_frozen_public_wire_scope_bound_and_restart_budget_not_reset(monkeypatch):
    from datetime import timedelta

    from packages.llm_provider.bounded_live import BoundedLiveBudget
    from packages.llm_provider.live_allocation import StageAllocation
    from packages.research_runtime.agent_records import RuntimeAgentJournal
    from tests.agent_provider.support import Documents
    from tests.planner.support import MemoryRuntime, binding

    config = load_config(CONFIG)
    b = binding()
    b = b.model_copy(
        update={
            "profile": b.profile.model_copy(
                update={"max_steps": 20, "max_no_progress": 2, "max_retries": 1, "seconds": 1200}
            ),
            "deadline": b.started_at + timedelta(seconds=1200),
        }
    )
    docs = Documents()
    docs.run_id, docs.sessions = b.run_id, MemoryRuntime()
    docs.sessions.origin = lambda _: (b.started_at, b.run_context_checksum)
    docs.sessions.begin(b)
    allocation = StageAllocation(
        session_id="batch-review",
        stage="public",
        public_run_id=b.run_id,
        public_used=0,
        public_journal_checksum=None,
        configuration_checksums=(digest(config),),
    )
    budget = BoundedLiveBudget(
        binding=b, config=config, journal=RuntimeAgentJournal(docs), allocation=allocation
    )
    calls = injected_worker(monkeypatch, [batch_reply, batch_reply])
    provider = StructuredProvider(config, transport=HTTPTransport(), live_opt_in=True)
    # Explicit synthetic Runtime owner prevents a test from minting real qualification.
    from types import SimpleNamespace

    docs.contexts = SimpleNamespace(read_research_run=lambda _: SimpleNamespace(synthetic=True))
    q = run_public_batches(config, budget, docs, digest("round"), provider=provider)
    assert q["execution"] == "synthetic_test" and len(calls) == budget.used == 2
    again = BoundedLiveBudget(
        binding=b, config=config, journal=RuntimeAgentJournal(docs), allocation=allocation
    )
    assert again.used == 2 and again.deadline == budget.deadline
    scope = PublicReviewMaterials(b.run_id, config, task_count=1).state.material.scope
    with pytest.raises(ProviderError, match="configuration"):
        again.before_wire(digest(config), scope, "review", {"unapproved": True})
    with pytest.raises(ProviderError):
        BoundedLiveBudget(
            binding=b.model_copy(update={"deadline": b.deadline + timedelta(seconds=1)}),
            config=config,
            journal=RuntimeAgentJournal(docs),
            allocation=allocation,
        )
    from agents.planner.service import Planner
    from packages.planner_control.models import Command

    planner = Planner(runtime=docs.sessions, owner=None)
    state = planner.read(b.run_id)
    planner.step(b.run_id, Command(key="stop", expected_revision=state.revision, action="stop"))
    assert run_public_batches(config, again, docs, digest("round"), provider=provider) == q
    assert again.used == 2
