"""Original Planner + Runtime journal, synthetic transport only; no patient or API use."""

import json

import pytest

from agents.planner.service import Planner
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_probes import run_probes
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import Reply
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.support import review_from_wire
from tests.agent_provider.test_bounded_live import fixture, generate
from tests.planner.support import FakeOwner, advance, command


def controlled(replies=()):
    f = fixture(replies)
    runtime = f.docs.sessions
    runtime.origin = lambda _: (f.binding.started_at, f.binding.run_context_checksum)
    f.budget = BoundedLiveBudget(
        binding=f.binding, config=f.config, journal=RuntimeAgentJournal(f.docs)
    )
    f.planner = Planner(runtime=runtime, owner=FakeOwner(f.binding))
    return f


def stop(f, action="stop"):
    f.planner.step(f.binding.run_id, command(f.planner, action))


def bad_source(_):
    return dict(
        scope_checksum=digest("wrong"),
        source_code="SOURCE:PUBLIC_ARITHMETIC",
        quote="Two plus two equals four.",
        target="GOAL:PUBLIC_SUM",
        relation="requires",
        missing=[],
        reason="Synthetic counterexample",
    )


def no_acceptance(f):
    assert not any(
        v.get("status") == "accepted" for values in f.docs.saved.values() for v in values
    )


@pytest.mark.parametrize(
    "mode,expected", [("active", 2), ("already_stopped", 0), ("stop_during_catalog", 1)]
)
def test_independent_counterexamples_and_active_control(mode, expected):
    def catalog(_):
        if mode == "stop_during_catalog":
            stop(f)
        return dict(model_listed=True)

    f = controlled([catalog, bad_source])
    if mode == "already_stopped":
        stop(f)
    with pytest.raises(ProviderError, match="scope" if mode == "active" else "stopped"):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.budget.used == len(f.wire.calls) == expected
    if mode != "active":
        assert not f.docs.read("probe:text:claim")
    if expected == 1:
        assert f.docs.read("probe:models:result")[0]["status"] == "passed"
        assert f.budget.transport_counts()["remote_response_confirmed"] == 1
    saved = f.budget.journal.read()
    f.budget = BoundedLiveBudget(
        binding=f.binding, config=f.config, journal=RuntimeAgentJournal(f.docs)
    )
    with pytest.raises(ProviderError):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.budget.journal.read() == saved
    assert len(f.wire.calls) == expected
    no_acceptance(f)


@pytest.mark.parametrize("mode", ["cancel", "deadline", "missing", "unreadable", "pending"])
def test_current_owner_refusal_never_claims_or_reserves(mode, monkeypatch):
    f = controlled([dict(model_listed=True)])
    if mode == "cancel":
        stop(f, "cancel")
    elif mode == "deadline":
        f.budget.clock = lambda: f.binding.deadline
    elif mode == "missing":
        monkeypatch.setattr(f.docs.sessions, "read", lambda _: None)
    elif mode == "unreadable":

        def fail(_):
            raise OSError("synthetic_owner_io")

        monkeypatch.setattr(f.docs.sessions, "read", fail)
    else:
        advance(f.planner)
        f.planner.owner.failure = "execute"
        f.planner.step(f.binding.run_id, command(f.planner, "execute"))
        assert f.planner.read(f.binding.run_id).pending.launch_claimed
    before = dict(f.docs.saved)
    with pytest.raises((ValueError, OSError)):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.docs.saved == before and f.budget.used == 0 and not f.wire.calls
    no_acceptance(f)


def test_stop_before_internal_retry_preserves_response_usage():
    f = controlled()

    def rate_limit(_):
        stop(f)
        from packages.llm_provider.models import Usage

        return Reply(
            error="rate_limit",
            api_requests=1,
            response_confirmed=True,
            usage=Usage(status="observed", total_tokens=7),
        )

    f.wire.replies = [rate_limit, review_from_wire]
    with pytest.raises(ProviderError, match="stopped"):
        generate(f)
    assert f.budget.used == len(f.wire.calls) == 1
    assert f.budget.records()[0].usage.total_tokens == 7
    assert f.budget.records()[0].error == "rate_limit"
    no_acceptance(f)


def test_stop_on_success_keeps_raw_response_but_cannot_accept():
    f = controlled()

    def response(body):
        answer = review_from_wire(body)
        stop(f)
        return answer

    f.wire.replies = [response]
    with pytest.raises(ProviderError, match="stopped"):
        generate(f)
    record = f.budget.records()[0]
    assert record.error is None and record.api_requests == 1
    assert f.docs.read("live-response:" + record.request_id)[0]["raw_model_text"]
    assert f.budget.used == len(f.wire.calls) == 1
    no_acceptance(f)


def test_stop_after_submission_denies_acceptance_and_keeps_prior_facts():
    f = fixture()
    f.planner = Planner(runtime=f.docs.sessions, owner=FakeOwner(f.binding))
    sub = generate(f)
    acceptance = ResearchAcceptance(
        owner=f.reader, documents=f.docs, requests=f.budget, live_permission=f.permission
    )
    assert acceptance.submit(sub, key="candidate")["status"] == "submitted_not_accepted"
    stop(f)
    assert acceptance.accept(key="candidate").status == "rejected"
    assert acceptance.submit(sub, key="candidate")["status"] == "submitted_not_accepted"
    assert f.budget.used == len(f.wire.calls) == 1
    no_acceptance(f)


def test_saved_scope_failure_is_not_relabelled_schema_or_passed():
    f = controlled([dict(model_listed=True), bad_source])
    with pytest.raises(ProviderError, match="scope"):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    stop(f)
    before = json.dumps(f.docs.saved, sort_keys=True)
    with pytest.raises(ProviderError, match="scope"):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert json.dumps(f.docs.saved, sort_keys=True) == before
    assert f.budget.used == len(f.wire.calls) == 2


def test_stop_between_reservation_and_dispatch_never_sends(monkeypatch):
    f = controlled([review_from_wire])
    reserve = f.budget.reserve

    def stopped_reservation(*args):
        result = reserve(*args)
        stop(f)
        return result

    monkeypatch.setattr(f.budget, "reserve", stopped_reservation)
    with pytest.raises(ProviderError, match="stopped"):
        generate(f)
    assert not f.wire.calls and f.budget.used == 1
    assert f.budget.records()[0].api_requests == 0
    assert f.budget.summary().pending_unknown == 0


def test_stopped_execute_reentry_does_not_reassemble_or_create_run(monkeypatch, tmp_path):
    from apps.vectorengine_live_assembly import execute
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    f = controlled()
    f.reader.state = ready_entry_state(f.reader.state)
    stop(f)
    before = dict(f.docs.saved)
    monkeypatch.setattr("apps.vectorengine_live_assembly.check_regression_gate", lambda: None)
    monkeypatch.setattr(
        "db.runtime.repository.planner.PlannerRepository", lambda _: f.docs.sessions
    )
    # Accessing any context method would fail: reentry must stop before run creation/assembly.
    monkeypatch.setattr("db.runtime.repository.research.ResearchRunRepository", lambda _: object())
    with pytest.raises(ValueError, match="stopped"):
        execute(
            evidence=None, target=None, initial=f.reader.state, config=f.config, output=tmp_path
        )
    assert f.docs.saved == before and f.budget.used == 0 and not f.wire.calls
