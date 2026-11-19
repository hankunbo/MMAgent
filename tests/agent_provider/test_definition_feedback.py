"""Bounded Source schema feedback; no remote/unknown replay or candidate cleanup."""

import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical
from packages.llm_provider.config import review_format_upgrade
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.transport import Reply
from packages.multimodal_research.definition_expansion import definition_config
from tests.evaluation_preparation.test_confirmed_interpretation_definition import (
    exercise_source_proposal,
)
from tests.evaluation_preparation.test_definition_expansion import fixture
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed


def setup(tmp_path, *, prompt="definition-draft-6.5.0"):
    e = fixture(tmp_path)
    e.scope = e.scope.model_copy(update={"case_code": "CASE:SYNTHETIC_DEFINITION_FEEDBACK"})
    config = definition_config(review_format_upgrade(e.provider.config), prompt_version=prompt)
    value = confirmed(tmp_path / "candidate")[1]
    good = dict(
        schema_version="research-definition-proposal/5.0.0",
        definitions=[value],
        no_addition_reason=None,
    )
    bad = copy.deepcopy(good)
    bad["definitions"][0]["dependency_contracts"][0]["input_alias"] = "input:0"
    return (
        e,
        config,
        good,
        Reply(
            content=canonical(bad), finish_reason="stop", response_confirmed=True, api_requests=1
        ),
    )


def generate(e, config, replies):
    return DeterministicFakeProvider(config, replies).generate_definitions(
        payload={}, scope=e.scope, budget=e.budget, before_request=lambda: None
    )


def test_corrected_nonempty_proposal_original_owner_revision_and_cold_no_dispatch(tmp_path):
    exercise_source_proposal(tmp_path, typed=True, candidate_choices=True, feedback=True)


@pytest.mark.parametrize(
    "prompt,requests", [("definition-draft-6.4.0", 1), ("definition-draft-6.5.0", 2)]
)
def test_old_no_retry_and_new_second_bad_answer_stop(tmp_path, prompt, requests):
    e, config, good, bad = setup(tmp_path, prompt=prompt)
    with pytest.raises(ProviderError, match="schema"):
        generate(e, config, [bad, bad, good])
    assert e.budget.used == requests
    assert all(r.error == "schema" for r in e.budget.records())


@pytest.mark.parametrize("error", ["timeout", "transport", "rate_limit"])
def test_unknown_or_non_schema_does_not_get_format_retry(tmp_path, error):
    e, config, good, _ = setup(tmp_path)
    with pytest.raises(ProviderError, match=error):
        generate(e, config, [Reply(error=error, response_confirmed=False), good])
    assert e.budget.used == 1


def test_empty_corrected_answer_is_valid_not_forced_nonempty(tmp_path):
    e, config, _, bad = setup(tmp_path)
    result = generate(
        e,
        config,
        [
            bad,
            dict(
                schema_version="research-definition-proposal/5.0.0",
                definitions=[],
                no_addition_reason="No supported expressible definition in the material.",
            ),
        ],
    )
    assert not result.definitions and e.budget.used == 2


def test_budget_exhaustion_preserves_first_failure_no_second_dispatch(tmp_path):
    e, config, good, bad = setup(tmp_path)
    e.budget.max_requests = 1
    with pytest.raises(ProviderError, match="budget"):
        generate(e, config, [bad, good])
    assert e.budget.used == 1 and e.budget.records()[0].error == "schema"


def test_feedback_is_format_only_not_quote_selection_or_rewritten_answer(tmp_path):
    e, config, good, bad = setup(tmp_path)
    before = bad.content

    def response(body):
        extra = json.loads(body["messages"][-1]["content"])
        assert "SOURCE_LITERAL_CHOICES" not in str(extra)
        assert "no medical answer" in extra["constraints"]
        assert "empty definitions array" in extra["constraints"]
        return good

    assert generate(e, config, [bad, response]).definitions
    assert bad.content == before and e.budget.used == 2


def test_real_http_parser_and_original_journal_verify_correction_without_socket(
    tmp_path, monkeypatch
):
    from packages.llm_provider.provider import RequestBudget, StructuredProvider
    from packages.llm_provider.review_correction import verify_correction
    from packages.llm_provider.transport import HTTPTransport
    from packages.research_runtime.agent_records import RuntimeAgentJournal
    from tests.agent_provider.test_http_recovery import injected_worker

    e, config, good, bad = setup(tmp_path)
    config = config.model_copy(update={"mode": "live"})
    docs = e.documents
    docs.run_id = e.scope.run_id
    deadline = e.budget.deadline
    docs.sessions = SimpleNamespace(origin=lambda run: (deadline - timedelta(hours=1), None))
    journal = RuntimeAgentJournal(docs)
    budget = RequestBudget(
        run_id=e.scope.run_id, deadline=deadline, max_requests=4, journal=journal
    )
    calls = injected_worker(monkeypatch, [json.loads(bad.content), good])
    provider = StructuredProvider(config, live_opt_in=True, transport=HTTPTransport())
    result = provider.generate_definitions(
        payload={}, scope=e.scope, budget=budget, before_request=lambda: None
    )
    assert result.definitions and len(calls) == budget.used == 2
    records = budget.records()
    assert [r.error for r in records] == ["schema", None]
    assert all(not r.simulated for r in records)  # HTTP parser path; injected worker, NOT real API.
    verify_correction(records[1], budget)
    assert (
        RequestBudget(
            run_id=e.scope.run_id,
            deadline=deadline,
            max_requests=4,
            journal=RuntimeAgentJournal(docs),
        ).used
        == 2
    )


@pytest.mark.parametrize("mode", ["public", "cancel", "extra_limit"])
def test_no_feedback_when_control_or_public_probe_forbids_it(tmp_path, mode):
    e, config, good, bad = setup(tmp_path)
    if mode == "public":
        e.scope = e.scope.model_copy(update={"case_code": "CASE:PUBLIC_PROBE"})
    elif mode == "extra_limit":
        # Existing shared extra-request cap is enforced before reserve/send.
        e.budget.records = lambda: [SimpleNamespace(retry_index=1)] * config.max_extra_requests
    else:

        def cancelled():
            if e.budget.used:
                raise ProviderError("cancelled")

        e.budget.check_control = cancelled
    expected = {"public": "schema", "cancel": "cancelled", "extra_limit": "budget"}[mode]
    with pytest.raises(ProviderError, match=expected):
        generate(e, config, [bad, good])
    assert e.budget.used == 1
