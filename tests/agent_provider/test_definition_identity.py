"""Request binding feedback never repairs or accepts medical source candidates."""

import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical
from packages.llm_provider.definition_identity import failures, project
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.transport import Reply
from tests.agent_provider.test_definition_feedback import setup


def inputs(good):
    draft = good["definitions"][0]
    materials = {}
    for excerpt in draft["excerpts"]:
        materials.setdefault(
            excerpt["source_code"],
            dict(
                code=excerpt["source_code"],
                checksum=excerpt["source_checksum"],
                text=excerpt["text"],
            ),
        )
    return dict(
        target=draft["target"],
        source_materials=list(materials.values()),
        public_queries=["Different public retrieval wording"],
    )


def changed(good, field):
    bad = copy.deepcopy(good)
    draft = bad["definitions"][0]
    if field == "target":
        draft[field] = "Different public retrieval wording"
    else:
        draft["excerpts"][0][field] = "not-the-request-value"
    return bad


@pytest.mark.parametrize("field", ["target", "source_code", "source_checksum", "text"])
def test_request_binding_rejected_then_one_feedback_without_response_cleanup(tmp_path, field):
    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.6.0")
    payload = inputs(good)
    bad = changed(good, field)
    raw = canonical(bad)
    response = Reply(content=raw, response_confirmed=True, finish_reason="stop", api_requests=1)

    def next_response(body):
        feedback = json.loads(body["messages"][-1]["content"])
        assert feedback["errors"][0]["path"][-1] == field
        original = json.loads(body["messages"][1]["content"][0]["text"])
        assert original["definition_identity_contract"]["target"] == payload["target"]
        return good

    result = DeterministicFakeProvider(config, [response, next_response]).generate_definitions(
        payload=payload, scope=e.scope, budget=e.budget, before_request=lambda: None
    )
    assert result.definitions[0].target == payload["target"]
    assert response.content == raw and bad == json.loads(raw)
    assert e.budget.used == 2 and [r.error for r in e.budget.records()] == ["schema", None]


def test_old_prompt_retains_old_binding_boundary(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.4.0")
    bad = changed(good, "target")
    result = DeterministicFakeProvider(config, [bad]).generate_definitions(
        payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
    )
    assert result.definitions[0].target == bad["definitions"][0]["target"]
    assert e.budget.used == 1  # Original domain owner still rejects the wrong target.


def test_second_mismatch_stops_and_empty_proposal_is_permitted(tmp_path):
    e, config, good, _ = setup(tmp_path / "stop", prompt="definition-draft-6.6.0")
    bad = Reply(
        content=canonical(changed(good, "target")),
        response_confirmed=True,
        finish_reason="stop",
        api_requests=1,
    )
    with pytest.raises(ProviderError, match="schema"):
        DeterministicFakeProvider(config, [bad, bad, good]).generate_definitions(
            payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
        )
    assert e.budget.used == 2
    e, config, good, _ = setup(tmp_path / "empty", prompt="definition-draft-6.6.0")
    empty = dict(
        schema_version="research-definition-proposal/5.0.0",
        definitions=[],
        no_addition_reason="Material does not support an applicable relation.",
    )
    result = DeterministicFakeProvider(
        config,
        [
            Reply(
                content=canonical(changed(good, "target")),
                response_confirmed=True,
                finish_reason="stop",
                api_requests=1,
            ),
            empty,
        ],
    ).generate_definitions(
        payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
    )
    assert not result.definitions and e.budget.used == 2


def test_dynamic_wire_binding_is_copy_only_not_medical_acceptance(tmp_path):
    _, _, good, _ = setup(tmp_path, prompt="definition-draft-6.6.0")
    payload = inputs(good)
    schema = {
        "$defs": {
            "Draft": {"properties": {"target": {"type": "string"}}},
            "Quote": {"properties": {"source_code": {}, "source_checksum": {}}},
        }
    }
    before = copy.deepcopy((payload, schema, good))
    projected, wire = project(payload, schema)
    assert wire["$defs"]["Draft"]["properties"]["target"]["const"] == payload["target"]
    assert (
        projected["definition_identity_contract"]["source_identities"][0]["source_checksum"]
        == payload["source_materials"][0]["checksum"]
    )
    assert not failures(good, payload)
    assert (payload, schema, good) == before
    assert "medical entailment" in projected["definition_identity_contract"]["meaning"]


def test_unknown_does_not_get_identity_feedback(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.6.0")
    with pytest.raises(ProviderError, match="timeout"):
        DeterministicFakeProvider(
            config, [Reply(error="timeout", response_confirmed=False), good]
        ).generate_definitions(
            payload=inputs(good), scope=e.scope, budget=e.budget, before_request=lambda: None
        )
    assert e.budget.used == 1


def test_real_http_parser_original_journal_binding_feedback(tmp_path, monkeypatch):
    from packages.llm_provider.provider import RequestBudget, StructuredProvider
    from packages.llm_provider.review_correction import verify_correction
    from packages.llm_provider.transport import HTTPTransport
    from packages.research_runtime.agent_records import RuntimeAgentJournal
    from tests.agent_provider.test_http_recovery import injected_worker

    e, config, good, _ = setup(tmp_path, prompt="definition-draft-6.6.0")
    config = config.model_copy(update={"mode": "live"})
    docs = e.documents
    docs.run_id = e.scope.run_id
    deadline = e.budget.deadline
    docs.sessions = SimpleNamespace(origin=lambda run: (deadline - timedelta(hours=1), None))
    budget = RequestBudget(
        run_id=e.scope.run_id, deadline=deadline, max_requests=4, journal=RuntimeAgentJournal(docs)
    )
    calls = injected_worker(monkeypatch, [changed(good, "target"), good])
    result = StructuredProvider(
        config, live_opt_in=True, transport=HTTPTransport()
    ).generate_definitions(
        payload=inputs(good), scope=e.scope, budget=budget, before_request=lambda: None
    )
    assert result.definitions and len(calls) == budget.used == 2
    assert [r.error for r in budget.records()] == ["schema", None]
    verify_correction(budget.records()[1], budget)
    cold = RequestBudget(
        run_id=e.scope.run_id, deadline=deadline, max_requests=4, journal=RuntimeAgentJournal(docs)
    )
    assert cold.used == 2  # Injected HTTP worker, not a real API or accepted clinical result.
