"""New selection protocol; original raw responses and admission rules remain intact."""

from copy import deepcopy

import pytest

from packages.grounding.models import canonical, digest
from packages.llm_provider import definition_passages as passages
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.transport import Reply
from tests.agent_provider.test_definition_feedback import setup
from tests.agent_provider.test_definition_identity import inputs
from tests.evaluation_preparation.test_owner_dependency_definition import confirmed


def selected(good, payload):
    value = deepcopy(good)
    value["schema_version"] = "research-definition-proposal/6.0.0"
    rows = passages.choices(payload)
    for draft in value["definitions"]:
        draft["schema_version"] = "source-definition-draft/7.2.0"
        draft["excerpts"] = [dict(passage_id=next(iter(rows)))]
    return value


def test_lossless_unicode_and_multichunk_without_quote_retyping():
    text = ("Table\u00a01. Exact–text\n" * 700) + "tail"
    payload = dict(
        target="GOAL:X", source_materials=[dict(code="s", checksum=digest(text), text=text)]
    )
    before = deepcopy(payload)
    rows = list(passages.choices(payload).values())
    assert len(rows) > 1 and "".join(r["text"] for r in rows) == text
    assert all(text[r["start"] : r["end"]] == r["text"] for r in rows)
    wire = passages.project(payload)
    assert payload == before and "text" not in wire["source_materials"][0]
    assert len(wire["definition_passage_contract"]["passages"]) == len(rows)


def test_original_provider_capture_owner_and_cold_read(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt=passages.PROMPT)
    owner, candidate, _, previous, _ = confirmed(tmp_path / "owner")
    good["definitions"] = [candidate]
    payload = inputs(good)
    raw = selected(good, payload)
    original = canonical(raw)
    e.policy = type("Policy", (), {"prompt_version": passages.PROMPT})()
    result = DeterministicFakeProvider(config, [raw]).generate_definitions(
        payload=payload,
        scope=e.scope,
        budget=e.budget,
        before_request=lambda: None,
        capture=lambda record, reply: e.capture(record, reply, payload=payload),
    )
    assert canonical(raw) == original and e.budget.used == 1
    saved = e.documents.read(e.attempt_key("definition:confirmed-proposal"))
    assert saved and digest(saved[0]) == digest(result)
    receipt = owner.accept("selection", result.definitions[0])
    assert owner.read("selection") == receipt
    assert owner.read("previous", historical=True) == previous
    proofs = [
        v
        for v in e.budget.journal.read()
        if v.get("schema_version") == "definition-passage-resolution/1.0.0"
    ]
    assert len(proofs) == 1 and not proofs[0]["medically_accepted"]
    assert proofs[0]["selection_checksum"] == digest(raw)


@pytest.mark.parametrize("fault", ["id", "source", "target", "duplicate", "text", "old"])
def test_wrong_selection_never_silently_repaired(tmp_path, fault):
    e, config, good, _ = setup(tmp_path, prompt=passages.PROMPT)
    payload = inputs(good)
    raw = selected(good, payload)
    quote = raw["definitions"][0]["excerpts"][0]
    if fault == "id":
        quote["passage_id"] = "passage:unknown"
    elif fault == "source":
        payload["source_materials"][0]["checksum"] = digest("changed")
    elif fault == "target":
        raw["definitions"][0]["target"] = "GOAL:OTHER"
    elif fault == "duplicate":
        raw["definitions"][0]["excerpts"].append(deepcopy(quote))
    elif fault == "text":
        quote["text"] = "caller-added quotation"
    else:
        raw = good
    before = deepcopy(raw)
    with pytest.raises(ProviderError, match="schema"):
        DeterministicFakeProvider(config, [raw]).generate_definitions(
            payload=payload,
            scope=e.scope,
            budget=e.budget,
            before_request=lambda: None,
        )
    assert raw == before and e.budget.used == 1


def test_confirmed_empty_and_unknown_are_not_forced_or_replayed(tmp_path):
    e, config, good, _ = setup(tmp_path, prompt=passages.PROMPT)
    payload = inputs(good)
    empty = dict(
        schema_version="research-definition-proposal/6.0.0",
        definitions=[],
        no_addition_reason="No new supported relation.",
    )
    result = DeterministicFakeProvider(config, [empty]).generate_definitions(
        payload=payload,
        scope=e.scope,
        budget=e.budget,
        before_request=lambda: None,
    )
    assert not result.definitions and e.budget.used == 1
    e, config, good, _ = setup(tmp_path / "unknown", prompt=passages.PROMPT)
    with pytest.raises(ProviderError, match="timeout"):
        DeterministicFakeProvider(
            config, [Reply(error="timeout", response_confirmed=False)]
        ).generate_definitions(
            payload=payload,
            scope=e.scope,
            budget=e.budget,
            before_request=lambda: None,
        )
    assert e.budget.used == 1
