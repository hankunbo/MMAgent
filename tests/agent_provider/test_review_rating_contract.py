"""Explicit Rating presence instructions; original scores and rejection stay strict."""

import copy
import json
from pathlib import Path

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.review_correction import corrected_wire, fields
from packages.llm_provider.review_probe import build
from packages.llm_provider.review_response import decode
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import generate
from tests.agent_provider.test_review_optional_null import upgraded
from tests.evaluation_preparation.test_directional_review import reply

PROMPT = "review-direction-2.9.0"


def configured(responses, *, prompt_version=PROMPT):
    f = upgraded(responses)
    raw = f.config.model_dump(mode="json")
    setting = next(r for r in raw["roles"] if r["role"] == "review")
    setting.update(
        prompt_version=prompt_version,
        prompt_checksum=digest(
            Path("configs/prompts/agents/" + prompt_version + ".txt").read_text()
        ),
    )
    f.config = validate(type(f.config), raw)
    state = f.owner.state.model_dump(mode="json")
    state.update(config_checksum=digest(f.config), prompt_checksum=setting["prompt_checksum"])
    f.owner.state = validate(type(f.owner.state), state)
    f.provider = f.agent.provider = DeterministicFakeProvider(f.config, responses)
    return f


@pytest.mark.parametrize("section", ["observation", "relation", "applicability"])
@pytest.mark.parametrize("kind", ["numeric_and_missing", "null_without_reason"])
def test_once_only_rating_feedback_then_original_acceptance(section, kind):
    def bad(body):
        raw = reply(body)
        row = raw["items"][0][section]
        if kind == "numeric_and_missing":
            row["missing_reason"] = "Synthetic contradictory presence"
        else:
            row.update(value=None, missing_reason=None)
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        assert feedback["errors"] == [
            {
                "path": ["items", 0, section],
                "type": "value_error",
                "constraint": "score_value_missing_reason_conflict",
            }
        ]
        contract = feedback["rating_presence_contract"]
        assert "value=null" in contract and "missing_reason=null" in contract
        assert "Do not invent" in contract and "not a required score" in contract
        return reply(body)

    f = configured([bad, corrected] + [reply] * 10)
    sub = generate(f)
    assert f.budget.records()[0].error == "schema"
    assert f.budget.records()[1].retry_index == 1
    failed = [
        v[0]
        for k, v in f.documents.saved.items()
        if ":attempt:" in k and v[0].get("error") == "schema"
    ]
    assert len(failed) == 1
    original = json.loads(failed[0]["raw_model_text"])["items"][0][section]
    assert (
        original["missing_reason"] is not None
        if kind == "numeric_and_missing"
        else original["value"] is None
    )
    f.acceptance.submit(sub, key="rating-presence")
    assert f.acceptance.accept(key="rating-presence").status == "accepted"
    used = f.budget.used
    assert generate(f) == sub and f.budget.used == used


def test_rating_feedback_opt_in_preserves_old_error_and_wire():
    from packages.evidence_scoring.models import Rating

    with pytest.raises(ValueError) as exc:
        Rating(value=0.3, reason="Synthetic", missing_reason="Contradiction")
    assert fields(exc.value) == [{"path": [], "type": "value_error"}]
    detail = fields(exc.value, rating_constraints=True)
    assert detail == [
        {"path": [], "type": "value_error", "constraint": "score_value_missing_reason_conflict"}
    ]
    body = {"messages": [{"role": "system", "content": "synthetic original schema"}]}
    old = [{"path": ["items", 0, "judgment"], "type": "value_error"}]
    assert (
        digest(corrected_wire(body, old))
        == "sha256:53601bdb047ebd90f7e4203229fb048d6be3363065f6c1c667df188027dd258f"
    )


def test_real_wire_keeps_owner_definition_gate_and_explicit_rating_presence():
    old = build(
        "gpt-6-sol",
        "c" * 32,
        "triple-expanded",
        0,
        provider_config="configs/agents/multimodal_relay_local_gpt6_review_v2.json",
    )
    new = build(
        "gpt-6-sol",
        "c" * 32,
        "triple-expanded",
        0,
        provider_config="configs/agents/multimodal_relay_local_gpt6_review_v3.json",
    )
    raw = new[0].model_dump(mode="json")
    next(r for r in raw["roles"] if r["role"] == "review").update(
        **next(r.model_dump(mode="json") for r in old[0].roles if r.role == "review")
    )
    assert raw == old[0].model_dump(mode="json")
    # A new Prompt must produce a new plan checksum; retain all material and gates.
    old_text = copy.deepcopy(old[3]["messages"][1:])
    new_text = copy.deepcopy(new[3]["messages"][1:])
    for text in (old_text, new_text):
        content = text[0]["content"]
        payload = json.loads(content[0]["text"])
        payload["batch"]["plan_checksum"] = "new-plan-identity-compared-separately"
        content[0]["text"] = canonical(payload).decode()
        schema = json.loads(content[-1]["text"].split("\n", 1)[1])
        schema["properties"]["plan_checksum"]["enum"] = ["new-plan-identity-compared-separately"]
        content[-1]["text"] = canonical(schema).decode()
    assert new_text == old_text
    assert digest(new[2]) != digest(old[2])
    assert "rating-presence/1.0.0" in new[3]["messages"][0]["content"]
    assert "missing_reason=null" in new[3]["messages"][0]["content"]
    candidate = reply(new[3])
    damaged = copy.deepcopy(candidate)
    damaged["items"][0]["relation"]["missing_reason"] = "not allowed alongside a number"
    with pytest.raises(ValueError, match="score_value_missing_reason_conflict"):
        decode(canonical(damaged), new[0])


def test_second_rating_error_remains_failure_not_loop_or_value_repair():
    def bad(body):
        raw = reply(body)
        raw["items"][0]["relation"]["missing_reason"] = "Synthetic contradictory presence"
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    f = configured([bad, bad, reply])
    with pytest.raises(ProviderError, match="schema"):
        generate(f)
    assert f.budget.used == 2


def test_unknown_never_triggers_rating_correction():
    f = configured([Reply(error="transport", response_confirmed=None), reply])
    with pytest.raises(ProviderError, match="transport"):
        generate(f)
    assert f.budget.used == 1
