"""Owner-produced identity literals, without repairing any model response."""

import json

import pytest

from packages.grounding.models import digest
from packages.llm_provider.review_probe import build


@pytest.mark.parametrize("rung", ["single-small", "single-expanded", "triple-expanded"])
def test_actual_wire_exposes_exact_owner_bindings(rung):
    _, state, plan, body, _ = build(
        "gpt-6-sol",
        "a" * 32,
        rung,
        0,
        provider_config="configs/agents/multimodal_relay_local_gpt6_review_v4.json",
        directional_material=True,
    )
    parts = body["messages"][1]["content"]
    (header,) = [p["text"] for p in parts if p["text"].startswith("REVIEW_RESPONSE_BINDINGS:\n")]
    value = json.loads(header.split("\n", 1)[1])
    assert value == dict(
        scope_checksum=digest(state.material.scope),
        round="A",
        plan_checksum=digest(plan),
        batch_id=plan.batches[0].id,
    )
    schema = json.loads(parts[-1]["text"].split("\n", 1)[1])
    for key, literal in value.items():
        assert schema["properties"][key]["enum"] == [literal]
    assert "Do not calculate" in body["messages"][0]["content"]
    assert "rating-presence/1.0.0" in body["messages"][0]["content"]


def test_old_wire_stays_without_new_header():
    _, _, _, body, _ = build(
        "gpt-6-sol",
        "a" * 32,
        "triple-expanded",
        0,
        provider_config="configs/agents/multimodal_relay_local_gpt6_review_v3.json",
        directional_material=True,
    )
    assert not any(
        p["text"].startswith("REVIEW_RESPONSE_BINDINGS:\n") for p in body["messages"][1]["content"]
    )


def test_wrong_identity_still_rejected_and_original_correction_copies_only_bindings():
    from packages.grounding.models import canonical
    from packages.llm_provider.transport import Reply
    from tests.agent_provider.batch_support import generate
    from tests.agent_provider.test_review_rating_contract import configured
    from tests.evaluation_preparation.test_directional_review import reply

    def bad(body):
        value = reply(body)
        value["scope_checksum"] = "sha256:abc"
        return Reply(content=canonical(value), finish_reason="stop", response_confirmed=True)

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        original = json.loads(body["messages"][1]["content"][0]["text"])
        assert feedback["copy_binding_values_exactly"] == {
            "scope_checksum": digest(original["material"]["scope"])
        }
        return reply(body)

    f = configured([bad, corrected] + [reply] * 10, prompt_version="review-direction-2.10.0")
    sub = generate(f)
    assert f.budget.records()[0].error == "schema"
    assert f.budget.records()[1].retry_index == 1
    assert f.acceptance.submit(sub, key="binding-test")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="binding-test").status == "accepted"
