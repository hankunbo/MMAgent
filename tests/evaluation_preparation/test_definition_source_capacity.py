"""Versioned Source output capacity, without enlarging Review or request budgets."""

import json
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign_renewal import compact_configuration
from packages.multimodal_research.definition_expansion import (
    definition_config,
    read_expansion_policy,
)


def test_new_source_capacity_is_explicit_and_old_profiles_unchanged(monkeypatch):
    from packages.multimodal_research import definition_trial as t

    config = compact_configuration()
    policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v9.json").read_bytes())
    )
    old = definition_config(config, prompt_version="definition-draft-3.0.0")
    new = definition_config(
        config,
        prompt_version=policy.prompt_version,
        source_output_tokens=policy.source_output_tokens,
    )
    roles = {r.role: r for r in new.roles}
    assert roles["source"].max_output_tokens == 8192
    assert next(r for r in old.roles if r.role == "source").max_output_tokens == 2048
    assert roles["review"].max_output_tokens == 2048
    assert new.transport == old.transport and new.transport.max_retries == 0
    assert all(r == roles[r.role] for r in old.roles if r.role != "source")
    from packages.evidence_reuse.definitions import ScopedDefinitionProposal
    from packages.llm_provider.config import prompt
    from packages.llm_provider.transport import wire_request

    body = wire_request(
        new.transport,
        roles["source"],
        prompt(new, "source")[1],
        {"synthetic": True},
        ScopedDefinitionProposal.model_json_schema(),
        (),
        wire_policy=new.wire_policy,
        wire_mode=new.wire_mode,
    )
    assert body["max_completion_tokens"] == 8192
    assert body["response_format"]["type"] == "json_object"
    monkeypatch.setattr(t, "freeze", lambda: {"test": "freeze"})
    spec = dict(
        implementation=t.freeze(),
        schema_version="multimodal-live-entry/1.13.0",
        intent=dict(target_code=policy.public_target),
    )
    trial = t.CapacityTrial(
        entry_checksum=digest(spec),
        authorization="offline capacity contract test",
        expansion_policy=policy.model_dump(mode="json"),
        max_new_requests=7,
        seconds=1800,
    ).model_dump(mode="json")
    assert t.checked(trial, spec) == trial
    trial["schema_version"] = "multimodal-definition-trial/1.3.0"
    with pytest.raises(ValueError, match="definition_trial_policy"):
        t.checked(trial, spec)


@pytest.mark.parametrize("tokens", [0, 2048, 4096, 16385])
def test_capacity_cannot_silently_reconfigure_legacy(tokens):
    with pytest.raises(ValueError, match="definition_source_capacity"):
        definition_config(compact_configuration(), source_output_tokens=tokens)


def test_truncated_json_remains_refused_even_if_partial_looks_complete():
    from packages.llm_provider.transport import parse_reply

    reply = parse_reply(
        json.dumps(
            dict(
                choices=[dict(finish_reason="length", message=dict(content='{"definitions":[]}'))],
                usage=dict(prompt_tokens=100, completion_tokens=2048, total_tokens=2148),
            )
        ).encode(),
        "chat_completions",
        retain_metadata=True,
    )
    assert reply.error == "truncated" and reply.finish_reason == "length"
    assert reply.response_confirmed and reply.usage.output_tokens == 2048
    assert reply.content == b'{"definitions":[]}'
