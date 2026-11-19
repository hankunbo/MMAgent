"""Qualified 2.3 explicitly restores the owner table; historic 2.2 stays byte-stable."""

import json

from packages.evidence_details.review_batches import batch_payload, make_plan, proposal_type
from packages.grounding.models import digest
from packages.llm_provider.config import prompt, qualified_citation_upgrade
from packages.llm_provider.public_shape import PublicReviewMaterials
from packages.llm_provider.transport import wire_request
from packages.multimodal_research.campaign import configuration


def body(config, state):
    plan = make_plan(state, config.review_batching, key="qualified-wire")
    role, text = prompt(config, "review")
    return wire_request(
        config.transport,
        role,
        text,
        batch_payload(state, plan, plan.batches[0]),
        proposal_type(config).model_json_schema(),
        (),
        scope_binding=digest(state.material.scope),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )


def test_new_qualified_wire_supplies_promised_owner_table_without_score_values():
    old = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=True,
    )
    before = old.model_dump(mode="json")
    new = qualified_citation_upgrade(old)
    state = PublicReviewMaterials("RUN:QUALIFIED_PUBLIC", old, task_count=1).current()
    original = body(old, state)
    current = body(new, state)
    assert old.model_dump(mode="json") == before
    assert original == body(old, state)
    assert not any(
        c["text"].startswith("REVIEW_CITATION_CODES:") for c in original["messages"][1]["content"]
    )
    texts = [c["text"] for c in current["messages"][1]["content"]]
    table = json.loads(
        next(t.split("\n", 1)[1] for t in texts if t.startswith("REVIEW_CITATION_CODES:"))
    )
    task = table["tasks"][0]
    assert task["assessment_id"] == "probe:task00"
    assert task["observation_citations"] == ["OBS:PUBLIC_00"]
    assert task["reference_citations"] == [
        r.code for r in state.policy.items[0].definition.source_refs
    ]
    assert task["required_observation_available"] is True
    assert "input_missing" not in task["allowed_primary_statuses"]
    assert "observation_ungradable" not in task["allowed_primary_statuses"]
    assert {"assessed", "method_mismatch", "reference_missing"} <= set(
        task["allowed_primary_statuses"]
    )
    assert "judgment" not in task
    assert current["response_format"] == {"type": "json_object"}
    assert current["max_completion_tokens"] == original["max_completion_tokens"]
    assert new.transport == old.transport
    assert [r for r in new.roles if r.role != "review"] == [
        r for r in old.roles if r.role != "review"
    ]
