"""Acquisition status and an unavailable directional interpretation stay distinct."""

import copy
import json

from packages.grounding.models import digest
from tests.evaluation_preparation.test_directional_review import fixture


def test_new_wire_excludes_false_missing_and_ungradable_without_forcing_direction():
    from packages.evidence_details.review_batches import (
        ReasonedReviewBatch,
        batch_payload,
        make_plan,
    )
    from packages.llm_provider.config import prompt
    from packages.llm_provider.transport import wire_request
    from packages.multimodal_research.campaign import configuration

    state = fixture().owner.current()
    config = configuration(directional=True, exact_primary_status=True)
    old = configuration(directional=True)
    assert (
        next(r for r in old.roles if r.role == "review").prompt_version == "review-direction-2.0.0"
    )
    plan = make_plan(state, config.review_batching, key="status-wire")
    role, text = prompt(config, "review")
    payload = batch_payload(state, plan, plan.batches[0])
    before = copy.deepcopy(payload)
    body = wire_request(
        config.transport,
        role,
        text,
        payload,
        ReasonedReviewBatch.model_json_schema(),
        (),
        scope_binding=digest(state.material.scope),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )
    texts = [v["text"] for v in body["messages"][1]["content"]]
    table = json.loads(
        next(v.split("\n", 1)[1] for v in texts if v.startswith("REVIEW_CITATION_CODES:"))
    )
    for item in table["tasks"]:
        if item["required_observation_available"]:
            assert "observation_ungradable" not in item["allowed_primary_statuses"]
            assert "input_missing" not in item["allowed_primary_statuses"]
            assert {"assessed", "method_mismatch", "reference_missing"} <= set(
                item["allowed_primary_statuses"]
            )
    assert "e=null" in text and payload == before
