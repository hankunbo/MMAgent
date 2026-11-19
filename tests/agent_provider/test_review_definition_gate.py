"""Definition authority projected to wire; no medical answer is manufactured."""

import copy
import json

import pytest

from packages.evidence_details.review_batches import check_batch
from packages.llm_provider.review_probe import build
from tests.evaluation_preparation.test_directional_review import reply


def prepared(version):
    return build(
        "gpt-6-sol",
        "f" * 32,
        "triple-expanded",
        0,
        provider_config=f"configs/agents/multimodal_relay_local_gpt6_review_v{version}.json",
    )


def table(body):
    return json.loads(
        next(
            p["text"].split("\n", 1)[1]
            for p in body["messages"][1]["content"]
            if p["text"].startswith("REVIEW_CITATION_CODES:")
        )
    )


def test_current_wire_definition_gate_and_old_wire_retained():
    old = prepared(1)
    new = prepared(2)
    rows = {r["assessment_id"]: r for r in table(new[3])["tasks"]}
    assert rows["probe:task00"]["assessed_permitted_by_definition"] is True
    assert "assessed" in rows["probe:task00"]["allowed_primary_statuses"]
    assert rows["probe:task04"]["interpretation_status"] == "missing"
    assert rows["probe:task04"]["assessed_permitted_by_definition"] is False
    assert "assessed" not in rows["probe:task04"]["allowed_primary_statuses"]
    assert "reference_missing" in rows["probe:task04"]["allowed_primary_statuses"]
    assert all("assessed_permitted_by_definition" not in r for r in table(old[3])["tasks"])
    schema = json.loads(
        next(
            p["text"].split("\n", 1)[1]
            for p in new[3]["messages"][1]["content"]
            if p["text"].startswith("OUTPUT_JSON_SCHEMA:")
        )
    )
    for v in schema["properties"]["items"]["items"]["anyOf"]:
        p = v["properties"]
        assert (
            p["status"]["enum"] == rows[p["assessment_id"]["enum"][0]]["allowed_primary_statuses"]
        )


def test_original_owner_accepts_valid_missing_and_rejects_forged_assessed():
    _, state, plan, body, _ = prepared(2)
    candidate = reply(body)
    check_batch(state, plan, plan.batches[0], candidate)
    bad = copy.deepcopy(candidate)
    source = next(i for i in bad["items"] if i["assessment_id"] == "probe:task00")
    index = next(n for n, i in enumerate(bad["items"]) if i["assessment_id"] == "probe:task04")
    bad["items"][index] = {
        **source,
        "assessment_id": "probe:task04",
        "observation_citations": ["OBS:PUBLIC_04"],
    }
    with pytest.raises(ValueError, match="review_applicability"):
        check_batch(state, plan, plan.batches[0], bad)


def test_nonapplicable_definition_also_excludes_assessed():
    from packages.evidence_details.review_batches import batch_payload, proposal_type
    from packages.llm_provider.review_citations import project

    config, state, plan, _, _ = prepared(2)
    payload = batch_payload(state, plan, plan.batches[0])
    payload["review_tasks"]["tasks"][0]["applicable"] = False
    schema = proposal_type(config).model_json_schema()
    result = project(
        schema, payload, primary_status=True, exact_primary_status=True, definition_gate=True
    )
    row = json.loads(result.split("\n", 1)[1])["tasks"][0]
    assert row["owner_applicable"] is False
    assert "assessed" not in row["allowed_primary_statuses"]
