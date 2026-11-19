"""Explicit, lossless explanation recovery; never relax default JSON parsing."""

import json

import pytest

from packages.evidence_details.recorded_explanation import interpret
from packages.grounding.models import digest
from packages.llm_provider.review_response import decode
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_directional_review import fixture


@pytest.fixture
def response():
    f = fixture()
    sub = generate(f)
    return f.config, sub.batches[0].candidate.model_dump(mode="json")


def repeated(raw, field, value):
    token = json.dumps(field) + ": " + json.dumps(value)
    text = json.dumps(raw)
    assert token in text
    return text.replace(token, token + ", " + token, 1)


@pytest.mark.parametrize("field", ["quality_limitation", "resolvable_gap", "next_action"])
def test_explicit_identical_explanation_keeps_original_scores_and_default_rejection(
    response, field
):
    config, raw = response
    text = repeated(raw, field, raw["items"][0]["direction_assessment"][field])
    with pytest.raises(ValueError, match="duplicate_json_key"):
        decode(text, config)
    candidate, proof = interpret(text, config)
    original, _ = decode(json.dumps(raw), config)
    assert candidate == original
    assert proof.candidate_checksum == digest(original)
    assert [(r.item_index, r.field) for r in proof.repeated_fields] == [(0, field)]
    assert proof.scores_citations_status_unchanged


@pytest.mark.parametrize(
    "field", ["value", "assessment_id", "scope_checksum", "status", "observation_citations"]
)
def test_even_equal_score_citation_identity_duplicates_are_rejected(response, field):
    config, raw = response
    obj = raw if field == "scope_checksum" else raw["items"][0]
    if field == "value":
        obj = obj["judgment"]
    text = repeated(raw, field, obj[field])
    with pytest.raises(ValueError, match="forbidden_duplicate"):
        interpret(text, config)


def test_conflicting_explanations_cannot_be_chosen(response):
    config, raw = response
    text = repeated(raw, "next_action", None).replace(
        '"next_action": null, "next_action": null',
        '"next_action": null, "next_action": "different"',
        1,
    )
    with pytest.raises(ValueError, match="forbidden_duplicate"):
        interpret(text, config)


def test_no_duplicate_and_nonfinite_are_not_interpretations(response):
    config, raw = response
    with pytest.raises(ValueError, match="no_redundant_fields"):
        interpret(json.dumps(raw), config)
    with pytest.raises(ValueError, match="nonfinite"):
        interpret('{"x": NaN}', config)
