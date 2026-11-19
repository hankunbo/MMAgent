import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from packages.evidence_details.scope_finalization import RecordedScopeRequests, interpret
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.input_scope import (
    STORAGE_PROMPT,
    ScopeVisionProposalV2,
    ScopeVisionProposalV3,
    proposal_type,
)
from tests.agent_provider.test_input_scope_storage import real_candidate


def text_candidate():
    value = real_candidate()
    value["schema_version"] = "visual-input-scope-candidate/1.1.0"
    value["regions"][0]["limitation"] = "L" * 106
    return value


def test_region_text_capacity_explicit_lossless_version_and_old_wire_rejection():
    raw = text_candidate()
    original = deepcopy(raw)
    candidate, proof = interpret(canonical(raw).decode(), prompt=STORAGE_PROMPT)
    assert candidate.model_dump(mode="json") == dict(
        raw, schema_version="visual-input-scope-candidate/1.2.0"
    )
    assert raw == original
    assert candidate.regions[0].limitation == "L" * 106
    assert proof.affected_fields == ("/regions/0/limitation",)
    assert proof.original_candidate_checksum == digest(raw)
    assert proof.changed_fields == ("schema_version",)
    assert proof.original_text_limit == 100 and proof.current_text_limit == 512
    assert proposal_type(STORAGE_PROMPT) is ScopeVisionProposalV2
    with pytest.raises(ValidationError):
        validate(proposal_type(STORAGE_PROMPT), raw)
    assert validate(ScopeVisionProposalV3, candidate.model_dump(mode="json")) == candidate


@pytest.mark.parametrize("field", ["finding", "limitation"])
@pytest.mark.parametrize("length", [101, 512])
def test_both_region_descriptions_have_explicit_bounded_storage(field, length):
    raw = text_candidate()
    raw["regions"][0][field] = "Q" * length
    candidate, proof = interpret(canonical(raw).decode(), prompt=STORAGE_PROMPT)
    assert getattr(candidate.regions[0], field) == "Q" * length
    assert "/regions/0/" + field in proof.affected_fields


@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "extra",
        "status",
        "old_version",
        "region",
        "coverage",
        "unidentified",
        "duplicate_region",
        "missing",
        "over_capacity",
        "extra_long_top_finding",
        "empty_region",
        "coercion",
    ],
)
def test_capacity_recovery_cannot_hide_other_validation_failures(change):
    raw = text_candidate()
    if change == "valid":
        raw["regions"][0]["limitation"] = "short"
    elif change == "extra":
        raw["unexpected"] = "bad"
    elif change == "status":
        raw["status"] = "ungradable"
    elif change == "old_version":
        raw["schema_version"] = "visual-input-scope-candidate/1.0.0"
    elif change == "region":
        raw["regions"][0]["visibility"] = "invented"
    elif change == "coverage":
        raw["regions"][0]["coverage"] = 1.1
    elif change == "unidentified":
        raw["regions"][0]["visibility"] = "not_identified"
    elif change == "duplicate_region":
        raw["regions"].append(deepcopy(raw["regions"][0]))
    elif change == "missing":
        del raw["regions"][0]["clarity"]
    elif change == "over_capacity":
        raw["regions"][0]["limitation"] = "x" * 513
    elif change == "extra_long_top_finding":
        raw["finding"] = "x" * 2000
    elif change == "empty_region":
        raw["regions"] = []
    elif change == "coercion":
        raw["regions"][0]["clarity"] = "0.5"
    with pytest.raises(ValueError):
        interpret(canonical(raw).decode(), prompt=STORAGE_PROMPT)


@pytest.mark.parametrize("bad", ["duplicate", "nan", "oversize"])
def test_same_json_boundaries_hold_for_new_local_interpretation(bad):
    raw = json.dumps(text_candidate())
    if bad == "duplicate":
        raw = '{"finding":"duplicate",' + raw[1:]
    elif bad == "nan":
        raw = raw.replace('"coverage":', '"coverage": NaN, "ignored":')
    else:
        raw = raw + " " * 32768
    with pytest.raises(ValueError):
        interpret(raw, prompt=STORAGE_PROMPT)


def test_actual_harvard_confirmed_request_is_eligible_journal_not_promoted_success():
    events = json.loads(
        Path(
            "docs/acceptance/capability-harvard-scope-response-20261108/original-failure-owner.json"
        ).read_text()
    )["journal"]
    before = digest(events)
    journal = SimpleNamespace(durable=True, read=lambda: events)
    record = RecordedScopeRequests(journal).records()[0]
    assert record.error == "schema" and record.prompt_version == STORAGE_PROMPT
    assert record.api_requests == 1
    assert digest(events) == before
