"""Observation-first wire facts, not rewritten model answers or relaxed acceptance."""

import json
from pathlib import Path

import pytest

from apps.vectorengine_single_case import load_config
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import (
    ReviewBatchPlan,
    batch_payload,
    check_batch,
    make_plan,
)
from packages.grounding.models import validate
from packages.llm_provider.public_batches import batch_wire
from packages.llm_provider.review_response import decode
from tests.agent_provider.test_review_boundaries import rebind
from tests.agent_provider.test_review_citation_contract import owner, schema_from

CONFIG = "configs/agents/vectorengine_active_acquisition_r01.json"


@pytest.mark.parametrize(
    "status,expected",
    [
        ("unknown", "input_missing"),
        ("missing", "input_missing"),
        ("ungradable", "observation_ungradable"),
        ("obtained", None),
    ],
)
def test_status_projection_is_derived_not_a_rating(status, expected):
    reader = owner(count=1)
    observations = reader.state.material.model_dump(mode="json")["observations"]
    observations[0].update(status=status, value="0" if status == "obtained" else None)
    rebind(reader, observations=observations)
    config = load_config(CONFIG)
    state = reader.current()
    plan = make_plan(state, config.review_batching, key="status-test")
    wire = batch_wire(config, state, plan, plan.batches[0])
    schema = schema_from(wire)
    row = schema["properties"]["items"]["items"]["anyOf"][0]["properties"]
    assert row["observation_available"]["enum"] == [status == "obtained"]
    if expected:
        assert row["status"]["enum"] == [expected]
    table = next(
        p["text"]
        for p in wire["messages"][1]["content"]
        if p["text"].startswith("REVIEW_CITATION_CODES:")
    )
    value = json.loads(table.split("\n", 1)[1])["tasks"][0]
    assert value["required_primary_status"] == expected
    assert not {"judgment", "observation", "relation", "applicability"} & value.keys()


def saved_failure():
    root = Path("docs/acceptance/12-live-active-acquisition-v1-v1/readback")
    docs = json.loads((root / "original-case-documents.json").read_text())
    event = docs["live-stage-intent:review"][0]["event"]["code"]
    state = validate(DomainState, docs["review:" + event + ":intent"][0]["state"])
    progress = json.loads((root / "batch-progress.json").read_text())
    plan = validate(ReviewBatchPlan, progress["plan"])
    requests = json.loads((root / "actual-requests.json").read_text())
    raw = next(r["raw_response"][0] for r in requests if r["record"]["error"] == "scope")
    candidate, _ = decode(
        raw["raw_model_text"], load_config("configs/agents/vectorengine_active_acquisition_v1.json")
    )
    return state, plan, candidate


def test_real_missing_status_failure_is_not_relabelled_as_success():
    state, plan, candidate = saved_failure()
    with pytest.raises(ValueError, match="review_missing_status"):
        check_batch(state, plan, plan.batches[5], candidate)
    config = load_config(CONFIG)
    wire = batch_wire(config, state, plan, plan.batches[5])
    schema = schema_from(wire)
    variants = schema["properties"]["items"]["items"]["anyOf"]
    rnfl = next(
        v["properties"]
        for v in variants
        if v["properties"]["assessment_id"]["enum"] == ["criterion:RNFL_DEFECT"]
    )
    assert rnfl["status"]["enum"] == ["input_missing"]
    assert candidate.items[-1].status == "reference_missing"
    assert candidate.items[-1].judgment.value is None
    # Original controlled observation/definition, not a synthetic medical answer.
    payload = batch_payload(state, plan, plan.batches[5])
    assert payload["review_tasks"]["tasks"][-1]["definition"]["interpretation_status"] == "missing"


def test_retry_explicit_identity_keeps_original_history_and_source_vision_wire():
    from apps.vectorengine_round import load_round
    from packages.llm_provider.public_batches import source_vision_reuse

    r = load_round("configs/research/vectorengine_active_acquisition_r01_case.json")
    assert len(r.prior_round_configs) == 2
    assert r.max_requests == 40 and r.seconds == 2700
    assert len(source_vision_reuse(load_config(r.provider_config))) == 2
    old = load_round("configs/research/vectorengine_active_acquisition_v1_case.json")
    assert old.run_id != r.run_id
    assert old.acquisition_checksum == r.acquisition_checksum
