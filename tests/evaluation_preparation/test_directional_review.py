"""Versioned research direction through real Provider/batch/admission; no sockets."""

import copy

import pytest

from packages.evidence_details.acceptance_models import DomainState, read_records
from packages.evidence_scoring.review_protocol import ReasonedReviewItem
from packages.grounding.models import digest, validate
from tests.agent_provider.batch_support import batch_reply, generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def explanation(e):
    return dict(
        direction="unassessable"
        if e is None
        else "support"
        if e > 0
        else "refute"
        if e < 0
        else "neutral",
        observed_features="Synthetic visible local feature, not a diagnosis",
        source_relation="Synthetic source describes a qualitative association",
        supporting_interpretation="Local feature is consistent with the proposed mechanism",
        opposing_interpretation="Alternative measurement context also explains this feature",
        alternatives=["Synthetic alternative context"],
        magnitude_reason="Weak net relation after comparing these explanations",
        neutral_reason="The observed feature fits both supplied explanations equally"
        if e == 0
        else None,
        unassessable_reason="No installed interpretation for this observation"
        if e is None
        else None,
        quality_limitation="Only the observed region; original Q applied by the owner",
        resolvable_gap=None,
        next_action=None,
    )


def reply(body):
    raw = batch_reply(body)
    raw["schema_version"] = "research-review-batch/2.0.0"
    for i in raw["items"]:
        if i["status"] == "assessed":
            i["judgment"].update(value=0.03, lower=-0.2, upper=0.3)
        i["direction_assessment"] = explanation(i["judgment"]["value"])
        i["direction_assessment"]["schema_version"] = "direction-assessment/1.0.0"
    return raw


def fixture():
    from packages.evidence_scoring.task_policy import POLICY
    from packages.llm_provider import DeterministicFakeProvider
    from packages.multimodal_research.campaign import configuration

    f = dynamic_fixture(11)
    config = configuration(directional=True).model_copy(update=dict(mode="fake"))
    raw = f.owner.state.model_dump(mode="json")
    p = raw["policy"]
    p.update(
        schema_version="research-task-policy/7.0.0",
        direction_policy="direction-assessment/1.0.0",
        target_code=raw["material"]["goal_code"],
        target_summary=raw["material"]["goal_summary"],
        aggregation_checksum=digest(POLICY),
    )
    for i in p["items"]:
        i["quality"] = dict(
            value=0.2,
            method="partial_visibility",
            reason="Test visible fraction",
            source=raw["resources"][0],
        )
    for g in p["groups"]:
        g["importance"] = dict(observed_input_research=1.0, task_requirement_research=1.0)
    raw["material"]["scope"]["policy_checksum"] = digest(p)
    raw["material"]["policy_ref"]["checksum"] = digest(p)
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    raw["config_checksum"] = digest(config)
    raw["prompt_checksum"] = next(r.prompt_checksum for r in config.roles if r.role == "review")
    f.owner.state = validate(DomainState, raw)
    f.provider = f.agent.provider = DeterministicFakeProvider(config, [reply] * 10)
    f.config = config
    return f


def test_reasoned_batches_acceptance_readback_and_old_contract():
    f = fixture()
    candidate = generate(f)
    f.acceptance.submit(candidate, key="direction")
    receipt = f.acceptance.accept(key="direction")
    assert receipt.status == "accepted"
    saved = f.acceptance.read(key="direction")
    records = read_records(saved[0])
    scores = saved[2]["scores"]
    assert records.schema_version == "case-observation-interpretation/2.5.0"
    assert scores["schema_version"] == "case-research-scoring/11.0.0"
    assert scores["evidence_counts"]["requirements"] == 11
    assert scores["evidence_counts"]["independent_evidence_count"] is None
    assert any(r.judgment.value == 0.03 and r.judgment.lower < 0 for r in candidate.candidate.items)
    assert any(r.judgment.value is None for r in candidate.candidate.items)
    assert all(
        r.review.direction_assessment.model_dump(mode="json")
        == scores["direction_assessments"][r.review.assessment_id]
        for r in records.interpretations
    )
    used = f.budget.used
    assert generate(f) == candidate
    assert f.budget.used == used
    assert f.acceptance.accept(key="direction") == receipt
    from packages.evidence_scoring.api import build
    from packages.evidence_scoring.api import validate as validate_score

    inputs = dict(state=f.owner.current(), candidate=candidate.candidate)
    assert validate_score(inputs, scores) == scores
    with pytest.raises(ValueError, match="version_mismatch"):
        build(inputs, version="9.0.0")


@pytest.mark.parametrize("e", [-0.001, 0.0, 0.001, None])
def test_zero_null_and_weak_direction_are_different(e):
    f = fixture()
    row = generate(f).candidate.items[0].model_dump(mode="json")
    row["judgment"].update(
        value=e,
        lower=-0.5 if e is not None else None,
        upper=0.5 if e is not None else None,
        method="manual_source_bound_review" if e is not None else "not_assessed",
        missing_reason=None if e is not None else "No compatible method",
    )
    row["status"] = "assessed" if e is not None else "method_mismatch"
    row["direction_assessment"] = explanation(e)
    assert validate(ReasonedReviewItem, row).judgment.value == e
    broken = copy.deepcopy(row)
    broken["direction_assessment"]["direction"] = "support" if e != 0.001 else "refute"
    with pytest.raises(ValueError):
        validate(ReasonedReviewItem, broken)


def test_generic_zero_reason_missing_comparison_contract_is_rejected():
    with pytest.raises(ValueError):
        from packages.evidence_scoring.review_protocol import DirectionAssessment

        validate(DirectionAssessment, dict(direction="neutral", neutral_reason="Cannot diagnose"))


def test_live_wire_uses_new_schema_per_task_codes_and_actual_prompt():
    import json

    from packages.evidence_details.review_batches import (
        ReasonedReviewBatch,
        batch_payload,
        make_plan,
    )
    from packages.llm_provider.config import prompt
    from packages.llm_provider.transport import wire_request

    f = fixture()
    state = f.owner.current()
    plan = make_plan(state, f.config.review_batching, key="direction-wire")
    role, text = prompt(f.config, "review")
    body = wire_request(
        f.config.transport,
        role,
        text,
        batch_payload(state, plan, plan.batches[0]),
        ReasonedReviewBatch.model_json_schema(),
        (),
        scope_binding=digest(state.material.scope),
        wire_policy=f.config.wire_policy,
        wire_mode=f.config.wire_mode,
    )
    texts = [x["text"] for x in body["messages"][1]["content"]]
    schema = json.loads(
        next(s.split("\n", 1)[1] for s in texts if s.startswith("OUTPUT_JSON_SCHEMA:"))
    )
    assert role.prompt_version == "review-direction-2.0.0"
    assert any(s.startswith("REVIEW_CITATION_CODES:") for s in texts)
    variants = schema["properties"]["items"]["items"]["anyOf"]
    assert all("direction_assessment" in v["properties"] for v in variants)
    assert len({tuple(v["properties"]["assessment_id"]["enum"]) for v in variants}) == len(variants)


def test_direction_partial_readback_keeps_confirmed_new_version():
    from packages.evidence_details.review_batches import collect_partial
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.provider import ProviderError

    f = fixture()
    f.provider = f.agent.provider = DeterministicFakeProvider(f.config, [reply, {"bad": "schema"}])
    with pytest.raises(ProviderError):
        generate(f)
    partial = collect_partial(f.owner.current(), f.documents, key="synthetic-batches")
    f.acceptance.submit(partial, key="direction-partial")
    assert f.acceptance.accept(key="direction-partial").status == "accepted"
    saved = f.acceptance.read(key="direction-partial")
    assert read_records(saved[0]).schema_version == "case-observation-interpretation/2.6.0"
    assert saved[2]["scores"]["review_completion"]["pending_ids"]
