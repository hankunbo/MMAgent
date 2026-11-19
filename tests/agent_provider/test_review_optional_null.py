"""Opt-in absence/null equivalence; no invented clinical values or old-byte edits."""

import copy
import json

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.config import review_format_upgrade
from packages.llm_provider.provider import DeterministicFakeProvider, ProviderError
from packages.llm_provider.review_response import decode
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_directional_review import fixture, reply


def upgraded(replies):
    f = fixture()
    old = f.config.model_copy(
        update={"transport": f.config.transport.model_copy(update={"max_retries": 0})}
    )
    f.config = review_format_upgrade(old)
    raw = f.owner.state.model_dump(mode="json")
    raw["config_checksum"] = digest(f.config)
    raw["material"]["scope"]["case_code"] = "CASE:SYNTHETIC_FORMAT"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    f.owner.state = validate(type(f.owner.state), raw)
    f.provider = f.agent.provider = DeterministicFakeProvider(f.config, replies)
    return f


def omitted(body):
    raw = reply(body)
    for item in raw["items"]:
        assert item["applicability"]["missing_reason"] is None
        del item["applicability"]["missing_reason"]
    return raw


def malformed(body):
    raw = reply(body)
    raw["items"][0]["unexpected"] = "must not silently discard"
    return Reply(
        content=canonical(raw), finish_reason="stop", response_confirmed=True, api_requests=1
    )


def test_optional_null_wire_domain_score_and_raw_proof():
    f = upgraded([omitted] * 10)
    sub = generate(f)
    f.acceptance.submit(sub, key="nullable")
    assert f.acceptance.accept(key="nullable").status == "accepted"
    responses = [v[0] for k, v in f.documents.saved.items() if k.endswith(":response")]
    assert responses
    for response in responses:
        assert response["schema_version"] == "review-batch-response/1.5.0"
        raw = response["raw_model_text"]
        candidate, proof = decode(raw, f.config)
        assert proof.optional_null_paths
        expected = json.loads(raw)
        for item in expected["items"]:
            assert "missing_reason" not in item["applicability"]
            item["applicability"]["missing_reason"] = None
        assert candidate.model_dump(mode="json") == expected
        old = f.config.model_dump(mode="json")
        old.update(
            schema_version="agent-provider-config/8.0.0",
            review_response_policy="review-missing-method/1.0.0",
        )
        old.pop("format_correction_budget")
        with pytest.raises(ValueError, match="non_equivalent_shape"):
            decode(raw, old)
        bad = copy.deepcopy(expected)
        bad["items"][0]["judgment"].pop("value")
        with pytest.raises(ValueError):
            decode(json.dumps(bad), f.config)
    before = f.budget.used
    assert generate(f) == sub and f.budget.used == before


def test_format_correction_with_network_retries_disabled():
    f = upgraded([malformed] + [reply] * 10)
    sub = generate(f)
    records = f.budget.records()
    assert records[0].error == "schema" and records[1].retry_index == 1
    assert records[1].error is None
    f.acceptance.submit(sub, key="corrected-nullable")
    assert f.acceptance.accept(key="corrected-nullable").status == "accepted"


def test_second_bad_response_stops_without_loop():
    f = upgraded([malformed, malformed, reply])
    with pytest.raises(ProviderError, match="schema"):
        generate(f)
    assert f.budget.used == 2


def test_cancellation_prevents_format_retry():
    f = upgraded([malformed, reply])

    def check():
        if f.budget.records():
            raise ProviderError("cancelled")

    f.budget.check_control = check
    with pytest.raises(ProviderError, match="cancelled"):
        generate(f)
    assert f.budget.used == 1


def test_direction_conflict_feedback_is_specific_without_selecting_a_sign():
    def conflict(body):
        raw = reply(body)
        item = raw["items"][0]
        item["direction_assessment"].update(direction="neutral", neutral_reason="No net direction")
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    seen = []

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        seen.append(feedback)
        assert any(e.get("constraint") == "direction_value_conflict" for e in feedback["errors"])
        assert "Zero and null remain valid" in feedback["contract_consistency"]
        return reply(body)

    f = upgraded([conflict, corrected] + [reply] * 10)
    sub = generate(f)
    assert len(seen) == 1
    f.acceptance.submit(sub, key="direction-consistency")
    assert f.acceptance.accept(key="direction-consistency").status == "accepted"
    assert f.budget.records()[0].error == "schema"
    assert f.budget.records()[1].retry_index == 1


def test_unrecognized_error_message_is_not_sent_in_feedback():
    from pydantic import BaseModel, ValidationError, model_validator

    from packages.llm_provider.review_correction import fields

    class PrivateError(BaseModel):
        value: int

        @model_validator(mode="after")
        def reject(self):
            raise ValueError("private_untrusted_payload")

    with pytest.raises(ValidationError) as exc:
        PrivateError(value=1)
    assert fields(exc.value) == [{"path": [], "type": "value_error"}]


@pytest.mark.parametrize("kind", ["bounds", "method", "missing"])
def test_judgment_shape_feedback_explains_contract_without_imputing_scores(kind):
    def malformed_judgment(body):
        raw = reply(body)
        judgment = raw["items"][0]["judgment"]
        if kind == "bounds":
            judgment.update(lower=None, upper=None)
        elif kind == "method":
            judgment.update(
                value=None, lower=None, upper=None, missing_reason="Unavailable relation"
            )
        else:
            judgment.update(
                value=None, lower=None, upper=None, missing_reason=None, method="not_assessed"
            )
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    seen = []

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        seen.append(feedback)
        assert any(
            e.get("constraint") == f"continuous_judgment_{kind}_conflict"
            for e in feedback["errors"]
        )
        assert "lower <= value <= upper" in feedback["judgment_contract"]
        assert "not_assessed" in feedback["judgment_contract"]
        assert "Do not invent" in feedback["judgment_contract"]
        return reply(body)

    f = upgraded([malformed_judgment, corrected] + [reply] * 10)
    sub = generate(f)
    assert len(seen) == 1
    f.acceptance.submit(sub, key="judgment-shape")
    assert f.acceptance.accept(key="judgment-shape").status == "accepted"
    assert f.budget.records()[0].error == "schema"
    assert f.budget.records()[1].retry_index == 1
    before = f.budget.used
    assert generate(f) == sub and f.budget.used == before


def test_old_feedback_without_new_constraint_preserves_exact_wire():
    from packages.llm_provider.review_correction import corrected_wire

    body = {"messages": [{"role": "system", "content": "synthetic original schema"}]}
    failures = [{"path": ["items", 0, "judgment"], "type": "value_error"}]
    assert digest(corrected_wire(body, failures)) == (
        "sha256:53601bdb047ebd90f7e4203229fb048d6be3363065f6c1c667df188027dd258f"
    )


@pytest.mark.parametrize("field", ["method", "lower", "upper", "value"])
def test_missing_judgment_field_feedback_includes_both_valid_states(field):
    def missing(body):
        raw = reply(body)
        raw["items"][0]["judgment"].pop(field)
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    seen = []

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        seen.append(feedback)
        assert any(e["path"] == ["items", 0, "judgment", field] for e in feedback["errors"])
        assert "method=not_assessed" in feedback["judgment_contract"]
        assert "manual_source_bound_review" in feedback["judgment_contract"]
        assert "Do not invent" in feedback["judgment_contract"]
        return reply(body)

    f = upgraded([missing, corrected] + [reply] * 10)
    sub = generate(f)
    assert len(seen) == 1
    f.acceptance.submit(sub, key="required-judgment-field")
    assert f.acceptance.accept(key="required-judgment-field").status == "accepted"
    before = f.budget.used
    assert generate(f) == sub and f.budget.used == before


def test_shortened_plan_identity_gets_original_literal_not_rewritten_answer():
    originals = []

    def truncated(body):
        raw = reply(body)
        originals.append(raw["plan_checksum"])
        raw["plan_checksum"] = raw["plan_checksum"][:-10]
        return Reply(content=canonical(raw), finish_reason="stop", response_confirmed=True)

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        assert feedback["copy_binding_values_exactly"] == {"plan_checksum": originals[0]}
        return reply(body)

    f = upgraded([truncated, corrected] + [reply] * 10)
    sub = generate(f)
    failed = next(
        v[0]
        for k, v in f.documents.saved.items()
        if ":attempt:" in k and v[0].get("error") == "schema"
    )
    assert json.loads(failed["raw_model_text"])["plan_checksum"] == originals[0][:-10]
    f.acceptance.submit(sub, key="identity-feedback")
    assert f.acceptance.accept(key="identity-feedback").status == "accepted"
