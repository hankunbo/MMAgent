"""Integration contracts with injected bytes, no real HTTP or patient input."""

import json

import pytest

from apps.vectorengine_batch_prepare import preview, round_path
from apps.vectorengine_single_case import load_config
from packages.evidence_details.review_batches import validate_batch_receipts
from packages.grounding.models import canonical, digest
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.public_batches import case_estimate
from packages.llm_provider.transport import Reply
from tests.agent_provider.batch_support import batch_reply, fixture, generate
from tests.expert_agent.batch_case_support import initial_state

CONFIG = "configs/agents/vectorengine_batch_integrated.json"


def malformed(body):
    raw = batch_reply(body)
    raw["items"][0]["observation"]["unrecognized_field"] = "not silently discarded"
    return Reply(
        content=canonical(raw), finish_reason="stop", response_confirmed=True, api_requests=1
    )


def case_fixture(count=3, replies=None, documents=None):
    f = fixture(count, replies=replies, documents=documents, config_path=CONFIG)
    f.owner.state = initial_state(f.binding, f.config, count)[0]
    return f


def test_one_schema_correction_original_owner_acceptance_and_no_repeat():
    f = case_fixture(replies=[malformed, batch_reply])
    sub = generate(f)
    validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)
    assert f.budget.used == 2 and [r.error for r in f.budget.records()] == ["schema", None]
    assert f.budget.records()[1].retry_index == 1
    event = next(
        e
        for e in f.budget.journal.read()
        if e.get("schema_version") == "review-schema-correction/1.0.0"
    )
    assert "unrecognized_field" in str(event["failures"])
    assert event["corrected_wire_checksum"] == digest(f.provider.transport.calls[1])
    assert len(f.provider.transport.calls[1]["messages"]) == 3
    f.acceptance.submit(sub, key="corrected")
    assert f.acceptance.accept(key="corrected").status == "accepted"
    assert generate(f) == sub and f.budget.used == 2


def test_bad_again_stops_after_one_correction():
    f = case_fixture(replies=[malformed, malformed, batch_reply])
    with pytest.raises(ProviderError, match="schema"):
        generate(f)
    assert f.budget.used == 2


def test_public_schema_error_has_no_correction_or_qualification():
    f = fixture(3, replies=[malformed, batch_reply], config_path=CONFIG)
    with pytest.raises(ProviderError, match="schema"):
        generate(f)
    assert f.budget.used == 1


def test_global_extra_budget_leaves_normal_work_bounded():
    f = case_fixture(12, replies=[malformed, batch_reply] * 4)
    with pytest.raises(ProviderError, match="budget"):
        generate(f)
    assert f.budget.used == 7 and sum(r.retry_index > 0 for r in f.budget.records()) == 3


def test_wrong_source_is_not_a_format_correction():
    def wrong(body):
        r = batch_reply(body)
        r["items"][0]["reference_citations"] = ["SOURCE:WRONG"]
        return r

    f = case_fixture(replies=[wrong, batch_reply])
    with pytest.raises(ProviderError, match="scope"):
        generate(f)
    assert f.budget.used == 1


def test_correction_receipt_cannot_be_rebound():
    f = case_fixture(replies=[malformed, batch_reply])
    sub = generate(f)
    event = next(
        e
        for e in f.budget.journal.events
        if e.get("schema_version") == "review-schema-correction/1.0.0"
    )
    event["failed_record_checksum"] = digest("wrong")
    with pytest.raises(ValueError, match="correction_binding"):
        validate_batch_receipts(f.owner.current(), sub, f.documents, f.budget)


@pytest.mark.parametrize("stage", ["public", "case"])
def test_real_entry_selects_new_configuration_and_same_history(stage):
    path = "configs/research/vectorengine_batch_integrated_" + stage + ".json"
    value = preview(stage, path)
    assert round_path(stage, path) == path
    assert value["round"]["provider_config"] == CONFIG
    assert len(value["round"]["prior_round_configs"]) == 3
    assert value["key_loaded"] is False and value["database_connected"] is False


def test_estimate_counts_history_and_only_actual_global_retry_allowance():
    estimate = case_estimate(45, load_config(CONFIG), 9, [40, 60])
    assert estimate["normal_case_requests"] == 17
    assert estimate["maximum_with_one_retry"] == 20
    assert estimate["remaining_requests"] == 31 and estimate["feasible"]
    assert not case_estimate(45, load_config(CONFIG), 21, [40, 60])["feasible"]


def test_history_allocation_rejects_overlap_and_preserves_usage():
    from packages.llm_provider.live_allocation import prior_case_used, read_allocation

    raw = dict(
        schema_version="bounded-live-allocation/3.0.0",
        session_id="batch-integrated",
        stage="public",
        public_run_id="RUN:NEW",
        public_used=0,
        public_journal_checksum=None,
        configuration_checksums=[digest("config")],
        history=[
            dict(
                stage="case",
                run_id="RUN:OLD",
                configuration_checksum=digest("old"),
                control_checksum=digest("control"),
                journal_checksum=digest("journal"),
                used=7,
            )
        ],
    )
    assert prior_case_used(read_allocation(raw)) == 7
    raw["history"].append(raw["history"][0])
    with pytest.raises(ValueError, match="historical_run_identity"):
        read_allocation(raw)


def test_schema_error_feedback_contains_no_raw_patient_or_model_text():
    f = case_fixture(replies=[malformed, batch_reply])
    generate(f)
    feedback = json.loads(f.provider.transport.calls[1]["messages"][-1]["content"])
    assert "not silently discarded" not in str(feedback)


def test_cancellation_before_correction_does_not_send_or_reset_claim():
    f = case_fixture(replies=[malformed, batch_reply])

    def control():
        if f.budget.records():
            raise ProviderError("cancelled")

    f.budget.check_control = control
    with pytest.raises(ProviderError, match="cancelled"):
        generate(f)
    assert f.budget.used == 1 and len(f.provider.transport.calls) == 1
    with pytest.raises(ProviderError, match="cancelled"):
        generate(f)
    assert f.budget.used == 1
