"""Original legacy acceptance plus confirmed truncation; no remote requests."""

import json
from types import SimpleNamespace

import pytest

from packages.evidence_details.pending_review import known_truncated_failure, read_pending_review
from packages.evidence_details.review_batches import collect_partial
from packages.grounding.models import digest, validate
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import parse_reply
from packages.planner_control.models import Control
from tests.agent_provider.batch_support import batch_reply, generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def partial(monkeypatch):
    reply = parse_reply(
        json.dumps(
            {"choices": [{"finish_reason": "length", "message": {"content": '{"items":['}}]}
        ),
        "chat_completions",
        retain_metadata=True,
    )
    f = dynamic_fixture(
        17,
        replies=[batch_reply, reply],
        config_path="configs/agents/fairvlmed_relay_review_shape_v1.json",
    )
    with pytest.raises(ProviderError, match="truncated"):
        generate(f)
    sub = collect_partial(f.owner.current(), f.documents, "synthetic-batches")
    f.acceptance.submit(sub, key="partial")
    assert f.acceptance.accept(key="partial").status == "accepted"
    f.documents.run_id = f.binding.run_id
    control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    monkeypatch.setattr(
        "packages.evidence_details.pending_review.RuntimeAgentJournal", lambda _: f.budget.journal
    )
    return f, sub


def inspect(f):
    return read_pending_review(
        documents=f.documents,
        state=f.owner.current(),
        acceptance_key="partial",
        allow_legacy_partial=True,
        inspect_truncated_failures=True,
    )


def test_legacy_partial_truncation_is_proof_not_replay_permission(monkeypatch):
    f, sub = partial(monkeypatch)
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    with pytest.raises(ValueError, match="requires_partial_acceptance"):
        read_pending_review(
            documents=f.documents, state=f.owner.current(), acceptance_key="partial"
        )
    value = inspect(f)
    assert value["schema_version"] == "pending-review-inventory/1.2.0"
    assert value["accepted_ids"] == sorted(i.assessment_id for i in sub.candidate.items)
    assert value["selected_ids"] == sorted(
        i for b in sub.plan.batches[2:] for i in b.assessment_ids
    )
    assert value["selected_ids"]
    failed = value["quarantined_batches"][0]
    assert not failed["new_reassessment_authorized"]
    assert failed["known_truncated_failure"]["original_answer_not_accepted"]
    assert failed["known_truncated_failure"]["last_request_id"] == f.budget.records()[-1].request_id
    assert not set(value["selected_ids"]) & set(value["accepted_ids"] + failed["assessment_ids"])
    assert inspect(f) == value
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


@pytest.mark.parametrize("change", ["text", "finish", "config", "record", "claim", "unknown"])
def test_changed_or_unconfirmed_response_fails_closed(monkeypatch, change):
    f, _ = partial(monkeypatch)
    key = next(
        k
        for k in f.documents.saved
        if ":attempt:" in k and f.documents.saved[k][0]["error"] == "truncated"
    )
    raw = f.documents.saved[key][0]
    if change == "text":
        raw["raw_model_text"] += " "
    elif change == "finish":
        raw["finish_reason"] = "stop"
    elif change == "config":
        raw["provider_config"]["transport"]["timeout_seconds"] += 1
    elif change == "record":
        raw["record_checksum"] = digest("wrong")
    elif change == "claim":
        f.documents.saved["review-batches:synthetic-batches:batch:001:claim"][0][
            "input_checksum"
        ] = digest("wrong")
    else:
        f.budget.journal.append(dict(phase="reserved", request_id="request:unknown"))
    with pytest.raises(
        ValueError,
        match="pending_truncated_response_binding|pending_review_claim_binding|pending_review_request_inventory",
    ):
        inspect(f)


def test_truncation_not_confused_with_unknown(monkeypatch):
    f, _ = partial(monkeypatch)
    record = f.budget.records()[-1].model_copy(update={"error": "timeout"})
    assert known_truncated_failure(f.documents, f.budget.journal, "unused", [record]) is None


@pytest.mark.parametrize(
    "http", [None, {"origin": "http_worker", "http_status": 429, "operation": "chat_completions"}]
)
def test_live_truncation_needs_original_http_confirmation(monkeypatch, http):
    f, _ = partial(monkeypatch)
    record = f.budget.records()[-1].model_copy(update={"simulated": False, "api_requests": 1})
    slot = "review-batches:synthetic-batches:batch:001"
    f.documents.saved[slot + ":attempt:" + record.request_id][0]["record_checksum"] = digest(record)
    monkeypatch.setattr("packages.llm_provider.http_recovery.observation_for", lambda *_: http)
    with pytest.raises(ValueError, match="pending_truncated_not_confirmed_http"):
        known_truncated_failure(f.documents, f.budget.journal, slot, [record])
