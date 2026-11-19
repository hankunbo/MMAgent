"""An empty visible answer at the confirmed token cap is not transport UNKNOWN."""

import pytest

from packages.evidence_details.pending_review import known_truncated_failure, read_pending_review
from packages.grounding.models import digest
from packages.llm_provider.models import Usage
from packages.object_store import checksum
from tests.agent_provider.test_pending_truncated_review import partial


def empty_attempt(monkeypatch):
    fixture, _ = partial(monkeypatch)
    old = fixture.budget.records()[-1]
    slot = "review-batches:synthetic-batches:batch:001"
    saved = fixture.documents.saved[slot + ":attempt:" + old.request_id][0]
    role = next(r for r in saved["provider_config"]["roles"] if r["role"] == "review")
    record = old.model_copy(
        update=dict(
            output_checksum=None,
            usage=Usage(status="observed", output_tokens=role["max_output_tokens"]),
        )
    )
    saved.update(raw_model_text="", raw_checksum=checksum(b""), record_checksum=digest(record))
    for event in fixture.budget.journal.events:
        if event.get("phase") == "completed" and event["record"]["request_id"] == record.request_id:
            event["record"] = record.model_dump(mode="json")
    # Fake transport intentionally does not emit live confirmations. Supply the
    # explicit synthetic confirmation required by this proof-only fixture.
    fixture.budget.journal.append(
        dict(phase="response_confirmation", request_id=record.request_id, confirmed=True)
    )
    return fixture, slot, saved, record


def test_empty_token_cap_is_explicit_proof_not_replay(monkeypatch):
    f, slot, saved, record = empty_attempt(monkeypatch)
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    with pytest.raises(ValueError, match="pending_truncated_response_binding"):
        known_truncated_failure(f.documents, f.budget.journal, slot, [record])
    proof = known_truncated_failure(
        f.documents, f.budget.journal, slot, [record], allow_empty_output=True
    )
    assert proof["schema_version"] == "confirmed-review-truncation/1.1.0"
    assert proof["empty_visible_output"] is True
    assert proof["original_answer_not_accepted"] is True
    assert proof["last_request_id"] == record.request_id
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))
    assert saved["raw_model_text"] == ""


@pytest.mark.parametrize(
    "change", ["usage", "finish", "checksum", "output", "confirmation", "duplicate", "false"]
)
def test_empty_output_does_not_waive_original_bindings(monkeypatch, change):
    f, slot, saved, record = empty_attempt(monkeypatch)
    if change == "usage":
        record = record.model_copy(update=dict(usage=Usage(status="observed", output_tokens=1)))
        saved["record_checksum"] = digest(record)
    elif change == "finish":
        saved["finish_reason"] = "stop"
    elif change == "checksum":
        saved["raw_checksum"] = checksum(b"wrong")
    elif change == "output":
        record = record.model_copy(update=dict(output_checksum=digest("not-empty")))
        saved["record_checksum"] = digest(record)
    elif change == "confirmation":
        original = f.budget.journal.read
        monkeypatch.setattr(
            f.budget.journal,
            "read",
            lambda: [e for e in original() if e.get("phase") != "response_confirmation"],
        )
    else:
        f.budget.journal.append(
            dict(
                phase="response_confirmation",
                request_id=record.request_id,
                confirmed=change != "false",
            )
        )
    with pytest.raises(
        ValueError, match="pending_truncated_response_binding|pending_empty_truncated"
    ):
        known_truncated_failure(
            f.documents, f.budget.journal, slot, [record], allow_empty_output=True
        )


@pytest.mark.parametrize("status", [200, 429])
def test_live_empty_output_requires_confirmed_http(monkeypatch, status):
    f, slot, saved, record = empty_attempt(monkeypatch)
    record = record.model_copy(update=dict(simulated=False, api_requests=1))
    saved["record_checksum"] = digest(record)
    monkeypatch.setattr(
        "packages.llm_provider.http_recovery.observation_for",
        lambda *_: dict(origin="http_worker", http_status=status, operation="chat_completions"),
    )
    if status == 200:
        assert known_truncated_failure(
            f.documents, f.budget.journal, slot, [record], allow_empty_output=True
        )["empty_visible_output"]
    else:
        with pytest.raises(ValueError, match="pending_truncated_not_confirmed_http"):
            known_truncated_failure(
                f.documents, f.budget.journal, slot, [record], allow_empty_output=True
            )


def test_original_partial_owner_keeps_empty_truncation_quarantined(monkeypatch):
    f, _, _, record = empty_attempt(monkeypatch)
    args = dict(
        documents=f.documents,
        state=f.owner.current(),
        acceptance_key="partial",
        allow_legacy_partial=True,
        inspect_truncated_failures=True,
    )
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    with pytest.raises(ValueError, match="pending_truncated_response_binding"):
        read_pending_review(**args)
    value = read_pending_review(**args, inspect_empty_truncated_failures=True)
    assert value["schema_version"] == "pending-review-inventory/1.3.0"
    failed = value["quarantined_batches"][0]
    assert failed["known_truncated_failure"]["last_request_id"] == record.request_id
    assert failed["known_truncated_failure"]["empty_visible_output"]
    assert failed["new_reassessment_authorized"] is False
    assert not set(failed["assessment_ids"]) & set(value["selected_ids"] + value["accepted_ids"])
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))
    with pytest.raises(ValueError, match="pending_empty_truncated_explicit_inspection_required"):
        read_pending_review(
            **dict(args, inspect_truncated_failures=False), inspect_empty_truncated_failures=True
        )
