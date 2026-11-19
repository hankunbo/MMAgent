"""Known rejected replies are distinct from UNKNOWN and from accepted evidence."""

from types import SimpleNamespace

import pytest

from packages.evidence_details.pending_review import read_pending_review
from packages.grounding.models import digest
from packages.multimodal_research.campaign_renewal import validate_format_window
from packages.multimodal_research.pending_review import select_known_formats
from tests.agent_provider.test_pending_review import partial_fixture


def test_confirmed_format_inventory_preserves_raw_and_requires_exact_ack(monkeypatch):
    f, _ = partial_fixture(monkeypatch)
    old = f.old
    before = digest(old.documents.saved), digest(old.budget.journal.read())
    value = read_pending_review(
        documents=old.documents,
        state=old.owner.current(),
        acceptance_key="partial",
        inspect_format_failures=True,
    )
    batch = value["quarantined_batches"][0]
    value["historical_proof_row"] = batch
    proof = batch["known_format_failure"]
    assert proof and not batch["new_reassessment_authorized"]
    selected = select_known_formats(value, [proof["last_request_id"]])
    assert set(selected["selected_ids"]) == set(value["selected_ids"] + batch["assessment_ids"])
    assert not set(selected["selected_ids"]) & set(value["accepted_ids"])
    assert selected["quarantined_batches"][0]["new_reassessment_authorized"]
    assert not selected["historical_proof_row"]["new_reassessment_authorized"]
    with pytest.raises(ValueError, match="format_ack_mismatch"):
        select_known_formats(value, ["request:not_this_batch"])
    assert before == (digest(old.documents.saved), digest(old.budget.journal.read()))


def test_format_inventory_rejects_altered_original_reply(monkeypatch):
    f, _ = partial_fixture(monkeypatch)
    old = f.old
    key = next(
        k
        for k, v in old.documents.saved.items()
        if ":attempt:" in k and v[0].get("error") == "schema"
    )
    old.documents.saved[key][0]["raw_model_text"] += " "
    with pytest.raises(ValueError, match="format_response_binding"):
        read_pending_review(
            documents=old.documents,
            state=old.owner.current(),
            acceptance_key="partial",
            inspect_format_failures=True,
        )


def test_unknown_cannot_be_selected_as_known_format():
    value = dict(
        selected_ids=[],
        accepted_ids=[],
        quarantined_batches=[
            dict(
                assessment_ids=["a"],
                request_ids=["request:unknown"],
                known_format_failure=None,
                new_reassessment_authorized=False,
            )
        ],
    )
    with pytest.raises(ValueError, match="format_ack_mismatch"):
        select_known_formats(value, ["request:unknown"])


def test_new_format_window_bound_and_one_time_only():
    parent = SimpleNamespace(
        protocol=dict(deadline="2030-01-01T00:00:00+00:00"),
        base=SimpleNamespace(execution_profile="relay-snapshot-context-v1"),
        config={"original": True},
    )
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.24.0",
        mode="review",
        pending_review=dict(acknowledged_format_requests=["request:known"]),
        previous_window_deadline=parent.protocol["deadline"],
        whole_case_execution_profile=parent.base.execution_profile,
        provider=digest(parent.config),
        started_at="2030-01-01T01:00:00+00:00",
        deadline="2030-01-01T01:45:00+00:00",
        seconds=2700,
        max_new_requests=10,
        previous_attempts=[dict(requests=18)],
        new_vision_requests=0,
        new_public_search_allowed=False,
    )
    validate_format_window(plan, parent)
    for name, value in (
        ("seconds", 2701),
        ("max_new_requests", 11),
        ("new_vision_requests", 1),
        ("previous_attempts", [dict(requests=31)]),
        ("provider", digest({})),
    ):
        with pytest.raises(ValueError, match="window_binding"):
            validate_format_window(dict(plan, **{name: value}), parent)
    parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.24.0"
    with pytest.raises(ValueError, match="already_used"):
        validate_format_window(plan, parent)
