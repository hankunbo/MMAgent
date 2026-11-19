"""A confirmed response, a missing reservation, and acceptance are not equivalent."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from packages.evaluation_preparation.generic_capability_recovery import (
    phase_inventory,
    response_inventory,
)

from packages.grounding.models import digest
from packages.llm_provider.models import RequestRecord, Usage, VisionProposal
from packages.object_store import checksum
from tests.evaluation_preparation.test_directional_review import fixture


def response():
    scope = fixture().owner.current().material.scope
    candidate = VisionProposal(
        scope_checksum=digest(scope),
        image_checksum=digest("image"),
        preprocessing_checksum=digest("preprocessing"),
        location=(0, 0, 1, 1),
        status="observed",
        finding="synthetic",
    )
    text = json.dumps(candidate.model_dump(mode="json"))
    record = RequestRecord(
        request_id="request:synthetic",
        invocation_id="invocation:synthetic",
        retry_index=0,
        role="vision",
        scope=scope,
        provider="synthetic_fixture",
        model="test",
        endpoint="chat_completions",
        config_checksum=digest("config"),
        prompt_version="vision-1.0.0",
        prompt_checksum=digest("prompt"),
        input_checksum=digest("input"),
        output_checksum=digest(candidate),
        remote_request_checksum=None,
        attempt=1,
        simulated=False,
        api_requests=1,
        usage=Usage(),
        latency_seconds=0.0,
        error=None,
        deadline=datetime.now(UTC),
    )
    raw = dict(
        schema_version="bounded-live-response/1.0.0",
        request_id=record.request_id,
        raw_model_text=text,
        raw_checksum=checksum(text.encode()),
        validation_error=None,
        untrusted_model_output=True,
        accepted_medical_evidence=False,
    )
    events = [
        dict(phase="reserved", request_id=record.request_id),
        dict(phase="completed", record=record.model_dump(mode="json")),
        dict(phase="response_confirmation", request_id=record.request_id, confirmed=True),
    ]
    return record, raw, events


def test_exact_response_is_confirmed_but_not_new_dispatch_authority():
    record, raw, events = response()
    rows = response_inventory(SimpleNamespace(read=lambda key: (raw,)), events)
    assert rows[0]["status"] == "confirmed_response"
    assert rows[0]["record_checksum"] == digest(record)
    assert rows[0]["replay_authorized"] is False


@pytest.mark.parametrize("change", ["text", "checksum", "record", "accepted", "missing"])
def test_incomplete_or_rewritten_response_rejected(change):
    _, raw, events = response()
    if change == "text":
        raw["raw_model_text"] += "changed"
    elif change == "checksum":
        raw["raw_checksum"] = digest("changed")
    elif change == "record":
        events[1]["record"]["output_checksum"] = digest("changed")
    elif change == "accepted":
        raw["accepted_medical_evidence"] = True
    with pytest.raises(
        ValueError,
        match="generic_capability_recovery_(response_binding|confirmed_response_missing)",
    ):
        response_inventory(
            SimpleNamespace(read=lambda key: None if change == "missing" else (raw,)), events
        )


def test_reserved_without_confirmation_remains_unknown_and_charged():
    _, raw, events = response()
    events[-1]["confirmed"] = None
    rows = response_inventory(SimpleNamespace(read=lambda key: (raw,)), events)
    assert len(rows) == 1 and rows[0]["status"] == "unknown"
    rows = response_inventory(SimpleNamespace(read=lambda key: None), events[:1])
    assert len(rows) == 1 and rows[0]["status"] == "unknown"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_duplicate_journal_identity_rejected(index):
    _, raw, events = response()
    events.append(events[index])
    with pytest.raises(ValueError, match="request_accounting"):
        response_inventory(SimpleNamespace(read=lambda key: (raw,)), events)


def test_only_original_missing_registration_means_unregistered(tmp_path):
    def missing(key):
        raise ValueError("research_registration_pending")

    owner = SimpleNamespace(
        phase_path=lambda target: tmp_path,
        request_key=lambda target: "original-key",
        requests=SimpleNamespace(for_key=missing),
    )
    row = phase_inventory(owner, "target")
    assert row["status"] == "not_registered" and row["reservations"] == 0
    (tmp_path / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="unregistered_local_phase"):
        phase_inventory(owner, "target")


def test_database_error_is_not_unsent(tmp_path):
    def failed(key):
        raise ValueError("source_changed")

    owner = SimpleNamespace(
        phase_path=lambda target: tmp_path,
        request_key=lambda target: "original-key",
        requests=SimpleNamespace(for_key=failed),
    )
    with pytest.raises(ValueError, match="source_changed"):
        phase_inventory(owner, "target")


def test_snapshot_reuses_current_owner_root_proof_only_within_inventory(tmp_path, monkeypatch):
    from datetime import timedelta

    from packages.evaluation_preparation import generic_capability_recovery as module

    history = dict(reservations=10, original_result="unchanged")
    selection = dict(schema_version="generic-capability-successor/1.1.0", original_history=history)
    current = dict(original_history=history.copy())
    calls = []

    def read_current():
        calls.append("current")
        return current, {}

    def extra_read():
        pytest.fail("the current root proof was already read by the selection owner")

    campaign = SimpleNamespace(
        review_key="old",
        spec={"key": "root"},
        selection=selection,
        selection_owner=SimpleNamespace(current=read_current),
        checked=SimpleNamespace(history=history),
        visual_targets=(),
        protocol={"deadline": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
        usage=lambda: 0,
        accepted_observations=lambda: {},
        prior=extra_read,
    )
    monkeypatch.setattr(module, "directory", lambda _: tmp_path)
    monkeypatch.setattr(
        module,
        "phase_inventory",
        lambda *a: dict(target=None, status="not_registered", reservations=0),
    )
    first = module.snapshot(campaign)
    assert first["cumulative_requests"] == 10 and calls == ["current"]
    assert module.snapshot(campaign) == first and calls == ["current", "current"]
    current["original_history"]["reservations"] = 11
    with pytest.raises(ValueError, match="original_history_binding"):
        module.snapshot(campaign)
