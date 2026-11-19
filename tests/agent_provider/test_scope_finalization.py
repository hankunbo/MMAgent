"""No direction/content repair: only a proven storage-capacity reinterpretation."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evidence_details.scope_finalization import RecordedScopeRequests, interpret
from packages.grounding.models import canonical, digest
from tests.agent_provider.test_input_scope_storage import real_candidate


def test_actual_retained_scope_is_lossless_and_keeps_original_schema():
    old = real_candidate()
    before = digest(old)
    candidate, proof = interpret(canonical(old).decode())
    assert candidate.model_dump(mode="json") == dict(
        old, schema_version="visual-input-scope-candidate/1.1.0"
    )
    assert digest(old) == before == proof.original_candidate_checksum
    assert proof.candidate_checksum == digest(candidate)
    assert proof.changed_fields == ("schema_version",)


@pytest.mark.parametrize("change", ["valid", "unknown", "status", "version", "duplicate", "nan"])
def test_only_storage_capacity_failure_can_be_recovered(change):
    old = real_candidate()
    if change == "valid":
        old["finding"] = "short"
        old["regions"] = [dict(old["regions"][0], finding="short", limitation="short")]
    elif change == "unknown":
        old["unexpected"] = "not a storage problem"
    elif change == "status":
        old["status"] = "ungradable"
    elif change == "version":
        old["schema_version"] = "visual-input-scope-candidate/1.1.0"
    raw = json.dumps(old)
    if change == "duplicate":
        raw = '{"finding":"duplicate",' + raw[1:]
    elif change == "nan":
        raw = raw.replace('"coverage": 1.0', '"coverage": NaN')
        if "NaN" not in raw:
            old["regions"][0]["coverage"] = float("nan")
            raw = json.dumps(old)
    with pytest.raises(ValueError, match="recorded_scope_"):
        interpret(raw)


@pytest.mark.parametrize("change", [None, "unknown", "missing", "second", "error", "request"])
def test_retained_original_request_journal_is_not_upgraded_to_success(change):
    raw = json.loads(
        Path(
            "docs/acceptance/capability-scope-real-diagnostic-20261107/original-scope-payload.json"
        ).read_bytes()
    )["journal"]
    events = deepcopy(raw)
    complete = next(e for e in events if e.get("phase") == "completed")
    confirmation = next(e for e in events if e.get("phase") == "response_confirmation")
    if change == "unknown":
        confirmation["confirmed"] = None
    elif change == "missing":
        events.remove(confirmation)
    elif change == "second":
        events.append(deepcopy(complete))
    elif change == "error":
        complete["record"]["error"] = "transport"
    elif change == "request":
        confirmation["request_id"] = "request:wrong"
    journal = SimpleNamespace(durable=True, read=lambda: events)
    requests = RecordedScopeRequests(journal)
    before = digest(events)
    if change is None:
        record = requests.records()[0]
        assert record.error == "schema" and record.api_requests == 1
        assert record.model_dump(mode="json") == complete["record"]
    else:
        with pytest.raises(ValueError, match="recorded_scope_"):
            requests.records()
    assert digest(events) == before
