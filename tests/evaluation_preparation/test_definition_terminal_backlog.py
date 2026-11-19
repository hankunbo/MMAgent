"""Terminal read is not permission to replay a failed search or UNKNOWN LLM."""

import copy
import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal


def saved(tmp_path, monkeypatch):
    receipt = dict(
        adapter="europe_pmc",
        config_checksum=digest("config"),
        http_status=None,
        operation="search",
        provider_code="SOURCE:PROVIDER:EUROPEPMC",
        provider_version="2.0.0",
        request_checksum=digest("query"),
        requested_at="2026-10-27T12:00:00Z",
        response_checksum=None,
    )
    revision = dict(
        added=[],
        replaced=[],
        accepted={"draft": {"interpretation": None}},
        rejected=[
            dict(
                reason="public_search_incomplete",
                result=dict(
                    status="failed",
                    runs=[
                        dict(
                            outcome=dict(
                                stop_reason="provider_error",
                                error_category="retry_exhausted",
                                wire_receipts=[copy.deepcopy(receipt)],
                            )
                        )
                    ],
                ),
            )
        ],
    )
    protocol = dict(
        schema_version=renewal.RUNTIME_DEFINITION_VERSION,
        mode="definitions",
        definition_increment_only=True,
        input="input",
        task="task",
        previous_attempts=[],
    )
    phase = dict(run_id="phase", target=None)
    result = dict(
        case_code="case",
        task={"target": "test"},
        phases=[phase],
        complete=True,
        new_requests=2,
        previous_attempts=[],
    )
    records = {
        "dynamic-phase-result": (phase,),
        "dynamic-campaign": (protocol,),
        "multimodal-case-result": (result,),
        "definition:revision": (revision,),
        "definition:operation:0": (dict(kind="wire", key="definition:wire:0", wire_kind="search"),),
        "definition:wire:0:outcome": (receipt,),
    }
    events = [
        dict(phase="reserved", request_id="source"),
        dict(phase="completed", record=dict(request_id="source", role="source", error=None)),
    ]
    docs = SimpleNamespace(run_id="phase", read=lambda key: records.get(key))
    source_docs = SimpleNamespace(run_id="source", read=lambda key: records.get(key))
    task = SimpleNamespace(checksum="task", model_dump=lambda **_: result["task"])
    parent = SimpleNamespace(
        base=SimpleNamespace(
            binding={"case_code": "case"}, manifest=SimpleNamespace(checksum="input"), task=task
        ),
        documents=lambda rid: docs if rid == "phase" else source_docs,
        sessions=SimpleNamespace(read=lambda _: SimpleNamespace(phase="stopped", pending=None)),
    )
    monkeypatch.setattr(
        renewal,
        "RuntimeAgentJournal",
        lambda d: SimpleNamespace(read=lambda: [] if d is docs else events),
    )
    restored = []
    monkeypatch.setattr(renewal, "restore_origin_materials", lambda p, tip: restored.append(tip))
    return SimpleNamespace(
        parent=parent,
        records=records,
        revision=revision,
        receipt=receipt,
        events=events,
        protocol=protocol,
        result=result,
        restored=restored,
        path=tmp_path,
    )


def invoke(s, **kwargs):
    (s.path / "protocol.json").write_text(json.dumps(s.protocol))
    (s.path / "result.json").write_text(json.dumps(s.result))
    return renewal.stopped_tip(s.parent, s.path, completed_source=True, **kwargs)


def test_unused_receipt_and_confirmed_search_failure_are_readable_not_replayed(
    tmp_path, monkeypatch
):
    s = saved(tmp_path, monkeypatch)
    before = copy.deepcopy(s.records)
    result = invoke(s)
    assert result["previous_attempts"][-1]["requests"] == 2
    assert result["definition"]["terminal_backlog"]["replay_authorized"] is False
    assert result["definition"]["terminal_backlog"]["failed_public_search_wires"] == [
        "definition:wire:0"
    ]
    assert s.restored == [result] and s.records == before
    assert invoke(s) == result


@pytest.mark.parametrize(
    "fault",
    [
        "legacy",
        "incomplete",
        "added",
        "replaced",
        "wire_missing",
        "wire_changed",
        "wire_fetch",
        "security",
        "cancelled",
        "llm_unknown",
        "llm_duplicate",
        "wire_duplicate",
        "accounting",
        "phase_pending",
        "phase_changed",
        "source_revoked",
    ],
)
def test_unsafe_history_still_rejected(tmp_path, monkeypatch, fault):
    s = saved(tmp_path, monkeypatch)
    if fault == "legacy":
        s.protocol["schema_version"] = "multimodal-campaign-renewal/1.11.0"
    elif fault == "incomplete":
        s.result["complete"] = False
    elif fault in {"added", "replaced"}:
        s.revision[fault] = ["changed"]
    elif fault == "wire_missing":
        s.records.pop("definition:wire:0:outcome")
    elif fault == "wire_changed":
        s.receipt["request_checksum"] = digest("other")
    elif fault == "wire_fetch":
        s.receipt["operation"] = "fetch"
    elif fault == "security":
        s.revision["rejected"][0]["result"]["runs"][0]["outcome"]["error_category"] = "security"
    elif fault == "cancelled":
        s.revision["rejected"][0]["result"]["runs"][0]["outcome"]["stop_reason"] = "cancelled"
    elif fault == "llm_unknown":
        s.events.pop()
    elif fault == "llm_duplicate":
        s.events.append(copy.deepcopy(s.events[-1]))
    elif fault == "wire_duplicate":
        s.records["definition:operation:1"] = s.records["definition:operation:0"]
    elif fault == "accounting":
        s.result["new_requests"] = 1
    elif fault == "phase_pending":
        s.parent.sessions.read = lambda _: SimpleNamespace(phase="stopped", pending="unknown")
    elif fault == "phase_changed":
        s.records["dynamic-phase-result"] = ({"wrong": True},)
    else:

        def revoked(*args):
            raise ValueError("source_revoked")

        monkeypatch.setattr(renewal, "restore_origin_materials", revoked)
    with pytest.raises(ValueError):
        invoke(s)
    assert not s.restored
