"""Only the original zero-dispatch automatic Review closure permits Scope reuse."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import input_scope, scope_successor
from packages.planner_control.models import Command, Control, Event, Outcome, Phase
from tests.agent_provider.test_bounded_live import binding


@pytest.mark.parametrize(
    "fault",
    [None, "after", "before", "event", "request", "unknown", "user_cancel", "phase", "accepted"],
)
def test_cancelled_review_requires_exact_original_zero_request_closure(
    tmp_path, monkeypatch, fault
):
    b = binding()
    event = Event(
        command=Command(key="live-review:review", expected_revision=0, action="review"),
        request_checksum=digest("request"),
        before=Phase.REVIEW,
        after=Phase.STOPPED,
        graph=b.graph,
        outcome=Outcome(reason="cancelled"),
        at=b.started_at,
    )
    control = Control(
        binding=b, graph=b.graph, phase=Phase.STOPPED, cancelled=True, events=(event,), steps=1
    )
    phase = dict(
        run_id=b.run_id,
        target=None,
        accepted=False,
        acceptance_key=None,
        planner=dict(control_checksum=digest("before")),
    )
    closed = dict(
        schema_version="multimodal-review-cancellation/1.0.0",
        before=digest("before"),
        after=digest(control),
        event=event.model_dump(mode="json"),
        cancelled=True,
        new_requests=0,
        remote_response_unknown_preserved=True,
    )
    protocol = {"input": digest("input")}
    result = {"phases": [phase]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    entry = dict(target="input-scope:one", task={"component_id": "image"})
    fact = dict(
        phase={"target": entry["target"]}, task=entry["task"], observation={"visible": True}
    )
    saved = {
        "dynamic-campaign": (protocol,),
        "multimodal-case-result": (result,),
        "dynamic-phase-result": (phase,),
        "multimodal-failed-review-cancellation": (closed,),
    }
    parent = SimpleNamespace(
        output=tmp_path,
        protocol=protocol,
        scope_targets=(entry["target"],),
        documents=lambda _: SimpleNamespace(read=lambda k: saved.get(k)),
        sessions=SimpleNamespace(read=lambda _: control),
        base=SimpleNamespace(input_scope_plan={"entries": [entry]}),
    )
    events = []
    monkeypatch.setattr(
        "packages.research_runtime.agent_records.RuntimeAgentJournal.read", lambda _: events
    )
    reads = []
    monkeypatch.setattr(input_scope, "read_results", lambda *a, **kw: reads.append(1) or [fact])
    if fault in {"after", "before"}:
        closed[fault] = digest("other")
    elif fault == "event":
        closed["event"] = {}
    elif fault == "request":
        events.append({"phase": "completed"})
    elif fault == "unknown":
        events.append({"phase": "reserved"})
    elif fault == "user_cancel":
        (tmp_path / "cancel").touch()
    elif fault == "phase":
        saved["dynamic-phase-result"] = ({"wrong": True},)
    elif fault == "accepted":
        phase["accepted"] = True
        (tmp_path / "result.json").write_text(json.dumps(result))
    before = digest((protocol, result, control, closed))
    if fault:
        with pytest.raises(ValueError, match="scope_current_reuse_"):
            scope_successor.current_proof(parent, {"entries": [entry]})
        assert reads == []
    else:
        proof = scope_successor.current_proof(parent, {"entries": [entry]})
        assert proof["schema_version"] == "current-owner-scope-reuse/1.1.0"
        assert proof["terminal_closure_checksum"] == digest((closed,))
        assert proof["reused"][0]["fact"] == fact and reads == [1]
    assert digest((protocol, result, control, closed)) == before
