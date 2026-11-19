"""Pure contract probes, deliberately separate from the owned-PG integration proof."""

from copy import deepcopy

import pytest

from packages.grounding.models import digest
from packages.memory.writer_models import ExecutionSource, FieldRef
from packages.memory_writer.projection import NotReady, project, terminal
from tests.planner.support import advance, setup


def material():
    p, runtime, _ = setup(case_code="CASE:OID_PROJECTION", run_id="RUN:OID_PROJECTION")
    advance(p, until="done")
    snapshot = dict(
        schema_version="research-memory-execution/1.0.0",
        runtime=dict(
            case_code="CASE:OID_PROJECTION",
            run_id="RUN:OID_PROJECTION",
            runtime_status="CREATED",
            lifecycle_phase=None,
            report=None,
            attempts=[],
            preparation_rejected=False,
            versions={},
        ),
        control=runtime.state.model_dump(mode="json"),
        control_report=p.report(runtime.state.binding.run_id).model_dump(mode="json"),
    )
    return snapshot


def source(snapshot):
    return ExecutionSource(
        case_code=snapshot["runtime"]["case_code"],
        run_id=snapshot["runtime"]["run_id"],
        key=snapshot["runtime"]["run_id"],
        object_code=snapshot["runtime"]["run_id"],
        checksum=digest(snapshot),
        object_checksum=digest(snapshot),
        dependencies=(digest("owner"),),
        applicability=("USE:SYNTHETIC",),
        nature="source_fact",
    )


@pytest.mark.parametrize("count", [16, 17, 64, 65, 256, 257])
def test_explicit_long_trajectory_bound_without_truncation(count):
    s = material()
    template = s["control"]["events"][0]
    events = []
    for n in range(count):
        e = deepcopy(template)
        e["command"].update(key=f"event-{n}", expected_revision=n * 2)
        e["before"] = e["after"] = "retrieve"
        # Event ordering is owner append order, never wall-clock sorting.
        e["at"] = "2026-01-01T00:00:00+00:00"
        events.append(e)
    s["control"]["events"] = events
    if count > 256:
        with pytest.raises(ValueError, match="memory_trajectory_bound"):
            project(s, source(s), request_key="long")
        return
    records = project(s, source(s), request_key="long")
    d = records[-1].detail
    assert d.trajectory_count == count
    assert len(d.segments) == (count + 15) // 16
    steps = [step for r in records if r.kind == "trajectory_step" for step in r.detail.steps]
    assert [v.index for v in steps] == list(range(count))
    assert [v.command_key for v in steps] == [f"event-{n}" for n in range(count)]
    assert max(len(r.sources) for r in records) <= 64 and len(records[-1].parents) <= 64


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("active_attempt", "memory_execution_unconfirmed"),
        ("unknown_lifecycle", "memory_execution_unconfirmed"),
        ("unfinished_lifecycle", "memory_lifecycle_not_terminal"),
        ("runtime_without_report", "memory_runtime_terminal_evidence_missing"),
        ("pending_cancelled", "memory_execution_unconfirmed"),
    ],
)
def test_independent_terminal_fields(condition, reason):
    s = material()
    if condition == "active_attempt":
        s["runtime"]["attempts"] = [dict(status="ACTIVE")]
    elif condition == "unknown_lifecycle":
        s["runtime"]["lifecycle_phase"] = "UNKNOWN"
    elif condition == "unfinished_lifecycle":
        s["runtime"]["lifecycle_phase"] = "SCORING"
    elif condition == "runtime_without_report":
        s["runtime"]["runtime_status"] = "FAILED_TECHNICAL"
    else:
        s["control"].update(pending={"execution_key": "original"}, cancelled=True)
    with pytest.raises(NotReady, match=reason):
        terminal(s, source(s))


@pytest.mark.parametrize(
    "status",
    [
        "COMPLETED_SUPPORTED",
        "COMPLETED_REFUTED",
        "COMPLETED_UNCERTAIN",
        "COMPLETED_INSUFFICIENT_EVIDENCE",
        "FAILED_TECHNICAL",
        "STOPPED_UNSAFE",
    ],
)
def test_runtime_finished_report_consistency(status):
    s = material()
    s["control"] = None
    s["runtime"].update(
        runtime_status=status,
        lifecycle_phase="FINISHED",
        report=dict(
            terminal=True,
            run_status=status,
            technical_status="NOT_EXECUTED",
            reason="synthetic_finished",
        ),
    )
    assert terminal(s, source(s)).scope == "runtime_lifecycle"
    s["runtime"]["report"]["run_status"] = "CREATED"
    with pytest.raises(ValueError, match="memory_terminal_inconsistent"):
        terminal(s, source(s))


def test_order_and_missing_reason_are_not_inferred():
    s = material()
    s["control"]["events"] = list(reversed(s["control"]["events"]))
    with pytest.raises(ValueError, match="memory_event_order"):
        project(s, source(s), request_key="order")
    with pytest.raises(ValueError, match="memory_field_missing_reason"):
        FieldRef()
    with pytest.raises(ValueError, match="memory_field_ambiguous"):
        FieldRef(missing_reason="not_recorded", checksum=digest("not-a-fact"))
