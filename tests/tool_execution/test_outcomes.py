"""Execution result invariants, using immutable real 06D export facts."""

import json
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest

from packages.grounding.models import validate
from packages.tool_execution.models import StepResult, WorkflowResult, derive_workflow_status
from tests.tool_execution.support import prepared, runtime_files
from tests.tool_execution.test_dependencies import TestLedger

EXAMPLES = Path(__file__).resolve().parents[2] / "docs/acceptance/06d/examples"


@pytest.mark.parametrize("change", [{"exit_code": 7}, {"error_code": "process_nonzero"}])
def test_success_rejects_contradictory_process_fact(change):
    data = json.loads((EXAMPLES / "step-1.json").read_text())
    data.update(change)
    with pytest.raises(ValueError):
        validate(StepResult, data)


@pytest.fixture
def original():
    return json.loads((EXAMPLES / "two-step-workflow.json").read_text())


def outcome(data, status, error=None, exit_code=None, history=None):
    data = deepcopy(data)
    data["exit_code"], data["error_code"] = exit_code, error
    data["result"].update(
        status=status,
        error=None if error is None else dict(code=error, message=error, retryable=False),
    )
    data["status_history"] = history or [
        "REQUESTED",
        "DRY_RUN_VALIDATED",
        "QUEUED",
        "RUNNING",
        status,
    ]
    if status != "SUCCEEDED":
        data["result"].update(artifacts=[], qc_status="UNASSESSED")
        data["qc_records"] = []
    return data


@pytest.mark.parametrize(
    "status,error,exit_code,history",
    [
        ("SUCCEEDED", None, 0, None),
        ("FAILED", "process_nonzero", 7, None),
        ("FAILED", "isolation_setup_failed", 125, None),
        ("FAILED", "output_format_invalid", 0, None),
        ("FAILED", "execution_validation_failed", 0, None),
        ("FAILED", "execution_validation_failed", None, None),
        ("FAILED", "resource_limit", -24, None),
        ("FAILED", "log_size_limit", 0, None),
        ("TIMED_OUT", "timeout", -9, None),
        ("TIMED_OUT", "timeout", 0, None),
        ("CANCELLED", "cancelled", -15, None),
        ("CANCELLED", "cancelled", 0, None),
        ("CANCELLED", "cancelled", None, ["REQUESTED", "CANCELLED"]),
        (
            "CANCELLED",
            "subreaper_unavailable",
            None,
            ["REQUESTED", "DRY_RUN_VALIDATED", "QUEUED", "CANCELLED"],
        ),
        (
            "CANCELLED",
            "execution_validation_failed",
            None,
            ["REQUESTED", "DRY_RUN_VALIDATED", "QUEUED", "CANCELLED"],
        ),
        ("BLOCKED", "dependency_qc_not_satisfied", None, ["REQUESTED", "BLOCKED"]),
    ],
)
def test_legitimate_step_fact_table(original, status, error, exit_code, history):
    data = outcome(original["steps"][0], status, error, exit_code, history)
    assert validate(StepResult, data).model_dump(mode="json") == data


@pytest.mark.parametrize(
    "variant",
    [
        "error_layers",
        "cancel_observed_before_start",
        "blocked_exit",
        "timeout_reason",
        "cancel_reason",
        "nonzero_zero",
        "nonzero_absent",
        "setup_exit",
        "output_nonzero",
        "timeout_as_failed",
        "cancel_as_failed",
        "empty_cancellation_error",
    ],
)
def test_inconsistent_steps_reject_copy_and_construct(original, variant):
    data = outcome(original["steps"][0], "FAILED", "process_nonzero", 7)
    if variant == "error_layers":
        data["error_code"] = "timeout"
    elif variant == "cancel_observed_before_start":
        data = outcome(data, "CANCELLED", "cancelled", 0, ["REQUESTED", "CANCELLED"])
    elif variant == "blocked_exit":
        data = outcome(data, "BLOCKED", "dependency_qc_not_satisfied", -9, ["REQUESTED", "BLOCKED"])
    elif variant == "timeout_reason":
        data = outcome(data, "TIMED_OUT", "process_nonzero", 7)
    elif variant == "cancel_reason":
        data = outcome(data, "CANCELLED", "process_nonzero", 7)
    elif variant == "nonzero_zero":
        data["exit_code"] = 0
    elif variant == "nonzero_absent":
        data["exit_code"] = None
    elif variant == "setup_exit":
        data = outcome(data, "FAILED", "isolation_setup_failed", 7)
    elif variant == "output_nonzero":
        data = outcome(data, "FAILED", "output_format_invalid", 7)
    elif variant == "timeout_as_failed":
        data = outcome(data, "FAILED", "timeout", -9)
    elif variant == "cancel_as_failed":
        data = outcome(data, "FAILED", "cancelled", -9)
    else:
        data = outcome(data, "CANCELLED", None, None, ["REQUESTED", "CANCELLED"])
    safe = validate(StepResult, original["steps"][0])
    for candidate in (data, safe.model_copy(update=data), StepResult.model_construct(**data)):
        with pytest.raises(ValueError):
            validate(StepResult, candidate)


@pytest.mark.parametrize("qc", ["PASS", "FAIL", "PENDING", "UNASSESSED"])
def test_qc_is_separate_from_execution_and_all_workflow_outcomes_closed(original, qc):
    data = deepcopy(original)
    for s in data["steps"]:
        s["result"]["qc_status"] = qc
        for a, q in zip(s["result"]["artifacts"], s["qc_records"], strict=True):
            a["qc_status"] = q["status"] = qc
    expected = "SUCCEEDED" if qc == "PASS" else "PARTIAL_FAILED"
    for status in ("SUCCEEDED", "CANCELLED", "FAILED", "PARTIAL_FAILED"):
        data["status"] = status
        if status == expected:
            w = validate(WorkflowResult, data)
            assert all(s.result.status.value == "SUCCEEDED" and s.result.artifacts for s in w.steps)
        else:
            with pytest.raises(ValueError):
                validate(WorkflowResult, data)


@pytest.mark.parametrize(
    "prefix,last,reason,exit_code,expected",
    [
        (False, "FAILED", "process_nonzero", 7, "FAILED"),
        (True, "FAILED", "process_nonzero", 7, "PARTIAL_FAILED"),
        (True, "TIMED_OUT", "timeout", -9, "PARTIAL_FAILED"),
        (False, "CANCELLED", "cancelled", None, "CANCELLED"),
        (True, "CANCELLED", "cancelled", 0, "CANCELLED"),
        (False, "CANCELLED", "execution_validation_failed", None, "FAILED"),
        (False, "CANCELLED", "subreaper_unavailable", None, "FAILED"),
    ],
)
def test_workflow_priority_and_unstarted_suffix(
    original, prefix, last, reason, exit_code, expected
):
    data = deepcopy(original)
    index = int(prefix)
    history = None
    if not prefix and last == "CANCELLED":
        history = ["REQUESTED", "DRY_RUN_VALIDATED", "QUEUED", "CANCELLED"]
    data["steps"][index] = outcome(data["steps"][index], last, reason, exit_code, history)
    data["unstarted_steps"] = [s["step_id"] for s in data["steps"][index + 1 :]]
    data["steps"] = data["steps"][: index + 1]
    for status in ("SUCCEEDED", "FAILED", "PARTIAL_FAILED", "CANCELLED"):
        data["status"] = status
        if status == expected:
            w = validate(WorkflowResult, data)
            assert derive_workflow_status(w.steps, w.unstarted_steps) == expected
        else:
            with pytest.raises(ValueError):
                validate(WorkflowResult, data)


def test_legacy_export_is_exactly_readable_and_ordered(original):
    assert validate(WorkflowResult, original).model_dump(mode="json") == original
    reversed_steps = {**original, "steps": list(reversed(original["steps"]))}
    with pytest.raises(ValueError, match="derivation_order"):
        validate(WorkflowResult, reversed_steps)


def test_new_order_decision_binds_full_set_and_prefix(original):
    from packages.grounding.models import digest

    data = deepcopy(original)
    data["decision"].update(
        schema_version="execution-decision/1.1.0",
        planned_step_ids=[s["step_id"] for s in data["steps"]],
    )
    for s in data["steps"]:
        s["binding_checksum"] = digest(
            dict(
                mapping=s["logical_to_physical"],
                copies=s["copied_checksums"],
                decision=data["decision"],
            )
        )
    assert validate(WorkflowResult, data).status == "SUCCEEDED"
    for change in (
        {"steps": list(reversed(data["steps"]))},
        {"unstarted_steps": ["step_" + "a" * 64]},
        {"steps": data["steps"][:1]},
        {"attempt_id": "a" * 32},
    ):
        with pytest.raises(ValueError):
            validate(WorkflowResult, {**data, **change})


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    return runtime_files(tmp_path_factory.mktemp("06d-outcome-runtime"))


@pytest.mark.parametrize("fault", ["bootstrap", "launch_write"])
def test_actual_setup_failure_records_observed_facts(tmp_path, closure, fault):
    p = prepared(tmp_path, repository=TestLedger(), files=closure)
    run = p.executor._runner.run

    def fail_setup(base, root, *args):
        if fault == "launch_write":
            (base / "launch.json").mkdir()
        else:
            root = root / "absent"
        return run(base, root, *args)

    p.executor._runner.run = fail_setup
    w = p.executor.execute(p.plan, "setup-" + fault)
    assert w.status == "FAILED"
    step = w.steps[0]
    assert (step.result.status.value, step.error_code, step.exit_code) == (
        ("FAILED", "isolation_setup_failed", 125)
        if fault == "bootstrap"
        else ("CANCELLED", "execution_validation_failed", None)
    )


def test_failed_process_is_not_reclassified_by_late_cancel_event(tmp_path, closure):
    p = prepared(tmp_path, mode="nonzero", repository=TestLedger(), files=closure)
    cancel = Event()
    run = p.executor._runner.run

    def late_cancel(*args):
        result = run(*args)
        cancel.set()
        return result

    p.executor._runner.run = late_cancel
    w = p.executor.execute(p.plan, "late-cancel", cancel=cancel)
    assert w.status == "FAILED" and w.steps[0].exit_code == 7


def test_recovery_uses_same_partial_rule_without_running_suffix(tmp_path, closure):
    class Paused(TestLedger):
        pause = True

        def advance(self, attempt, step_id, expected, target, **kwargs):
            super().advance(attempt, step_id, expected, target, **kwargs)
            if target.value == "SUCCEEDED" and self.pause:
                self.pause = False
                raise RuntimeError("pause after committed prefix")

    ledger = Paused()
    p = prepared(tmp_path, chain=True, repository=ledger, files=closure)
    with pytest.raises(RuntimeError, match="prefix"):
        p.executor.execute(p.plan, "prefix")
    w = p.executor.recover_commit(p.plan, "prefix")
    assert w.status == "PARTIAL_FAILED" and len(w.steps) == len(w.unstarted_steps) == 1
    assert sum(status == "RUNNING" for _, status in ledger.transitions) == 1
