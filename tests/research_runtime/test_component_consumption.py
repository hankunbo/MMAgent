"""Narrow current QC-policy checks; not an original-owner integration substitute."""

from types import SimpleNamespace

import pytest

from packages.research_runtime.component_steps import ComponentPlannerSteps


@pytest.mark.parametrize("mode", ["same", "changed", "missing", "receipt_missing"])
@pytest.mark.parametrize("consumption", [False, True])
def test_current_consumption_requirements(mode, consumption):
    owner = object.__new__(ComponentPlannerSteps)
    owner._key = lambda state: "saved"
    original = {"policy": "original", "outputs": [{"qc": "policy1"}]}
    owner.documents = SimpleNamespace(
        read=lambda key: (
            None if mode == "receipt_missing" else ({"execution_requirements": original},)
        )
    )
    calls = []

    def requirements(plan):
        calls.append(plan)
        if mode == "missing":
            raise ValueError("qc_definition_not_installed")
        return {**original, "policy": "changed"} if mode == "changed" else original

    executor = SimpleNamespace(
        task_requirements=requirements, consumption_requirements=requirements
    )
    errors = {
        "changed": "component_execution_requirements_changed",
        "missing": "qc_definition_not_installed",
        "receipt_missing": "component_prepare_receipt_required",
    }
    if mode == "same":
        owner._consumption_requirements(None, "plan", executor, consumption=consumption)
    else:
        with pytest.raises(ValueError, match=errors[mode]):
            owner._consumption_requirements(None, "plan", executor, consumption=consumption)
    assert calls == ([] if mode == "receipt_missing" else ["plan"])


@pytest.mark.parametrize("changed", [False, True])
def test_consumption_last_requirement_callback_rechecks_current(monkeypatch, changed):
    """Narrow ordering test; real contract/owner persistence is covered by PG."""
    from packages.research_runtime import component_steps

    monkeypatch.setattr(component_steps, "validate", lambda cls, value: value)
    state = SimpleNamespace(
        binding=SimpleNamespace(run_id="run"),
        pending=None,
        confirmed=1,
        graph="graph",
        result_refs=("result",),
    )
    workflow = SimpleNamespace(status="SUCCEEDED")
    owner = object.__new__(ComponentPlannerSteps)
    owner.sessions = SimpleNamespace(read=lambda run: state)
    owner._saved = lambda value: ("plan", SimpleNamespace(run_id="child"))
    owner.executor_for = lambda plan: SimpleNamespace(
        recover=lambda *args: {"result": workflow},
        confirmed_result=lambda *args: workflow,
    )
    owner._result = lambda value: SimpleNamespace(refs=("result",))
    calls = []

    def requirements(*args, **kwargs):
        calls.append("requirements")

    def current(binding):
        if changed and calls.count("requirements") == 2:
            raise ValueError("component_admission_policy_changed")
        return "graph"

    owner.current, owner._consumption_requirements = current, requirements
    if changed:
        with pytest.raises(ValueError, match="component_admission_policy_changed"):
            owner.read_consumable_result(state, "key")
    else:
        assert owner.read_consumable_result(state, "key") is workflow
    assert calls == ["requirements", "requirements"]


@pytest.mark.parametrize("mode", ["ok", "source", "qc", "control", "unknown", "reader_error"])
def test_local_reader_bracket_preserves_post_return_checks(monkeypatch, mode):
    from packages.grounding.models import Value
    from packages.research_runtime import component_steps

    class Result(Value):
        status: str = "SUCCEEDED"

    monkeypatch.setattr(component_steps, "validate", lambda cls, value: value)
    state = SimpleNamespace(
        binding=SimpleNamespace(run_id="run"),
        pending=None,
        confirmed=1,
        graph="graph",
        result_refs=("result",),
    )
    workflow = Result()
    owner = object.__new__(ComponentPlannerSteps)
    calls = []
    owner.sessions = SimpleNamespace(
        read=lambda run: None if mode == "control" and "reader" in calls else state
    )
    owner._saved = lambda value: ("plan", SimpleNamespace(run_id="child"))
    owner.executor_for = lambda plan: SimpleNamespace(
        recover=lambda *args: None if mode == "unknown" else {"result": workflow},
        confirmed_result=lambda *args: workflow,
    )
    owner._result = lambda value: SimpleNamespace(refs=("result",))

    def current(binding):
        calls.append("current")
        if mode == "source" and "reader" in calls:
            raise ValueError("source_changed")
        return "graph"

    def requirements(*args, **kwargs):
        calls.append("qc")
        if mode == "qc" and "reader" in calls:
            raise ValueError("qc_changed")

    def reader(value):
        assert value == workflow
        assert calls == ["current", "qc"]
        calls.append("reader")
        if mode == "reader_error":
            raise OSError("decode_failed")
        return {"measurement": 0.5}

    owner.current, owner._consumption_requirements = current, requirements
    if mode == "ok":
        assert owner.consume_confirmed_result(state, "key", reader) == (
            workflow,
            {"measurement": 0.5},
        )
        assert calls == ["current", "qc", "reader", "current", "qc", "current"]
    else:
        errors = {
            "source": "source_changed",
            "qc": "qc_changed",
            "control": "component_consumption_control_changed",
            "unknown": "component_confirmed_result_required",
            "reader_error": "decode_failed",
        }
        with pytest.raises(OSError if mode == "reader_error" else ValueError, match=errors[mode]):
            owner.consume_confirmed_result(state, "key", reader)
        if mode == "unknown":
            assert "reader" not in calls
