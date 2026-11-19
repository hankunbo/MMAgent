"""Demand coverage is not Review completion or a new authorization."""

from copy import deepcopy

import pytest

from packages.multimodal_research.demand_coverage import project, render_summary


def report():
    return dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(case_code="CASE:TEST", phases=[]),
        task=dict(target_code="GOAL:TEST"),
        completion=dict(all_obtained_reviewed=True),
        requirements=[
            dict(id="one", examination="<observed>", observation=dict(status="obtained")),
            dict(
                id="two",
                examination="No input",
                source_requirement=dict(state="input_gap", reason="patient_input_missing:VF"),
            ),
            dict(
                id="three",
                examination="Scheduled",
                source_requirement=dict(state="scheduled", method="vision", component="fundus"),
            ),
            dict(id="four", examination="Unknown", source_requirement=dict(state="scheduled")),
        ],
        acquisition_observations={"four": dict(status="unknown", reference=dict(code="OBS:FOUR"))},
        scores=dict(reviews=[dict(id="one", judgment=dict(value=0, reason="Balanced comparison"))]),
        acquisition_budget_stop=dict(
            reason="budget_reserved_for_pending_review", stop_acquisition=True
        ),
    )


def test_review_complete_does_not_mean_acquisition_complete_and_no_score_changes():
    data = report()
    original = deepcopy(data)
    result = project(data)
    assert data == original
    rows = {r["id"]: r for r in result["rows"]}
    assert rows["one"]["state"] == "reviewed_neutral"
    assert rows["two"]["state"] == "input_missing"
    assert rows["three"]["state"] == "scheduled_without_observation"
    assert rows["four"]["state"] == "observation_not_interpretable"
    assert rows["three"]["stop_context"]["reason"] == "budget_reserved_for_pending_review"
    assert rows["three"]["attempt_status"] == "not_recorded_in_report"
    assert result["counts"]["obtained"] == 1
    assert result["counts"]["scored"] == 1
    assert result["counts"]["nonzero"] == 0
    assert result["fully_obtained"] is False
    assert result["current_execution_authorization"] is False
    assert "&lt;observed&gt;" in render_summary(result)


@pytest.mark.parametrize(
    "value,state",
    [
        (0.001, "reviewed_directional"),
        (-0.001, "reviewed_directional"),
        (None, "reviewed_unassessable"),
    ],
)
def test_small_signed_and_null_scores_are_preserved(value, state):
    data = report()
    data["scores"]["reviews"][0]["judgment"]["value"] = value
    result = project(data)
    assert result["rows"][0]["state"] == state
    assert result["rows"][0]["e"] == value


def test_attempted_failure_is_not_missing_input_or_unknown_observation():
    data = report()
    data["result"]["phases"] = [
        dict(target="three", run_id="RUN:TEST", accepted=False, error="socket_timeout")
    ]
    result = project(data)
    row = result["rows"][2]
    assert row["state"] == "attempt_recorded_without_observation"
    assert row["attempt_status"] == "phase_recorded"
    assert row["attempts"][0]["error"] == "socket_timeout"
    assert row["next_step"] == "verify_terminal_attempt_before_alternative"


@pytest.mark.parametrize("fault", ["duplicate", "unknown_review", "observation_conflict"])
def test_inconsistent_report_bindings_rejected(fault):
    data = report()
    if fault == "duplicate":
        data["requirements"].append(data["requirements"][0])
    elif fault == "unknown_review":
        data["scores"]["reviews"][0]["id"] = "not-a-requirement"
    else:
        data["acquisition_observations"]["one"] = dict(status="missing")
    with pytest.raises(ValueError, match="demand_coverage"):
        project(data)


def test_no_pending_review_is_not_proof_of_neutral_or_unattempted():
    data = report()
    data["scores"] = None
    result = project(data)
    assert result["rows"][0]["state"] == "obtained_pending_review"
    assert result["rows"][0]["e"] is None
    assert result["counts"]["scored"] == 0


def test_missing_report_demand_contract_rejected():
    with pytest.raises(ValueError, match="demand_coverage_report_contract"):
        project({"unrelated": True})
