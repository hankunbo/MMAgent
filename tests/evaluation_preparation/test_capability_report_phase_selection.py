"""Reporting selection regression reproducing the real capability export fault."""

import json

import pytest

from packages.evaluation_preparation.comprehensive_campaign import ComprehensiveCampaign


def campaign(tmp_path):
    value = object.__new__(ComprehensiveCampaign)
    value.output = tmp_path
    value.accepted_observations = lambda: {}
    return value


@pytest.mark.parametrize("action", ["record_input_gap", "review_measurement"])
def test_unselected_requirement_is_displayed_without_asking_for_execution(tmp_path, action):
    value = campaign(tmp_path)
    value.visual_targets = ()
    value.task_plan = [dict(id="question", action=action, reason="current_plan")]

    def forbidden(target):
        raise ValueError("generic_capability_unselected_phase")

    value.phase_path = forbidden
    rows = value.process_evaluations()
    assert len(rows) == 1 and rows[0]["id"] == "question"
    assert rows[0]["disease_direction"] is None
    assert "execution" not in rows[0]
    assert "technical_errors" not in rows[0]


def test_selected_phase_retains_its_exact_saved_execution(tmp_path):
    value = campaign(tmp_path)
    value.visual_targets = ("selected",)
    value.task_plan = [dict(id="selected", action="observe_partial")]
    folder = tmp_path / "selected"
    folder.mkdir()
    saved = dict(accepted=True, run_id="RUN:ORIGINAL", target="selected")
    (folder / "result.json").write_text(json.dumps(saved))
    value.phase_path = lambda target: folder
    rows = value.process_evaluations()
    assert rows[0]["execution"] == saved
    assert rows[0]["disease_direction"] is None
