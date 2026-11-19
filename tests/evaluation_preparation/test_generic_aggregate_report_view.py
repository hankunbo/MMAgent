"""A whole-case report reads both origins without widening dispatch paths."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.dynamic_report import GenericAggregateReportView
from packages.evaluation_preparation.generic_acquisition_successor import AcquisitionSuccessor
from packages.evaluation_preparation.generic_review import GenericReview
from tests.evaluation_preparation.test_partial_observation import proposal


def fixture(tmp_path):
    old = SimpleNamespace(phase_path=lambda t: tmp_path / "original" / t)
    source = AcquisitionSuccessor.__new__(AcquisitionSuccessor)
    source.output = tmp_path / "increment"
    source.selected_target = "criterion:NEW"
    source.acquisition = old
    source.task_plan = [dict(id=t, action="vision") for t in ["criterion:OLD", "criterion:NEW"]]
    observation = SimpleNamespace(status="obtained", value=proposal().observation_value())
    source.accepted_observations = lambda: {r["id"]: observation for r in source.task_plan}
    source.read_model = lambda: None
    for target, where in [
        ("criterion:OLD", old.phase_path("criterion:OLD")),
        ("criterion:NEW", source.phase_path("criterion:NEW")),
    ]:
        where.mkdir(parents=True)
        (where / "result.json").write_text(
            json.dumps(dict(target=target, accepted=True, run_id=target))
        )
    out = tmp_path / "whole"
    (out / "review").mkdir(parents=True)
    (out / "review/review-report.json").write_text(
        json.dumps(
            dict(
                scores=dict(
                    reviews=[
                        dict(
                            id="criterion:OLD",
                            observation=dict(value=0.7),
                            judgment=dict(value=0.2),
                        ),
                        dict(
                            id="criterion:NEW",
                            observation=dict(value=0.4),
                            judgment=dict(value=-0.15),
                        ),
                    ]
                )
            )
        )
    )
    return source, out


def test_both_original_executions_and_aggregate_directions_preserved(tmp_path):
    source, out = fixture(tmp_path)
    before = source.output
    view = GenericAggregateReportView(source, out)
    rows = {r["id"]: r for r in view.process_evaluations()}
    assert set(rows) == {"criterion:OLD", "criterion:NEW"}
    assert rows["criterion:OLD"]["execution"]["run_id"] == "criterion:OLD"
    assert rows["criterion:NEW"]["execution"]["run_id"] == "criterion:NEW"
    assert rows["criterion:OLD"]["disease_direction"] == 0.2
    assert rows["criterion:NEW"]["disease_direction"] == -0.15
    assert all(r["availability_score"] == 1 for r in rows.values())
    assert source.output == before
    assert set(view.accepted_observations()) == set(rows)
    with pytest.raises(ValueError, match="generic_acquisition_unselected_phase"):
        source.phase_path("criterion:OLD")
    assert view.review_report_path == out / "review/review-report.json"


def test_review_only_wrapper_walks_to_original_observation(tmp_path):
    source, out = fixture(tmp_path)
    wrapper = GenericReview.__new__(GenericReview)
    wrapper.acquisition = source
    view = GenericAggregateReportView(wrapper, out)
    assert view.observation_phase_path("criterion:NEW") == source.phase_path("criterion:NEW")
    assert view.observation_phase_path("criterion:OLD") == source.acquisition.phase_path(
        "criterion:OLD"
    )


def test_cycle_rejected_instead_of_unbounded_walk(tmp_path):
    source, out = fixture(tmp_path)
    source.acquisition = source
    with pytest.raises(ValueError, match="aggregate_report_observation_cycle"):
        GenericAggregateReportView(source, out).observation_phase_path("criterion:OLD")


def test_missing_old_execution_stays_missing_not_current_selected(tmp_path):
    source, out = fixture(tmp_path)
    source.acquisition.phase_path("criterion:OLD").joinpath("result.json").unlink()
    rows = {r["id"]: r for r in GenericAggregateReportView(source, out).process_evaluations()}
    assert "execution" not in rows["criterion:OLD"]
    assert rows["criterion:NEW"]["execution"]["target"] == "criterion:NEW"
