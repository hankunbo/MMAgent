"""The recovered-case gate counts coverage, never demands a nonzero quota."""

import copy
import runpy

import pytest

from packages.grounding.models import digest


@pytest.fixture
def coverage():
    return runpy.run_path(
        "docs/acceptance/generic-memory256-compat-ko9wqM1D/verify_recovery_gate.py"
    )["coverage"]


def test_neutral_null_observation_unknown_and_original_input_gap_are_distinct(coverage):
    observation = dict(
        status="unknown", reference={"code": "OBS:ONE"}, value="visible band, boundary unclear"
    )
    report = dict(
        requirements=[dict(id=i) for i in ("zero", "null", "unknown", "missing", "prior")],
        scores=dict(
            reviews=[
                dict(id="zero", judgment=dict(value=0)),
                dict(id="null", judgment=dict(value=None)),
            ]
        ),
        process_evaluations=[
            dict(
                id="missing",
                action="record_input_gap",
                reason="physical_scale_not_provided",
                sources=[{"code": "SOURCE:ONE"}],
            )
        ],
    )
    report["requirements"][2]["unscored_observation"] = observation
    report["requirements"][4]["unscored_observation"] = observation
    audit = dict(
        retained_historical_reviews={
            "prior": dict(
                source_definition_unchanged=True,
                observation_checksum=digest(observation),
                original_reason="target relation absent",
            )
        }
    )
    before = copy.deepcopy(report)
    rows = coverage(report, audit)
    assert [r["disposition"] for r in rows] == [
        "reviewed_neutral",
        "reviewed_null",
        "saved_observation_not_yet_interpretable",
        "owner_recorded_input_gap",
        "retained_prior_null_not_new_vote",
    ]
    assert all(
        r["new_direction_invented"] is False and r["replay_authorized"] is False for r in rows
    )
    assert report == before


@pytest.mark.parametrize(
    "fault",
    [
        "unattempted",
        "duplicate",
        "changed_definition",
        "changed_observation",
        "missing_proof",
        "unbound_review",
    ],
)
def test_uncovered_or_unbound_requirement_is_not_a_pass(coverage, fault):
    obs = dict(status="obtained", reference={"code": "OBS:ONE"}, value=1)
    report = dict(requirements=[dict(id="one", observation=obs)], scores=dict(reviews=[]))
    audit = dict(
        retained_historical_reviews={
            "one": dict(
                source_definition_unchanged=True,
                observation_checksum=digest(obs),
                original_reason="original null",
            )
        }
    )
    if fault == "unattempted":
        audit["retained_historical_reviews"] = {}
    elif fault == "duplicate":
        report["requirements"] *= 2
    elif fault == "changed_definition":
        audit["retained_historical_reviews"]["one"]["source_definition_unchanged"] = False
    elif fault == "changed_observation":
        obs["value"] = 2
    elif fault == "missing_proof":
        audit["retained_without_review_proof"] = ["one"]
    else:
        report["scores"]["reviews"] = [dict(id="other", judgment=dict(value=0.1))]
    with pytest.raises(ValueError, match="recovery_gate_"):
        coverage(report, audit)
