"""Evaluation arithmetic/entry boundaries; synthetic labels only."""

import copy

import pytest

from packages.evidence_details.models import digest
from packages.evidence_scoring.cohort import Protocol, auc, evaluate, preflight
from tests.test_evidence_records import parent  # noqa: F401
from tests.test_evidence_scoring_current import current  # noqa: F401
from tests.test_evidence_scoring_interpretation import interpreted  # noqa: F401
from tests.test_evidence_scoring_participation import graded  # noqa: F401


def protocol(score):
    p = {
        "schema_version": "graph-cohort-protocol/1.0.0",
        "policy_checksum": score["policy_checksum"],
        "members": [
            {
                "case_code": score["case_code"],
                "graph_code": "ASSET:SYNTHETIC_GRAPH",
                "view_checksum": digest("synthetic_view"),
                "input_checksum": score["input_checksum"],
                "partition": "workflow_development",
                "patient_group": None,
            }
        ],
        "classification_threshold": None,
        "selection_rule": "synthetic software fixture",
        "frozen_before_label_access": True,
    }
    return {**p, "checksum": digest(p)}


def test_one_case_is_not_cross_case_validation(current):  # noqa: F811
    _, s = current
    p = protocol(s)

    def reader(c, a):
        assert (c, a) == (s["case_code"], "ASSET:SYNTHETIC_GRAPH")
        return {"checksum": digest("synthetic_view"), "current_scoring": s}

    out = preflight(p, reader)
    assert not out["performance_ready"] and not out["labels_read"]
    assert "only_one_independently_reviewed_graph" in out["missing"]
    with pytest.raises(ValueError, match="not_ready"):
        evaluate(p, reader, {})


def test_auc_ties_and_missing_classes():
    assert auc([0, 1], [0.1, 0.9]) == 1.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert auc([0, 1], [0.9, 0.1]) == 0.0
    assert auc([1, 1], [0.2, 0.8]) is None
    with pytest.raises(ValueError):
        auc([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError):
        auc([True, 1], [0.2, 0.8])


def test_cross_case_and_unknown_policy_refused(current):  # noqa: F811
    _, s = current
    p = protocol(s)
    p["policy_checksum"] = digest("other_policy")
    p["checksum"] = digest({k: v for k, v in p.items() if k != "checksum"})
    with pytest.raises(ValueError, match="scope_or_policy"):
        preflight(p, lambda *_: {"checksum": digest("synthetic_view"), "current_scoring": s})


def test_duplicate_image_and_cross_partition_patient_refused(current):  # noqa: F811
    _, s = current
    p = protocol(s)
    p["members"].append(copy.deepcopy(p["members"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        Protocol.model_validate(p)
    p["members"][0]["patient_group"] = "synthetic_patient"
    p["members"][1].update(
        case_code="CASE:SECOND",
        graph_code="ASSET:SECOND",
        input_checksum=digest("second_image"),
        patient_group="synthetic_patient",
        partition="held_out_evaluation",
    )
    with pytest.raises(ValueError, match="partition_overlap"):
        Protocol.model_validate(p)


def test_independent_synthetic_cohort_evaluates_without_claiming_calibration(current):  # noqa: F811
    _, s = current
    p = protocol(s)
    first = p["members"][0]
    first.update(partition="held_out_evaluation", patient_group="SYNTHETIC:P1")
    second = copy.deepcopy(first)
    second.update(
        case_code="CASE:SECOND",
        graph_code="ASSET:SECOND",
        patient_group="SYNTHETIC:P2",
        input_checksum=digest("second_image"),
    )
    p["members"].append(second)
    p["checksum"] = digest({k: v for k, v in p.items() if k != "checksum"})

    def reader(c, g):
        m = next(m for m in p["members"] if m["case_code"] == c and m["graph_code"] == g)
        v = copy.deepcopy(s)
        # Explicit artificial reader for evaluator contract testing, not a saved patient record.
        v.update(case_code=c, input_checksum=m["input_checksum"])
        v["checksum"] = digest({k: val for k, val in v.items() if k != "checksum"})
        return {"checksum": m["view_checksum"], "current_scoring": v}

    out = evaluate(p, reader, {first["case_code"]: 0, second["case_code"]: 1})
    assert out["auc"] == 0.5 and out["brier"] is None and out["accuracy"] is None
    assert not out["clinical_acceptance"] and not out["patient_labels_saved_to_graph"]
