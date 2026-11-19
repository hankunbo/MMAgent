"""Frozen-cohort evaluation of saved graph scores, not model-only predictions.

Label-free preflight is the default. Labels enter only the explicit evaluator.
Uncalibrated evidence scores permit ranking evaluation, not probability metrics.
"""

import math
from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict, digest
from packages.evidence_details.records_models import SHA, Code
from packages.evidence_scoring.current_models import CurrentScores


class Member(Strict):
    case_code: Code
    graph_code: Code
    view_checksum: SHA
    input_checksum: SHA
    partition: Literal["workflow_development", "calibration", "held_out_evaluation"]
    patient_group: Code | None


class Protocol(Strict):
    schema_version: Literal["graph-cohort-protocol/1.0.0"]
    policy_checksum: SHA
    members: list[Member] = Field(min_length=1, max_length=1000)
    classification_threshold: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    selection_rule: str = Field(min_length=1, max_length=4000)
    frozen_before_label_access: bool
    checksum: SHA

    @model_validator(mode="after")
    def unique(self):
        for key in ("case_code", "input_checksum", "graph_code"):
            if len({getattr(m, key) for m in self.members}) != len(self.members):
                raise ValueError("cohort_duplicate:" + key)
        groups = {}
        for m in self.members:
            if m.patient_group is not None:
                if m.patient_group in groups and groups[m.patient_group] != m.partition:
                    raise ValueError("cohort_patient_partition_overlap")
                groups[m.patient_group] = m.partition
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("cohort_protocol_checksum")
        return self


def _protocol(value):
    return Protocol.model_validate(
        value.model_dump(mode="json") if isinstance(value, Protocol) else value
    )


def load(protocol, reader):
    p = _protocol(protocol)
    scores = {}
    for m in p.members:
        view = reader(m.case_code, m.graph_code)
        if view["checksum"] != m.view_checksum or "current_scoring" not in view:
            raise ValueError("cohort_exact_current_graph_required")
        s = CurrentScores.model_validate(view["current_scoring"]).model_dump(mode="json")
        if (
            s["case_code"] != m.case_code
            or s["input_checksum"] != m.input_checksum
            or s["policy_checksum"] != p.policy_checksum
        ):
            raise ValueError("cohort_scope_or_policy_drift")
        scores[m.case_code] = s
    return p, scores


def preflight(protocol, reader):
    p, scores = load(protocol, reader)
    available = [
        m.case_code
        for m in p.members
        if scores[m.case_code]["root"]["summary"]["confidence"]["value"] is not None
    ]
    review_counts = {
        m.case_code: sum(r["judgment"]["value"] is not None for r in scores[m.case_code]["reviews"])
        for m in p.members
    }
    problems = []
    if len(p.members) < 2:
        problems.append("only_one_independently_reviewed_graph")
    if any(m.patient_group is None for m in p.members):
        problems.append("patient_independence_not_established")
    if not p.frozen_before_label_access:
        problems.append("protocol_not_frozen_before_labels")
    if not any(m.partition == "held_out_evaluation" for m in p.members):
        problems.append("no_held_out_graph_cohort")
    if len(available) != len(p.members):
        problems.append("some_graphs_have_no_case_reviewed_direction")
    return {
        "schema_version": "graph-cohort-preflight/1.0.0",
        "protocol_checksum": p.checksum,
        "complete_graph_count": len(scores),
        "direction_available_count": len(available),
        "case_review_counts": review_counts,
        "missing": problems,
        "performance_ready": not problems,
        "labels_read": False,
        "calibration_status": "not_fitted_no_probability_output",
        "model_only_validation_substituted": False,
    }


def auc(labels, values):
    """Mann-Whitney ranking statistic, ties contribute one half."""
    if len(labels) != len(values) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("cohort_label_domain")
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("cohort_score_domain")
    positive = [v for y, v in zip(labels, values, strict=True) if y == 1]
    negative = [v for y, v in zip(labels, values, strict=True) if y == 0]
    if not positive or not negative:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (
        len(positive) * len(negative)
    )


def evaluate(protocol, reader, labels):
    """Explicit evaluator-only labels; no write to the case or Knowledge."""
    p, scores = load(protocol, reader)
    if not p.frozen_before_label_access:
        raise ValueError("cohort_label_access_before_freeze")
    selected = [m for m in p.members if m.partition == "held_out_evaluation"]
    if len(selected) < 2 or any(m.patient_group is None for m in p.members):
        raise ValueError("cohort_independent_evaluation_not_ready")
    if len({m.patient_group for m in selected}) != len(selected):
        raise ValueError("cohort_v1_requires_one_case_per_patient")
    if set(labels) != {m.case_code for m in selected}:
        raise ValueError("cohort_exact_label_set_required")
    ys, values, without_age = [], [], []
    for m in selected:
        s = scores[m.case_code]
        value = s["root"]["summary"]["confidence"]["value"]
        if value is None:
            raise ValueError("cohort_missing_score_report_preflight_first")
        ys.append(labels[m.case_code])
        values.append(value)
        without_age.append(s["conclusion"]["without_age"]["confidence"]["value"])
    result = {
        "schema_version": "graph-cohort-evaluation/1.0.0",
        "protocol_checksum": p.checksum,
        "n": len(values),
        "positives": sum(y == 1 for y in ys),
        "auc": auc(ys, values),
        "without_age_auc": auc(ys, without_age)
        if all(v is not None for v in without_age)
        else None,
        "accuracy": None,
        "sensitivity": None,
        "specificity": None,
        "brier": None,
        "calibration": "requires_separate_fitted_probability_mapping_and_independent_evaluation",
        "threshold": p.classification_threshold,
        "clinical_acceptance": False,
        "patient_labels_saved_to_graph": False,
    }
    if p.classification_threshold is not None:
        predictions = [v >= p.classification_threshold for v in values]
        pos, neg = sum(ys), len(ys) - sum(ys)
        result.update(
            accuracy=sum(int(a) == b for a, b in zip(predictions, ys, strict=True)) / len(ys),
            sensitivity=sum(a and b == 1 for a, b in zip(predictions, ys, strict=True)) / pos
            if pos
            else None,
            specificity=sum(not a and b == 0 for a, b in zip(predictions, ys, strict=True)) / neg
            if neg
            else None,
        )
    return result
