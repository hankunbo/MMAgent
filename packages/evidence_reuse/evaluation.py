"""Read-only fixed-evidence probes using the existing numerical owner."""

import json
from pathlib import Path

from packages.evidence_scoring.continuous import summarize
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.grounding.models import digest

PROTOCOL = Path("configs/research/upstream_mechanism_protocol_v1.json")


def evaluate_saved(report):
    if report.get("schema_version") != "dynamic-case-report/1.0.0":
        raise ValueError("mechanism_report_version")
    scores = report.get("scores")
    if not scores or scores.get("schema_version") not in {
        "case-research-scoring/7.0.0",
        "case-research-scoring/8.0.0",
    }:
        raise ValueError("mechanism_scores_missing_or_unsupported")
    if scores["checksum"] != digest({k: v for k, v in scores.items() if k != "checksum"}):
        raise ValueError("mechanism_scores_checksum")
    rows = [ContinuousReview.model_validate(r) for r in scores["reviews"]]
    full = summarize(rows)
    if full != scores["root"]["summary"]:
        raise ValueError("mechanism_baseline_reconstruction")
    duplicate = summarize(rows + rows)
    if duplicate != full:
        raise ValueError("mechanism_duplicate_invariance")
    removals = []
    for item in scores["removals"]:
        derived = summarize(rows, removed_resources={item["resource"]})
        if derived != item["summary"]:
            raise ValueError("mechanism_removal_reconstruction")
        removals.append(
            dict(resource=item["resource"], summary=derived, frozen_coverage=item.get("coverage"))
        )
    protocol = json.loads(PROTOCOL.read_bytes())
    if protocol.get("schema_version") != "upstream-mechanism-protocol/1.0.0":
        raise ValueError("mechanism_protocol_version")
    return dict(
        schema_version="upstream-mechanism-result/1.0.0",
        report_checksum=digest(report),
        protocol_checksum=digest(protocol),
        evaluation_kind="retrospective_fixed_evidence_software_probe",
        not_prospective_trial=True,
        original_scores_modified=False,
        original_requirement_denominator=len(scores["requirements"]),
        full=full,
        exact_duplicate_equal=duplicate == full,
        risk_removed=summarize(rows, removed_ids={r.id for r in rows if r.role == "risk"}),
        nonrisk_removed=summarize(rows, removed_ids={r.id for r in rows if r.role != "risk"}),
        resource_removals=removals,
        labels_read=False,
        api_requests=0,
        model_starts=0,
        prospective_experiments_pending=protocol["future_end_to_end_arms"],
        clinical_effectiveness_established=False,
    )
