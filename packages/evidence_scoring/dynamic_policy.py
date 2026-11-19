"""Explicit dynamic denominator; unchanged bipolar numerical aggregation kernel."""

from packages.evidence_scoring.continuous_policy import POLICY as NUMERICAL_POLICY

POLICY = {
    "version": "source-driven-scoring/1.0.0",
    "numerical_policy": NUMERICAL_POLICY,
    "requirements": "frozen_source_backed_snapshot_not_evidence_rows",
    "evidence": "only_owner_confirmed_obtained_values",
    "qualitative": "source_bound_assessment_without_invented_numeric_threshold",
    "partial": "exact_verified_request_subset_with_explicit_pending_ids",
    "zero": "normal_evidence_limited_result_without_disease_direction",
    "model": "immutable_output_direction_with_new_case_reviewed_OLA",
    "removal": "same_requirement_denominator_and_original_dependency_closure",
}
