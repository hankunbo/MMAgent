"""Additive quality weighting; the existing bipolar aggregation kernel is reused."""

from packages.evidence_scoring.dynamic_policy import POLICY as DYNAMIC_POLICY

POLICY = {
    "version": "quality-weighted-dynamic-scoring/1.0.0",
    "base": DYNAMIC_POLICY,
    "effective_observation": "raw_O * owner_quality_factor",
    "weight": "raw_O * L * A * Q",
    "missing": "quality_zero_and_gap_report_not_invented_patient_value",
    "uncertainty": "original_kernel_on_quality_adjusted_observation",
}
