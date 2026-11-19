"""Task-explicit use of the unchanged bipolar/quality numerical kernel."""

from packages.evidence_scoring.quality_policy import POLICY as QUALITY_POLICY

SCOPES = ("observed_input_research", "task_requirement_research")
POLICY = {
    "version": "task-relative-quality-scoring/1.0.0",
    "base": QUALITY_POLICY,
    "target": "one_source_bound_research_hypothesis_per_aggregate",
    "coverage_scopes": list(SCOPES),
    "ordinal": "not_class_probabilities_or_diagnostic_grading",
    "change": "observed_change_not_future_prediction_or_causality",
}
