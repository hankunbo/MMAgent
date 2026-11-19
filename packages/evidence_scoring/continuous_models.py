"""V2 research judgments; preserve the v1 contract and its historical meaning."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict, digest
from packages.evidence_scoring.models import Channel, Coverage, Rating, ResourceBinding, Unit

Signed = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class Judgment(Strict):
    value: Signed | None
    lower: Signed | None
    upper: Signed | None
    reason: str = Field(min_length=1, max_length=2000)
    missing_reason: str | None = Field(default=None, min_length=1, max_length=512)
    method: Literal["manual_source_bound_review", "fixed_model_margin", "not_assessed"]

    @model_validator(mode="after")
    def bounds(self):
        if self.value is None:
            if self.lower is not None or self.upper is not None or not self.missing_reason:
                raise ValueError("continuous_judgment_missing_conflict")
            if self.method != "not_assessed":
                raise ValueError("continuous_judgment_method_conflict")
        elif (
            self.lower is None
            or self.upper is None
            or not self.lower <= self.value <= self.upper
            or self.missing_reason is not None
            or self.method == "not_assessed"
        ):
            raise ValueError("continuous_judgment_bounds_conflict")
        return self


class ContinuousReview(Strict):
    id: str
    node: str
    title: str
    kind: str
    assessment_checksum: str
    source_binding_checksum: str
    observation: Rating
    relation: Rating
    applicability: Rating
    role: Literal["diagnostic", "risk", "measurement", "model", "visual", "association"]
    target: Literal["glaucoma_research_hypothesis"]
    judgment: Judgment
    direction: Literal["support", "refute", "neutral", "unavailable"]
    observation_available: bool
    criterion_checked: bool
    reason: str
    correlation_group: str
    dependence_block: str
    dependency_paths: list[list[str]] = Field(max_length=8)
    weight: Unit | None
    support: Unit
    refutation: Unit

    @model_validator(mode="after")
    def contributions(self):
        values = [self.observation.value, self.relation.value, self.applicability.value]
        weight = values[0] * values[1] * values[2] if all(v is not None for v in values) else None
        effect = self.judgment.value
        expected_direction = (
            "unavailable"
            if effect is None
            else "support"
            if effect > 0
            else "refute"
            if effect < 0
            else "neutral"
        )
        if self.weight != weight or self.direction != expected_direction:
            raise ValueError("continuous_review_weight_or_direction")
        if effect is not None and (
            weight is None
            or not self.observation_available
            or not any(self.dependency_paths)
            or self.role not in {"diagnostic", "risk", "model", "association", "visual"}
        ):
            raise ValueError("continuous_judgment_without_evidence")
        support = weight * max(effect, 0.0) if effect is not None else 0.0
        refutation = weight * max(-effect, 0.0) if effect is not None else 0.0
        if self.support != support or self.refutation != refutation:
            raise ValueError("continuous_review_contribution_mismatch")
        return self


class TaskContinuousReview(ContinuousReview):
    """New target namespace; identical arithmetic, not a change to legacy semantics."""

    target: str = Field(pattern=r"^GOAL:[A-Za-z0-9_:.\-]+$")


class Confidence(Strict):
    value: Unit | None
    lower: Unit | None
    upper: Unit | None
    status: Literal["available", "unavailable"]
    direction: Literal["support_side", "refutation_side", "balanced", "unavailable"]
    stability: Literal["support_side", "refutation_side", "direction_sensitive", "unavailable"]
    kind: Literal["uncalibrated_research_evidence_score"]
    basis: list[Literal["diagnostic", "risk", "model", "association", "visual"]]
    missing_reason: str | None

    @model_validator(mode="after")
    def consistency(self):
        if self.value is None:
            if (
                self.lower is not None
                or self.upper is not None
                or self.status != "unavailable"
                or self.direction != "unavailable"
                or self.stability != "unavailable"
                or not self.missing_reason
                or self.basis
            ):
                raise ValueError("continuous_confidence_missing_conflict")
        else:
            if (
                self.lower is None
                or self.upper is None
                or not self.lower <= self.value <= self.upper
                or self.status != "available"
                or self.missing_reason is not None
                or not self.basis
            ):
                raise ValueError("continuous_confidence_bounds_conflict")
            expected = (
                "support_side"
                if self.value > 0.5
                else ("refutation_side" if self.value < 0.5 else "balanced")
            )
            stability = (
                "support_side"
                if self.lower > 0.5
                else ("refutation_side" if self.upper < 0.5 else "direction_sensitive")
            )
            if self.direction != expected or self.stability != stability:
                raise ValueError("continuous_confidence_direction_conflict")
        return self


class ContinuousSummary(Strict):
    assessment_ids: list[str]
    diagnostic: Channel
    risk: Channel
    model: Channel
    combined: Channel
    confidence: Confidence
    criterion_coverage: Coverage
    observation_coverage: Coverage
    judgment_coverage: Coverage
    contributing_ids: list[str]


class ContinuousRemoval(Strict):
    id: str
    title: str
    unavailable_resources: list[str]
    removed_assessments: list[str]
    affected_assessments: list[str]
    after: ContinuousSummary
    changed_nodes: list[str]
    diagnostic_support_delta: float = Field(ge=-1, le=1, allow_inf_nan=False)
    risk_support_delta: float = Field(ge=-1, le=1, allow_inf_nan=False)
    confidence_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    criterion_coverage_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    explanation: str


class ContinuousScores(Strict):
    schema_version: Literal["case-research-scoring/2.0.0"]
    policy_version: Literal["argument-review/2.0.0"]
    policy_checksum: str
    review_bundle_checksum: str
    review_status: Literal["bound_case_review", "not_reviewed"]
    case_code: str
    input_checksum: str
    base_checksum: str
    trace_checksum: str
    reviewer: Literal["assistant_manual_simulation_of_llm"]
    reviewed_at: Literal["2026-09-20"]
    purpose: Literal["research_explanation_not_clinical_authorization"]
    disease_probability: None = None
    clinical_acceptance: Literal[False] = False
    reviews: list[ContinuousReview] = Field(max_length=512)
    resources: dict[str, list[ResourceBinding]]
    nodes: dict[str, ContinuousSummary]
    removals: list[ContinuousRemoval] = Field(max_length=512)
    method_notes: list[str]
    checksum: str

    @model_validator(mode="after")
    def seal(self):
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("continuous_score_checksum_mismatch")
        ids = [r.id for r in self.reviews]
        if len(ids) != len(set(ids)) or any(
            len(n.assessment_ids) != len(set(n.assessment_ids))
            or not set(n.assessment_ids) <= set(ids)
            for n in self.nodes.values()
        ):
            raise ValueError("continuous_score_duplicate_or_unknown_review")
        return self
