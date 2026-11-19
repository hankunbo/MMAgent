"""Strict candidate review format, separate from accepted production score records."""

from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict
from packages.evidence_scoring.continuous_models import Judgment
from packages.evidence_scoring.models import Rating


class ReviewItem(Strict):
    assessment_id: str = Field(min_length=1, max_length=128)
    status: Literal[
        "assessed",
        "reference_missing",
        "method_mismatch",
        "context_only",
        "observation_ungradable",
        "input_missing",
        "intermediate",
    ]
    observation_available: bool
    observation_citations: list[str] = Field(max_length=32)
    reference_citations: list[str] = Field(max_length=32)
    comparison: str = Field(min_length=1, max_length=2000)
    observation: Rating
    relation: Rating
    applicability: Rating
    judgment: Judgment
    limitation_allocation: str = Field(min_length=1, max_length=2000)
    unresolved: list[str] = Field(max_length=32)

    @model_validator(mode="after")
    def assessed_evidence(self):
        if (self.status == "assessed") != (self.judgment.value is not None):
            raise ValueError("review_status_judgment_mismatch")
        if self.status == "assessed" and (
            not self.observation_available
            or not self.observation_citations
            or not self.reference_citations
            or any(r.value is None for r in (self.observation, self.relation, self.applicability))
        ):
            raise ValueError("review_assessed_requires_evidence")
        if self.status == "input_missing" and self.observation_available:
            raise ValueError("review_missing_input_has_observation")
        return self


class DirectionAssessment(Strict):
    """Research explanation, not an automatically verified medical conclusion."""

    schema_version: Literal["direction-assessment/1.0.0"] = "direction-assessment/1.0.0"
    direction: Literal["support", "refute", "neutral", "unassessable"]
    observed_features: str = Field(min_length=1, max_length=2000)
    source_relation: str = Field(min_length=1, max_length=2000)
    supporting_interpretation: str = Field(min_length=1, max_length=2000)
    opposing_interpretation: str = Field(min_length=1, max_length=2000)
    alternatives: tuple[str, ...] = Field(min_length=1, max_length=8)
    magnitude_reason: str = Field(min_length=1, max_length=2000)
    neutral_reason: str | None = Field(default=None, min_length=1, max_length=2000)
    unassessable_reason: str | None = Field(default=None, min_length=1, max_length=2000)
    quality_limitation: str = Field(min_length=1, max_length=1000)
    resolvable_gap: str | None = Field(default=None, min_length=1, max_length=1000)
    next_action: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def explanation(self):
        texts = (
            self.observed_features,
            self.source_relation,
            self.supporting_interpretation,
            self.opposing_interpretation,
            self.magnitude_reason,
            self.quality_limitation,
            *self.alternatives,
            *(
                s
                for s in (
                    self.neutral_reason,
                    self.unassessable_reason,
                    self.resolvable_gap,
                    self.next_action,
                )
                if s is not None
            ),
        )
        if any(not s.strip() for s in texts):
            raise ValueError("direction_empty_explanation")
        if (self.direction == "neutral") != (self.neutral_reason is not None):
            raise ValueError("direction_neutral_reason")
        if (self.direction == "unassessable") != (self.unassessable_reason is not None):
            raise ValueError("direction_unassessable_reason")
        if (self.resolvable_gap is None) != (self.next_action is None):
            raise ValueError("direction_gap_action_binding")
        return self


class ReasonedReviewItem(ReviewItem):
    direction_assessment: DirectionAssessment

    @model_validator(mode="after")
    def direction_binding(self):
        e = self.judgment.value
        expected = (
            "unassessable"
            if e is None
            else "support"
            if e > 0
            else "refute"
            if e < 0
            else "neutral"
        )
        if self.direction_assessment.direction != expected:
            raise ValueError("direction_value_conflict")
        return self


class ReviewRound(Strict):
    schema_version: Literal["case-review-round/1.0.0"]
    protocol_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    material_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    round_id: Literal["A", "B", "C"]
    reviewer_kind: Literal["assistant_manual_simulation", "llm"]
    model_identifier: str = Field(min_length=1, max_length=256)
    session_identifier: str = Field(min_length=1, max_length=256)
    prompt_checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    generation_settings: dict[str, str] = Field(max_length=16)
    prior_answers_visible: Literal[False]
    items: list[ReviewItem] = Field(min_length=1, max_length=512)
    purpose: Literal["candidate_review_not_production_authorization"]

    @model_validator(mode="after")
    def unique(self):
        ids = [r.assessment_id for r in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("review_duplicate_item")
        return self


def validate_round(value, *, expected_ids, material_checksum, protocol_checksum, citation_ids):
    """Coordinator checks against their frozen manifest, never caller-chosen scope.

    This verifies structural bindings, not medical correctness or independence.
    Never pass its output directly to production register/accept operations.
    """
    raw = value.model_dump(mode="json") if isinstance(value, ReviewRound) else value
    result = ReviewRound.model_validate(raw)
    if (
        result.material_checksum != material_checksum
        or result.protocol_checksum != protocol_checksum
        or {i.assessment_id for i in result.items} != set(expected_ids)
    ):
        raise ValueError("review_frozen_scope_mismatch")
    if any(
        not set(i.observation_citations + i.reference_citations) <= set(citation_ids)
        for i in result.items
    ):
        raise ValueError("review_unknown_citation")
    return result
