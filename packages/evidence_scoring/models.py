"""Strict, source-bound numeric reviews and deterministic propagation records."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict, digest

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Rating(Strict):
    value: Unit | None
    reason: str = Field(min_length=1, max_length=1000)
    missing_reason: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def presence(self):
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("score_value_missing_reason_conflict")
        return self


class Review(Strict):
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
    direction: Literal["support", "refute", "neutral", "unavailable"]
    observation_available: bool
    criterion_checked: bool
    reason: str
    correlation_group: str
    # OR of paths, AND within each path. Empty means unavailable, not success.
    dependency_paths: list[list[str]] = Field(max_length=8)
    weight: Unit | None
    support: Unit
    refutation: Unit

    @model_validator(mode="after")
    def contributions(self):
        values = [self.observation.value, self.relation.value, self.applicability.value]
        expected = values[0] * values[1] * values[2] if all(v is not None for v in values) else None
        if self.weight != expected:
            raise ValueError("research_score_weight_mismatch")
        if self.direction in {"support", "refute"} and (
            self.weight is None or not self.observation_available
        ):
            raise ValueError("research_score_direction_without_evidence")
        if self.support != (self.weight if self.direction == "support" else 0.0):
            raise ValueError("research_score_support_mismatch")
        if self.refutation != (self.weight if self.direction == "refute" else 0.0):
            raise ValueError("research_score_refutation_mismatch")
        return self


class Coverage(Strict):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: Unit | None
    status: Literal["available", "not_applicable"]

    @model_validator(mode="after")
    def ratio(self):
        if self.numerator > self.denominator:
            raise ValueError("research_score_coverage_count")
        expected = self.numerator / self.denominator if self.denominator else None
        if self.value != expected or self.status != (
            "available" if self.denominator else "not_applicable"
        ):
            raise ValueError("research_score_coverage_ratio")
        return self


class Channel(Strict):
    support: Unit
    refutation: Unit
    conflict: Unit
    net: float = Field(ge=-1, le=1, allow_inf_nan=False)
    active_groups: int = Field(ge=0)


class Summary(Strict):
    assessment_ids: list[str]
    diagnostic: Channel
    risk: Channel
    criterion_coverage: Coverage
    observation_coverage: Coverage
    contributing_ids: list[str]


class Removal(Strict):
    id: str
    title: str
    unavailable_resources: list[str]
    removed_assessments: list[str]
    affected_assessments: list[str]
    after: Summary
    changed_nodes: list[str]
    diagnostic_support_delta: float
    risk_support_delta: float
    criterion_coverage_delta: float | None
    explanation: str


class ResourceBinding(Strict):
    pointer: str
    checksum: str
    purpose: str


class Scores(Strict):
    schema_version: Literal["case-research-scoring/1.0.0"]
    policy_version: Literal["argument-review/1.0.0"]
    policy_checksum: str
    case_code: str
    input_checksum: str
    base_checksum: str
    trace_checksum: str
    reviewer: Literal["assistant_manual_simulation_of_llm"]
    reviewed_at: Literal["2026-09-20"]
    purpose: Literal["research_explanation_not_clinical_authorization"]
    disease_probability: None = None
    clinical_acceptance: Literal[False] = False
    reviews: list[Review] = Field(max_length=512)
    resources: dict[str, list[ResourceBinding]]
    nodes: dict[str, Summary]
    removals: list[Removal] = Field(max_length=512)
    sensitivity: dict[str, list[Unit]]
    method_notes: list[str]
    checksum: str

    @model_validator(mode="after")
    def seal(self):
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("research_score_checksum_mismatch")
        ids = [r.id for r in self.reviews]
        if len(ids) != len(set(ids)) or any(
            len(n.assessment_ids) != len(set(n.assessment_ids))
            or not set(n.assessment_ids) <= set(ids)
            for n in self.nodes.values()
        ):
            raise ValueError("research_score_duplicate_or_unknown_review")
        return self
