"""V5: acquisition, interpretation, conclusion and bounded robustness analysis."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict, digest
from packages.evidence_details.records_models import SHA, Code
from packages.evidence_scoring.continuous_models import ContinuousReview, ContinuousSummary
from packages.evidence_scoring.models import ResourceBinding, Unit


class Slot(Strict):
    assessment_ids: list[Code] = Field(min_length=1, max_length=16)
    acquired: bool
    interpreted: bool

    @model_validator(mode="after")
    def compatible(self):
        if self.interpreted and not self.acquired:
            raise ValueError("current_interpretation_without_acquisition")
        return self


class Group(Strict):
    id: Code
    title: str
    importance: dict[
        Literal["slo_image_research", "comprehensive_research"],
        Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)],
    ]
    slots: list[Slot] = Field(min_length=1, max_length=32)


class Completeness(Strict):
    denominator: float = Field(ge=0, allow_inf_nan=False)
    acquisition: Unit | None
    interpretation: Unit | None


class State(Strict):
    summary: ContinuousSummary
    coverage: dict[Literal["slo_image_research", "comprehensive_research"], Completeness]


class Removal(Strict):
    id: Code
    removed_assessments: list[Code]
    removed_resources: list[Code]
    affected_assessments: list[Code]
    after: State
    confidence_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)


class Conclusion(Strict):
    all_evidence: ContinuousSummary
    without_age: ContinuousSummary
    without_background: ContinuousSummary
    image_evidence: ContinuousSummary
    background_only: ContinuousSummary
    age_changes_direction: bool
    background_changes_direction: bool
    age_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    conclusion: str
    disease_probability: None = None
    decision_threshold: None = None


class KeySet(Strict):
    assessment_ids: list[Code]
    value: Unit
    interpretation_retention: Unit


class KeyAnalysis(Strict):
    status: Literal["exhaustive", "bounded_not_run"]
    candidates: int = Field(ge=0)
    evaluated_subsets: int = Field(ge=0)
    score_tolerance: Unit
    minimum_interpretation_retention: Unit
    required_roles: list[str]
    smallest_fidelity_sets: list[KeySet] = Field(max_length=4096)
    smallest_direction_flip_removals: list[list[Code]] = Field(max_length=4096)
    meaning: Literal["score_fidelity_not_sufficient_for_diagnosis_or_biological_causality"]


class MissingScenario(Strict):
    group_id: Code
    unresolved_assessments: list[Code]
    kind: Literal["coverage_only_no_assumed_patient_result"]
    acquisition_if_completed: dict[str, Unit | None]
    interpretation_if_reviewed: dict[str, Unit | None]
    disease_effect: None = None
    reason: str


class MaskMetrics(Strict):
    vertical_cup_disc_ratio: Unit
    minimum_radial_ratio: Unit
    disc_pixels: int = Field(gt=0)
    cup_pixels: int = Field(gt=0)


class AssetRef(Strict):
    code: Code
    checksum: SHA


class Replacement(Strict):
    schema_version: Literal["existing-mask-replacement/1.0.0"]
    substitution: str
    status: Literal["inputs_unavailable", "qc_rejected", "measured_requires_new_interpretation"]
    source_refs: dict[str, AssetRef]
    original: MaskMetrics | None
    alternative: MaskMetrics | None
    deltas: dict[str, float | int] | None
    error: str | None
    disease_score_after_replacement: None
    next_action: str
    invalidated_assessments: list[Code]
    remaining_evidence_summary: ContinuousSummary


class CurrentScores(Strict):
    schema_version: Literal["case-research-scoring/5.0.0"]
    policy_version: Literal["record-connected-scoring/1.0.0"]
    case_code: Code
    input_checksum: SHA
    base_checksum: SHA
    records_checksum: SHA
    policy_checksum: SHA
    reviews: list[ContinuousReview] = Field(min_length=1, max_length=512)
    resources: dict[Code, list[ResourceBinding]]
    groups: list[Group] = Field(min_length=1, max_length=64)
    root: State
    nodes: dict[Code, ContinuousSummary]
    conclusion: Conclusion
    removals: list[Removal] = Field(max_length=1024)
    key_analysis: KeyAnalysis
    missing_scenarios: list[MissingScenario] = Field(max_length=64)
    tool_replacement: Replacement
    method_notes: list[str]
    clinical_acceptance: Literal[False]
    checksum: SHA

    @model_validator(mode="after")
    def identity(self):
        if len({r.id for r in self.reviews}) != len(self.reviews):
            raise ValueError("current_duplicate_review")
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("current_checksum")
        return self
