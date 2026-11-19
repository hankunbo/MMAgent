"""V3 all-node roles and task-scoped acquisition adequacy, not medical probabilities."""

from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict
from packages.evidence_scoring.continuous_models import (
    ContinuousRemoval,
    ContinuousScores,
    ContinuousSummary,
)
from packages.evidence_scoring.models import Coverage, Rating, Unit


class AcquisitionSlot(Strict):
    assessment_ids: list[str] = Field(min_length=1, max_length=8)
    acquired: bool
    reason: str


class EvidenceGroup(Strict):
    id: str
    title: str
    nodes: list[str]
    importance: dict[str, float]
    slots: list[AcquisitionSlot] = Field(min_length=1, max_length=32)
    coverage: Coverage
    definition_checksum: str
    source_assessment_checksums: dict[str, str]
    reason: str
    next_action: str

    @model_validator(mode="after")
    def counts(self):
        if self.coverage.numerator != sum(
            s.acquired for s in self.slots
        ) or self.coverage.denominator != len(self.slots):
            raise ValueError("participation_slot_count_mismatch")
        if set(self.importance) != {"comprehensive_research", "slo_image_research"}:
            raise ValueError("participation_scope_mismatch")
        import math

        if any(not math.isfinite(v) or not 0 <= v <= 10 for v in self.importance.values()):
            raise ValueError("participation_importance_invalid")
        return self


class Adequacy(Strict):
    scope: Literal["comprehensive_research", "slo_image_research"]
    group_ids: list[str]
    weighted_acquired: float = Field(ge=0, le=5120, allow_inf_nan=False)
    total_importance: float = Field(ge=0, le=5120, allow_inf_nan=False)
    value: Unit | None
    missing: Unit | None
    adjusted: Unit | None
    adjusted_lower: Unit | None
    adjusted_upper: Unit | None
    status: Literal["available", "not_applicable"]

    @model_validator(mode="after")
    def ratio(self):
        d = self.total_importance
        if self.weighted_acquired > d or len(self.group_ids) != len(set(self.group_ids)):
            raise ValueError("participation_adequacy_count")
        q = self.weighted_acquired / d if d else None
        if self.value != q or self.missing != (1 - q if q is not None else None):
            raise ValueError("participation_adequacy_ratio")
        if self.status != ("available" if d else "not_applicable"):
            raise ValueError("participation_adequacy_status")
        if self.adjusted is None:
            if self.adjusted_lower is not None or self.adjusted_upper is not None:
                raise ValueError("participation_adjusted_missing")
        elif (
            q is None
            or self.adjusted_lower is None
            or self.adjusted_upper is None
            or not self.adjusted_lower <= self.adjusted <= self.adjusted_upper
        ):
            raise ValueError("participation_adjusted_bounds")
        return self


class MissingImpact(Strict):
    group_id: str
    scope: str
    missing_contribution: Unit
    coverage_after_completion: Unit
    adjusted_after_completion: Unit | None
    adjusted_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    scenario_kind: Literal["coverage_only_holds_direction_fixed"]
    disease_direction_impact: Literal["not_quantified"]
    reason: str


class ParticipationSummary(ContinuousSummary):
    adequacy: dict[str, Adequacy]
    missing_impacts: list[MissingImpact]

    @model_validator(mode="after")
    def adjustment(self):
        for scope, a in self.adequacy.items():
            if scope != a.scope:
                raise ValueError("participation_profile_key_mismatch")
            for field, original in (
                ("adjusted", "value"),
                ("adjusted_lower", "lower"),
                ("adjusted_upper", "upper"),
            ):
                x = getattr(self.confidence, original)
                expected = (
                    0.5 + a.value * (x - 0.5) if a.value is not None and x is not None else None
                )
                if getattr(a, field) != expected:
                    raise ValueError("participation_adjustment_mismatch")
        return self


class Evaluation(Strict):
    assessment_id: str
    target: Literal["disease_direction", "evidence_quality", "evidence_sufficiency"]
    observed: str = Field(max_length=2500)
    unresolved: str = Field(max_length=2500)
    next_action: str = Field(max_length=2500)
    group_ids: list[str]
    quality_review: Rating | None
    effective_support: Unit
    effective_refutation: Unit
    marginal_direction_delta: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    assessment_checksum: str
    provenance: Literal["source_bound_existing_record_review", "assistant_direct_image_inspection"]


class NodeParticipation(Strict):
    role: Literal[
        "organization", "disease_evidence", "evidence_quality", "evidence_sufficiency", "mixed"
    ]
    own_assessment_ids: list[str]
    group_ids: list[str]
    serves: list[str]
    required_resources: list[str]
    technical_coverage: Coverage
    quality_reviews: list[Rating]
    reason: str


class VisualReview(Strict):
    review_id: str
    observer: Literal["assistant_direct_image_inspection_not_deployed_vlm"]
    input_checksum: str
    overlay_checksum: str
    scope: str
    observed: str
    unresolved: str
    ratings: list[Unit] = Field(min_length=3, max_length=3)
    effect: float = Field(ge=-1, le=1, allow_inf_nan=False)
    lower: float = Field(ge=-1, le=1, allow_inf_nan=False)
    upper: float = Field(ge=-1, le=1, allow_inf_nan=False)
    reason: str

    @model_validator(mode="after")
    def bounds(self):
        if not self.lower <= self.effect <= self.upper:
            raise ValueError("participation_visual_bounds")
        return self


class ParticipationRemoval(ContinuousRemoval):
    after: ParticipationSummary
    adequacy_delta: dict[str, float | None]
    adjusted_delta: dict[str, float | None]


class ParticipationScores(ContinuousScores):
    schema_version: Literal["case-research-scoring/3.0.0"]
    policy_version: Literal["argument-participation/3.0.0"]
    direction_policy_checksum: str
    default_scope: Literal["comprehensive_research"]
    visual_review: VisualReview | None
    groups: list[EvidenceGroup] = Field(max_length=64)
    evaluations: list[Evaluation] = Field(max_length=512)
    node_participation: dict[str, NodeParticipation]
    nodes: dict[str, ParticipationSummary]
    removals: list[ParticipationRemoval] = Field(max_length=512)

    @model_validator(mode="after")
    def inventory(self):
        ids = {r.id for r in self.reviews}
        evaluated = [r.assessment_id for r in self.evaluations]
        if (
            set(evaluated) != ids
            or len(evaluated) != len(ids)
            or set(self.node_participation) != set(self.nodes)
        ):
            raise ValueError("participation_inventory_mismatch")
        gids = [g.id for g in self.groups]
        if len(gids) != len(set(gids)):
            raise ValueError("participation_duplicate_group")
        for g in self.groups:
            if any(not set(slot.assessment_ids) <= ids for slot in g.slots):
                raise ValueError("participation_unknown_slot")
        for n in self.node_participation.values():
            if not set(n.own_assessment_ids) <= ids or not set(n.group_ids) <= set(gids):
                raise ValueError("participation_unknown_reference")
        return self
