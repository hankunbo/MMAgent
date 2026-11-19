"""Generic research successor contracts. Legacy records/current remain exact v1/v5."""

from typing import Literal

from pydantic import Field, model_validator

from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.grounding.models import Checksum, Token, Value, digest
from packages.llm_provider.models import (
    Material,
    PartialVisionProposal,
    ReviewProposal,
    Scope,
    SourceProposal,
    TaskDefinition,
    VisionProposal,
)
from packages.planner_control.models import Reference


class ItemPolicy(Value):
    id: Token
    node: Token
    kind: Literal["criterion", "visual", "association", "model", "measurement"]
    role: Literal["diagnostic", "risk", "measurement", "model", "visual", "association"]
    observation_code: Token
    method: Token
    unit: Token
    laterality: Literal["LEFT", "RIGHT", "BILATERAL", "UNKNOWN"]
    source_codes: tuple[Token, ...] = Field(min_length=1, max_length=16)
    applicable: bool
    correlation_group: Token
    dependence_block: Token
    dependency_paths: tuple[tuple[Token, ...], ...] = Field(min_length=1, max_length=8)
    fixed_model_row: ContinuousReview | None = None

    @model_validator(mode="after")
    def fixed(self):
        if (self.role == "model") != (self.fixed_model_row is not None):
            raise ValueError("fixed_model_required")
        if self.fixed_model_row and self.fixed_model_row.id != self.id:
            raise ValueError("fixed_model_scope")
        if any(not p or len(p) != len(set(p)) for p in self.dependency_paths):
            raise ValueError("dependency_path")
        return self


class CoverageGroup(Value):
    id: Token
    slots: tuple[tuple[Token, ...], ...] = Field(min_length=1, max_length=64)
    importance: dict[Literal["slo_image_research", "comprehensive_research"], float]

    @model_validator(mode="after")
    def bounded(self):
        if set(self.importance) != {"slo_image_research", "comprehensive_research"} or any(
            not 0 <= v <= 100 for v in self.importance.values()
        ):
            raise ValueError("coverage_importance")
        if any(not s or len(s) != len(set(s)) for s in self.slots):
            raise ValueError("coverage_slots")
        return self


class TaskPolicy(Value):
    schema_version: Literal["research-task-policy/1.0.0"] = "research-task-policy/1.0.0"
    version: Token
    scale: Reference
    aggregation_checksum: Checksum
    items: tuple[ItemPolicy, ...] = Field(min_length=1, max_length=128)
    groups: tuple[CoverageGroup, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def inventory(self):
        ids = {i.id for i in self.items}
        if len(ids) != len(self.items) or len({g.id for g in self.groups}) != len(self.groups):
            raise ValueError("policy_duplicates")
        if any(not set(s) <= ids for g in self.groups for s in g.slots):
            raise ValueError("policy_unknown_slot")
        return self


class ItemPolicyV2(ItemPolicy):
    definition: TaskDefinition
    fixed_observation_ref: Reference | None = None

    @model_validator(mode="after")
    def fixed_binding(self):
        if (self.fixed_model_row is not None) != (self.fixed_observation_ref is not None):
            raise ValueError("fixed_model_observation_binding_required")
        return self


class TaskPolicyV2(TaskPolicy):
    schema_version: Literal["research-task-policy/2.0.0"] = "research-task-policy/2.0.0"
    items: tuple[ItemPolicyV2, ...] = Field(min_length=1, max_length=128)


class EvidenceRequirement(Value):
    """Frozen, source-backed demand; it is NOT a patient observation or a score."""

    id: Token
    definition: TaskDefinition
    input_modalities: tuple[Token, ...] = Field(min_length=1, max_length=16)
    basis_kind: Literal["quantitative", "qualitative", "model_validation"]
    missing_reason: str | None = Field(default=None, min_length=1, max_length=1024)


class ObservedItemPolicy(ItemPolicyV2):
    """New evidence has no compulsory classifier row or historical fixed ratings."""

    model_direction: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    model_observation_ref: Reference | None = None
    model_mapping_version: Literal["normalized-class-margin/1.0.0"] | None = None

    @model_validator(mode="after")
    def fixed(self):
        if self.fixed_model_row is not None or self.fixed_observation_ref is not None:
            raise ValueError("dynamic_policy_forbids_old_fixed_ratings")
        bound = (
            self.model_direction is not None,
            self.model_observation_ref is not None,
            self.model_mapping_version is not None,
        )
        if any(bound) != all(bound) or (self.role == "model") != all(bound):
            raise ValueError("dynamic_model_output_binding_required")
        if any(not p or len(p) != len(set(p)) for p in self.dependency_paths):
            raise ValueError("dependency_path")
        return self


class TaskPolicyV3(TaskPolicyV2):
    schema_version: Literal["research-task-policy/3.0.0"] = "research-task-policy/3.0.0"
    items: tuple[ObservedItemPolicy, ...] = Field(max_length=128)
    groups: tuple[CoverageGroup, ...] = Field(max_length=64)
    requirements: tuple[EvidenceRequirement, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def inventory(self):
        ids = {i.id for i in self.items}
        required = {r.id: r for r in self.requirements}
        if (
            len(ids) != len(self.items)
            or len(required) != len(self.requirements)
            or len({g.id for g in self.groups}) != len(self.groups)
            or not ids <= required.keys()
        ):
            raise ValueError("dynamic_policy_inventory")
        covered = {k for g in self.groups for slot in g.slots for k in slot}
        if covered != required.keys():
            raise ValueError("dynamic_requirement_denominator")
        if any(i.definition != required[i.id].definition for i in self.items):
            raise ValueError("dynamic_requirement_definition")
        return self


class QualityFactor(Value):
    value: float = Field(ge=0, le=1, allow_inf_nan=False)
    method: Token
    reason: str = Field(min_length=1, max_length=1024)
    source: Reference


class QualityItemPolicy(ObservedItemPolicy):
    quality: QualityFactor


class TaskPolicyV4(TaskPolicyV3):
    schema_version: Literal["research-task-policy/4.0.0"] = "research-task-policy/4.0.0"
    items: tuple[QualityItemPolicy, ...] = Field(max_length=128)


class TaskCoverageGroup(CoverageGroup):
    """Task-relative coverage; never labels an OCT/VF input as SLO."""

    importance: dict[Literal["observed_input_research", "task_requirement_research"], float]

    @model_validator(mode="after")
    def bounded(self):
        if set(self.importance) != {"observed_input_research", "task_requirement_research"} or any(
            not 0 <= v <= 100 for v in self.importance.values()
        ):
            raise ValueError("task_coverage_importance")
        if any(not s or len(s) != len(set(s)) for s in self.slots):
            raise ValueError("coverage_slots")
        return self


class TaskPolicyV5(TaskPolicyV4):
    """Explicit research hypothesis, independent of dataset and classifier slots."""

    schema_version: Literal["research-task-policy/5.0.0"] = "research-task-policy/5.0.0"
    target_code: str = Field(pattern=r"^GOAL:[A-Za-z0-9_:.\-]+$")
    target_summary: str = Field(min_length=1, max_length=240)
    groups: tuple[TaskCoverageGroup, ...] = Field(max_length=64)


class ResearchHypothesis(Value):
    code: str = Field(pattern=r"^GOAL:[A-Za-z0-9_:.\-]+$")
    label: Token
    title: str = Field(min_length=1, max_length=120)
    meaning: str = Field(min_length=1, max_length=1000)
    evaluation_label: Token


class TaskPolicyV6(TaskPolicyV5):
    """Separate hypotheses over shared facts; never pool mutually exclusive targets."""

    schema_version: Literal["research-task-policy/6.0.0"] = "research-task-policy/6.0.0"
    hypotheses: tuple[ResearchHypothesis, ...] = Field(min_length=2, max_length=8)
    hypothesis_bindings: dict[Token, Token]
    evidence_units: dict[Token, Token]

    @model_validator(mode="after")
    def hypothesis_inventory(self):
        ids = {r.id for r in self.requirements}
        codes = {h.code for h in self.hypotheses}
        if (
            len(codes) != len(self.hypotheses)
            or len({h.label for h in self.hypotheses}) != len(self.hypotheses)
            or self.target_code in codes
            or set(self.hypothesis_bindings) != ids
            or set(self.evidence_units) != ids
            or set(self.hypothesis_bindings.values()) != codes
            or any(i.role == "model" for i in self.items)
        ):
            raise ValueError("hypothesis_inventory_or_unbound_classifier")
        pairs = [(self.evidence_units[k], self.hypothesis_bindings[k]) for k in ids]
        if len(set(pairs)) != len(pairs):
            raise ValueError("hypothesis_duplicate_evidence_vote")
        for unit in set(self.evidence_units.values()):
            if {h for u, h in pairs if u == unit} != codes:
                raise ValueError("hypothesis_missing_requirement")
        return self


class TaskPolicyV7(TaskPolicyV5):
    schema_version: Literal["research-task-policy/7.0.0"] = "research-task-policy/7.0.0"
    direction_policy: Literal["direction-assessment/1.0.0"] = "direction-assessment/1.0.0"


class DomainState(Value):
    schema_version: Literal["research-domain-state/2.0.0"] = "research-domain-state/2.0.0"
    material: Material
    policy: (
        TaskPolicyV7
        | TaskPolicyV6
        | TaskPolicyV5
        | TaskPolicyV4
        | TaskPolicyV3
        | TaskPolicyV2
        | TaskPolicy
    )
    laterality: Literal["LEFT", "RIGHT", "BILATERAL", "UNKNOWN"]
    resources: tuple[Reference, ...] = Field(min_length=1, max_length=128)
    prompt_checksum: Checksum
    config_checksum: Checksum
    nodes: tuple[Token, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def locked(self):
        if isinstance(self.policy, TaskPolicyV5) and (
            self.policy.target_code != self.material.goal_code
            or self.policy.target_summary != self.material.goal_summary
        ):
            raise ValueError("task_score_target_binding")
        if digest(self.policy) != self.material.scope.policy_checksum:
            raise ValueError("domain_policy_lock")
        if self.material.scope.graph.kind != "research_artifact":
            raise ValueError("research_identity_required")
        if len(set(self.nodes)) != len(self.nodes) or len({r.code for r in self.resources}) != len(
            self.resources
        ):
            raise ValueError("domain_duplicates")
        if any(i.node not in self.nodes for i in self.policy.items):
            raise ValueError("domain_node")
        if isinstance(self.policy, TaskPolicyV3):
            observations = {o.reference.code: o for o in self.material.observations}
            sources = {s.reference for s in self.material.sources}
            for requirement in self.policy.requirements:
                if not set(requirement.definition.source_refs) <= sources:
                    raise ValueError("dynamic_requirement_source")
            for item in self.policy.items:
                observation = observations.get(item.observation_code)
                if (
                    observation is None
                    or observation.status != "obtained"
                    or not observation.value
                    or not observation.value.strip()
                ):
                    raise ValueError("dynamic_evidence_requires_observation")
                if item.role == "model" and item.model_observation_ref != observation.reference:
                    raise ValueError("dynamic_model_observation_binding")
                if (
                    isinstance(item, QualityItemPolicy)
                    and item.quality.source not in self.resources
                ):
                    raise ValueError("quality_source_not_bound")
        return self


class Submission(Value):
    schema_version: Literal["research-submission/1.0.0"] = "research-submission/1.0.0"
    scope: Scope
    actor: Literal["PLANNER"]
    correlation: Literal["SOURCE:AGENT_RESEARCH"]
    request_id: Token
    prompt_checksum: Checksum
    config_checksum: Checksum
    observation_checksums: tuple[Reference, ...]
    source_checksums: tuple[Reference, ...]
    candidate: ReviewProposal | SourceProposal | PartialVisionProposal | VisionProposal
    status: Literal["candidate_not_accepted"] = "candidate_not_accepted"


class AcceptanceReceipt(Value):
    schema_version: Literal["research-acceptance/1.0.0"] = "research-acceptance/1.0.0"
    submission_checksum: Checksum
    status: Literal["accepted", "rejected"]
    reason: Token
    parent_graph: Reference
    graph: Reference | None
    records_checksum: Checksum | None
    scores_checksum: Checksum | None
    clinical_acceptance: Literal[False] = False
    new_medical_review: Literal[False] = False
    execution: Literal["offline_fixture_or_candidate_only"] = "offline_fixture_or_candidate_only"


class AcceptedInterpretation(Value):
    from packages.evidence_scoring.review_protocol import ReviewItem as _ReviewItem

    id: Token
    version: Literal["2.0.0"] = "2.0.0"
    observation_ref: Reference
    source_refs: tuple[Reference, ...]
    policy_checksum: Checksum
    review: _ReviewItem
    correlation_group: Token
    dependence_block: Token
    dependency_paths: tuple[tuple[Token, ...], ...]
    request_id: Token


class ReasonedInterpretation(AcceptedInterpretation):
    from packages.evidence_scoring.review_protocol import ReasonedReviewItem as _Item

    version: Literal["2.1.0"] = "2.1.0"
    review: _Item


class AcceptedRecords(Value):
    from packages.llm_provider.models import ObservationInput as _Observation

    schema_version: Literal["case-observation-interpretation/2.0.0"] = (
        "case-observation-interpretation/2.0.0"
    )
    scope: Scope
    parent_records: Literal["preserved_by_parent_graph"]
    candidate: ReviewProposal | SourceProposal | PartialVisionProposal | VisionProposal
    observations: tuple[_Observation, ...]
    interpretations: tuple[AcceptedInterpretation, ...]
    invalidated_assessment_ids: tuple[Token, ...]
    request_id: Token
    status: Literal["accepted_research_fixture"]
    clinical_acceptance: Literal[False] = False


class AcceptedRecordsV21(AcceptedRecords):
    from packages.llm_provider.models import ObservationInput as _Observation
    from packages.llm_provider.models import ReviewTemplate as _Template

    schema_version: Literal["case-observation-interpretation/2.1.0"] = (
        "case-observation-interpretation/2.1.0"
    )
    review_tasks: _Template | None
    observation_facts: tuple[_Observation, ...] = Field(max_length=128)


class AcceptanceReceiptV2(AcceptanceReceipt):
    schema_version: Literal["research-acceptance/2.0.0"] = "research-acceptance/2.0.0"
    execution: Literal["bounded_live_research"] = "bounded_live_research"
    permission_checksum: Checksum
    new_medical_review: bool
    reviewer_kind: Literal["llm"] = "llm"


class AcceptedRecordsV22(AcceptedRecordsV21):
    schema_version: Literal["case-observation-interpretation/2.2.0"] = (
        "case-observation-interpretation/2.2.0"
    )
    status: Literal["accepted_live_research"]
    permission_checksum: Checksum
    execution: Literal["bounded_live_research"] = "bounded_live_research"


class AcceptedRecordsV23(AcceptedRecordsV21):
    from .review_batches import ReviewBatchPlan as _Plan
    from .review_batches import ReviewBatchResult as _Result

    schema_version: Literal["case-observation-interpretation/2.3.0"] = (
        "case-observation-interpretation/2.3.0"
    )
    candidate: ReviewProposal
    request_id: None = None
    request_ids: tuple[Token, ...] = Field(min_length=1, max_length=20)
    batch_plan: _Plan
    batch_results: tuple[_Result, ...] = Field(min_length=1, max_length=20)
    status: Literal["accepted_research_fixture", "accepted_live_research"]
    execution: Literal["offline_fixture_or_candidate_only", "bounded_live_research"]
    permission_checksum: Checksum | None

    @model_validator(mode="after")
    def batch_identity(self):
        live = self.execution == "bounded_live_research"
        if (
            live != (self.status == "accepted_live_research")
            or live != (self.permission_checksum is not None)
            or self.scope != self.batch_plan.scope
            or self.request_ids != tuple(r.request_id for r in self.batch_results)
            or len(set(self.request_ids)) != len(self.request_ids)
            or any(i.request_id not in self.request_ids for i in self.interpretations)
        ):
            raise ValueError("accepted_batch_identity")
        return self


class AcceptedRecordsV24(AcceptedRecordsV23):
    schema_version: Literal["case-observation-interpretation/2.4.0"] = (
        "case-observation-interpretation/2.4.0"
    )
    completion: Literal["partial"] = "partial"
    pending_batch_ids: tuple[Token, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def partial_identity(self):
        completed = {r.batch_id for r in self.batch_results}
        if (
            not completed <= {b.id for b in self.batch_plan.batches}
            or len(completed) != len(self.batch_results)
            or self.pending_batch_ids
            != tuple(b.id for b in self.batch_plan.batches if b.id not in completed)
        ):
            raise ValueError("partial_record_inventory")
        return self


class AcceptedRecordsV25(AcceptedRecordsV23):
    from packages.llm_provider.models import ReasonedReviewProposal as _Proposal

    from .review_batches import ReasonedBatchResult as _Result

    schema_version: Literal["case-observation-interpretation/2.5.0"] = (
        "case-observation-interpretation/2.5.0"
    )
    candidate: _Proposal
    interpretations: tuple[ReasonedInterpretation, ...]
    batch_results: tuple[_Result, ...] = Field(min_length=1, max_length=20)


class AcceptedRecordsV26(AcceptedRecordsV24):
    from packages.llm_provider.models import ReasonedReviewProposal as _Proposal

    from .review_batches import ReasonedBatchResult as _Result

    schema_version: Literal["case-observation-interpretation/2.6.0"] = (
        "case-observation-interpretation/2.6.0"
    )
    candidate: _Proposal
    interpretations: tuple[ReasonedInterpretation, ...]
    batch_results: tuple[_Result, ...] = Field(min_length=1, max_length=20)


class AcceptedRecordsV27(AcceptedRecordsV25):
    from .review_continuation import HistoricalReview as _Origin

    schema_version: Literal["case-observation-interpretation/2.7.0"] = (
        "case-observation-interpretation/2.7.0"
    )
    historical_review: _Origin

    @model_validator(mode="after")
    def batch_identity(self):
        all_results = (*self.historical_review.submission.batches, *self.batch_results)
        if self.schema_version in {
            "case-observation-interpretation/2.9.0",
            "case-observation-interpretation/2.11.0",
            "case-observation-interpretation/2.12.0",
            "case-observation-interpretation/2.16.0",
            "case-observation-interpretation/2.17.0",
            "case-observation-interpretation/2.18.0",
        }:
            from .accepted_review_reuse import provenance_batches

            all_results = (
                *provenance_batches(self.historical_review.submission),
                *self.batch_results,
            )
        live = self.execution == "bounded_live_research"
        if (
            live != (self.status == "accepted_live_research")
            or live != (self.permission_checksum is not None)
            or self.scope != self.batch_plan.scope
            or self.request_ids != tuple(r.request_id for r in all_results)
            or len(set(self.request_ids)) != len(self.request_ids)
            or {i.review.assessment_id for i in self.interpretations}
            != set(self.batch_plan.assessment_ids)
            or any(i.request_id not in self.request_ids for i in self.interpretations)
        ):
            raise ValueError("accepted_continued_identity")
        return self


class AcceptedRecordsV28(AcceptedRecordsV27):
    from .accepted_review_reuse import AcceptedReviewOrigin as _AcceptedOrigin

    schema_version: Literal["case-observation-interpretation/2.8.0"] = (
        "case-observation-interpretation/2.8.0"
    )
    historical_review: _AcceptedOrigin


class AcceptedRecordsV29(AcceptedRecordsV28):
    from .accepted_review_reuse import AcceptedReviewOriginV11 as _OriginV11

    schema_version: Literal["case-observation-interpretation/2.9.0"] = (
        "case-observation-interpretation/2.9.0"
    )
    historical_review: _OriginV11


class AcceptedRecordsV210(AcceptedRecordsV27):
    from .review_format_continuation import FormatHistoricalReview as _Origin

    schema_version: Literal["case-observation-interpretation/2.10.0"] = (
        "case-observation-interpretation/2.10.0"
    )
    historical_review: _Origin


class AcceptedRecordsV211(AcceptedRecordsV29):
    from .accepted_review_reuse import AcceptedReviewOriginV12 as _Origin

    schema_version: Literal["case-observation-interpretation/2.11.0"] = (
        "case-observation-interpretation/2.11.0"
    )
    historical_review: _Origin


class AcceptedRecordsV212(AcceptedRecordsV211):
    from .accepted_review_reuse import (
        AcceptedReviewOrigin as _V1,
    )
    from .accepted_review_reuse import (
        AcceptedReviewOriginV11 as _V11,
    )
    from .accepted_review_reuse import (
        AcceptedReviewOriginV12 as _V12,
    )
    from .accepted_review_reuse import (
        AcceptedReviewOriginV13 as _V13,
    )
    from .review_batches import ReusePartitionPlan as _Plan

    schema_version: Literal["case-observation-interpretation/2.12.0"] = (
        "case-observation-interpretation/2.12.0"
    )
    historical_review: _V1 | _V11 | _V12 | _V13 = Field(discriminator="schema_version")
    batch_plan: _Plan


class AcceptedRecordsV216(AcceptedRecordsV212):
    from .accepted_review_reuse import AcceptedReviewOriginV14 as _V14
    from .review_batches import ReusePartitionPlan as _PartitionPlan
    from .review_batches import ReviewBatchPlan as _OriginalPlan

    schema_version: Literal["case-observation-interpretation/2.16.0"] = (
        "case-observation-interpretation/2.16.0"
    )
    historical_review: _V14
    batch_plan: _OriginalPlan | _PartitionPlan = Field(discriminator="schema_version")


class AcceptedRecordsV217(AcceptedRecordsV216):
    from .accepted_review_reuse import AcceptedReviewOriginV13 as _V13
    from .accepted_review_reuse import AcceptedReviewOriginV15 as _V15

    schema_version: Literal["case-observation-interpretation/2.17.0"] = (
        "case-observation-interpretation/2.17.0"
    )
    historical_review: _V13 | _V15 = Field(discriminator="schema_version")


class AcceptedRecordsV218(AcceptedRecordsV217):
    from .accepted_review_reuse import AcceptedReviewOrigin as _V1
    from .accepted_review_reuse import AcceptedReviewOriginV11 as _V11
    from .accepted_review_reuse import AcceptedReviewOriginV12 as _V12
    from .accepted_review_reuse import AcceptedReviewOriginV13 as _V13
    from .accepted_review_reuse import AcceptedReviewOriginV14 as _V14
    from .accepted_review_reuse import AcceptedReviewOriginV15 as _V15
    from .accepted_review_reuse import AcceptedReviewOriginV16 as _V16
    from .review_batches import ReasonedBatchResult as _Result

    schema_version: Literal["case-observation-interpretation/2.18.0"] = (
        "case-observation-interpretation/2.18.0"
    )
    historical_review: _V1 | _V11 | _V12 | _V13 | _V14 | _V15 | _V16 = Field(
        discriminator="schema_version"
    )
    batch_results: tuple[_Result, ...] = Field(max_length=20)
    composition: Literal["all_current_items_reused", "accepted_origin_with_new_batches"]

    @model_validator(mode="after")
    def composition_identity(self):
        if (not self.batch_results) != (self.composition == "all_current_items_reused"):
            raise ValueError("accepted_composition_batch_identity")
        return self


def read_receipt(value):
    from packages.grounding.models import validate

    from .accepted_aggregation import AggregationReceipt

    types = {
        "research-acceptance/1.0.0": AcceptanceReceipt,
        "research-acceptance/2.0.0": AcceptanceReceiptV2,
        "research-acceptance/3.0.0": AggregationReceipt,
    }
    cls = types.get(value.get("schema_version"))
    if cls is None:
        raise ValueError("research_receipt_version")
    return validate(cls, value)


def read_records(value):
    from packages.grounding.models import validate

    from .accepted_aggregation import (
        AggregatedRecords,
        AliasAggregatedRecords,
        LegacyAggregatedRecords,
    )
    from .input_scope import (
        LiveScopeRecords,
        LiveScopeRecordsV2,
        LiveScopeRecordsV3,
        ScopeRecords,
        ScopeRecordsV2,
        ScopeRecordsV3,
    )
    from .recorded_explanation import ExplanationAcceptedRecords
    from .scope_finalization import RecordedScopeRecords, RecordedScopeTextRecords

    types = {
        "case-input-scope-observation/1.0.0": ScopeRecords,
        "case-input-scope-observation/1.1.0": LiveScopeRecords,
        "case-input-scope-observation/2.0.0": ScopeRecordsV2,
        "case-input-scope-observation/2.1.0": LiveScopeRecordsV2,
        "case-input-scope-observation/3.0.0": ScopeRecordsV3,
        "case-input-scope-observation/3.1.0": LiveScopeRecordsV3,
        "case-input-scope-observation/2.2.0": RecordedScopeRecords,
        "case-input-scope-observation/2.3.0": RecordedScopeTextRecords,
        "case-observation-interpretation/2.0.0": AcceptedRecords,
        "case-observation-interpretation/2.1.0": AcceptedRecordsV21,
        "case-observation-interpretation/2.2.0": AcceptedRecordsV22,
        "case-observation-interpretation/2.3.0": AcceptedRecordsV23,
        "case-observation-interpretation/2.4.0": AcceptedRecordsV24,
        "case-observation-interpretation/2.5.0": AcceptedRecordsV25,
        "case-observation-interpretation/2.6.0": AcceptedRecordsV26,
        "case-observation-interpretation/2.7.0": AcceptedRecordsV27,
        "case-observation-interpretation/2.8.0": AcceptedRecordsV28,
        "case-observation-interpretation/2.9.0": AcceptedRecordsV29,
        "case-observation-interpretation/2.10.0": AcceptedRecordsV210,
        "case-observation-interpretation/2.11.0": AcceptedRecordsV211,
        "case-observation-interpretation/2.12.0": AcceptedRecordsV212,
        "case-observation-interpretation/2.13.0": ExplanationAcceptedRecords,
        "case-observation-interpretation/2.14.0": AggregatedRecords,
        "case-observation-interpretation/2.15.0": LegacyAggregatedRecords,
        "case-observation-interpretation/2.16.0": AcceptedRecordsV216,
        "case-observation-interpretation/2.17.0": AcceptedRecordsV217,
        "case-observation-interpretation/2.18.0": AcceptedRecordsV218,
        "case-observation-interpretation/2.19.0": AliasAggregatedRecords,
    }
    cls = types.get(value.get("schema_version"))
    if cls is None:
        raise ValueError("research_records_version")
    return validate(cls, value)


class ResearchGraphValue(Value):
    schema_version: Literal["case-evidence-graph/2.0.0"] = "case-evidence-graph/2.0.0"
    scope: Scope
    parent_graph: Reference
    records_checksum: Checksum
    scores_checksum: Checksum | None
    nodes: tuple[Token, ...]
    invalidated_assessment_ids: tuple[Token, ...]
    accepted_validations: tuple[()] = ()
    clinical_patient_graph: Literal[False] = False
