"""Observation acquisition and medical interpretation are separate typed records."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.evidence_details.models import Strict, digest
from packages.evidence_scoring.continuous_models import Judgment
from packages.evidence_scoring.models import Rating

SHA = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=4000)]
Code = Annotated[str, Field(min_length=1, max_length=256)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class RecordBinding(Strict):
    """JSON pointer into the exact persisted parent view, never a filesystem path."""

    pointer: str = Field(pattern=r"^/", max_length=1024)
    checksum: SHA


class Datum(Strict):
    name: Code
    value: Number | int | str | bool | list[Number | int] | None
    unit: str = Field(max_length=128)
    missing_reason: str | None = Field(max_length=1000)

    @model_validator(mode="after")
    def presence(self):
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("record_datum_presence")
        if isinstance(self.value, str) and len(self.value) > 4000:
            raise ValueError("record_text_bound")
        if isinstance(self.value, list) and not 1 <= len(self.value) <= 16:
            raise ValueError("record_vector_bound")
        return self


class Observation(Strict):
    id: Code
    assessment_id: Code
    node: Code
    title: Text
    acquisition: Literal["obtained", "not_obtained"]
    result_kind: Literal[
        "measurement",
        "proxy_measurement",
        "visual_review",
        "model_output",
        "provided_information",
        "legacy_observation",
        "unavailable",
    ]
    summary: Text
    values: list[Datum] = Field(max_length=128)
    missing_reason: str | None = Field(max_length=4000)
    input_bindings: list[RecordBinding] = Field(min_length=1, max_length=32)
    result_bindings: list[RecordBinding] = Field(min_length=1, max_length=32)
    method_bindings: list[RecordBinding] = Field(max_length=32)
    context_bindings: list[RecordBinding] = Field(max_length=32)
    origin: Literal["persisted_fact_projection_no_new_observation"]

    @model_validator(mode="after")
    def acquisition_boundary(self):
        if self.acquisition == "not_obtained":
            if self.values or not self.missing_reason or self.result_kind != "unavailable":
                raise ValueError("record_absent_observation")
        elif self.missing_reason is not None or self.result_kind == "unavailable":
            raise ValueError("record_obtained_observation")
        return self


class Interpretation(Strict):
    id: Code
    assessment_id: Code
    node: Code
    observation_id: Code
    target: Literal["glaucoma_research_hypothesis"]
    status: Literal[
        "assessed",
        "method_mismatch",
        "reference_missing",
        "context_only",
        "observation_ungradable",
        "input_missing",
        "intermediate",
        "not_reviewed",
    ]
    comparison: Text
    reference_interpretation: Text
    unresolved: Text
    next_action: Text
    reference_bindings: list[RecordBinding] = Field(max_length=64)
    provenance_bindings: list[RecordBinding] = Field(min_length=1, max_length=32)
    observation_reliability: Rating
    relation_strength: Rating
    applicability: Rating
    judgment: Judgment
    correlation_group: Code
    dependency_paths: list[list[Code]] = Field(max_length=8)
    dependency_semantics: Literal["current_binding", "legacy_not_rebased"]
    score_action: Literal["preserve_bound_legacy_review", "new_observation_not_rescored"]
    legacy_score_binding: RecordBinding

    @model_validator(mode="after")
    def direction_boundary(self):
        if (self.status == "assessed") != (self.judgment.value is not None):
            raise ValueError("record_interpretation_direction")
        if self.status == "assessed" and (
            not self.reference_bindings
            or not self.dependency_paths
            or any(
                r.value is None
                for r in (self.observation_reliability, self.relation_strength, self.applicability)
            )
        ):
            raise ValueError("record_assessed_requires_basis")
        if self.score_action == "new_observation_not_rescored" and (
            self.judgment.value is not None or self.dependency_semantics != "legacy_not_rebased"
        ):
            raise ValueError("record_new_measurement_not_legacy_score")
        return self


class Inventory(Strict):
    total: int = Field(ge=1, le=512)
    observation_obtained: int = Field(ge=0)
    observation_not_obtained: int = Field(ge=0)
    direction_assessed: int = Field(ge=0)
    measurement_without_direction: int = Field(ge=0)
    meaning: Literal["record_counts_not_independent_evidence_or_clinical_coverage"]


class EvidenceRecords(Strict):
    schema_version: Literal["case-observation-interpretation/1.0.0"]
    projection_version: Literal["observation-interpretation-normalization/1.0.0"]
    case_code: Code
    input_checksum: SHA
    base_checksum: SHA
    task_scope: Literal["slo_image_research"]
    observations: list[Observation] = Field(min_length=1, max_length=512)
    interpretations: list[Interpretation] = Field(min_length=1, max_length=512)
    node_index: dict[Code, list[Code]]
    inventory: Inventory
    legacy_scoring_binding: RecordBinding
    scoring_status: Literal["historical_v4_unchanged_not_recomputed_for_new_measurements"]
    clinical_acceptance: Literal[False]
    checksum: SHA

    @model_validator(mode="after")
    def relationships(self):
        obs = {o.id: o for o in self.observations}
        records = {j.id: j for j in self.interpretations}
        if len(obs) != len(self.observations) or len(records) != len(self.interpretations):
            raise ValueError("record_duplicate_id")
        if len({o.assessment_id for o in obs.values()}) != len(obs):
            raise ValueError("record_duplicate_assessment")
        if {j.observation_id for j in records.values()} != set(obs) or len(records) != len(obs):
            raise ValueError("record_orphan_observation")
        for j in records.values():
            o = obs[j.observation_id]
            if o.node != j.node or o.assessment_id != j.assessment_id:
                raise ValueError("record_cross_node_link")
            if j.status == "assessed" and o.acquisition != "obtained":
                raise ValueError("record_direction_without_observation")
            if j.status == "input_missing" and o.acquisition != "not_obtained":
                raise ValueError("record_missing_input_contradiction")
        indexed = [k for ids in self.node_index.values() for k in ids]
        if (
            len(self.node_index) > 128
            or len(set(indexed)) != len(indexed)
            or set(indexed) != set(records)
        ):
            raise ValueError("record_node_inventory")
        if any(records[k].node != node for node, ids in self.node_index.items() for k in ids):
            raise ValueError("record_node_index_binding")
        expected = {
            "total": len(obs),
            "observation_obtained": sum(o.acquisition == "obtained" for o in obs.values()),
            "observation_not_obtained": sum(o.acquisition == "not_obtained" for o in obs.values()),
            "direction_assessed": sum(j.status == "assessed" for j in records.values()),
            "measurement_without_direction": sum(
                obs[j.observation_id].result_kind in {"measurement", "proxy_measurement"}
                and j.status != "assessed"
                for j in records.values()
            ),
            "meaning": "record_counts_not_independent_evidence_or_clinical_coverage",
        }
        if self.inventory.model_dump() != expected:
            raise ValueError("record_count_mismatch")
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("record_checksum_mismatch")
        return self
