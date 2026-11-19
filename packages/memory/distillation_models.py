"""10C candidate content, explicitly dispatched separately from 1.0 and 1.1."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from packages.grounding.models import Checksum, Token, Value, canonical
from packages.memory.models import Dependency, ReadResult, Record, Source
from packages.memory.writer_models import ExecutionSource, FieldRef, Terminal


class Context(Value):
    episode: Dependency
    case_code: Token
    run_id: Token
    input_checksum: Checksum | None
    input_missing_reason: Token | None
    episode_input_checksum: Checksum
    runtime_context_checksum: Checksum
    terminal: Terminal
    graph: FieldRef
    scores: FieldRef
    versions: tuple[dict[str, JsonValue], ...]
    runtime_versions: FieldRef
    scope: Literal["exact_saved_context_only"] = "exact_saved_context_only"
    patient_independence: Literal["unknown"] = "unknown"
    population: None = None
    clinical_validity: None = None


class Change(Value):
    intervention: Token
    target: Token
    operation: Literal["remove", "reverse", "replace"]
    removed: bool
    value_kind: Literal["interpretation_effect", "resource_reference"]
    original_value: JsonValue
    simulated_value: JsonValue
    original_observation: dict[str, JsonValue] | None
    simulated_observation: dict[str, JsonValue] | None
    missing_reason: Token | None

    # Values are saved judgments/resource references, never fabricated measurements.
    @model_validator(mode="after")
    def removal(self):
        if self.removed != (self.operation == "remove") or (
            self.removed and self.simulated_value is not None
        ):
            raise ValueError("memory_distillation_removal")
        return self


class PairData(Value):
    type: Literal["counterfactual_pair"] = "counterfactual_pair"
    request_key: Token
    context: Context
    report_source: Source
    simulation_id: Token
    policy: dict[str, JsonValue]
    assessment_contexts: dict[str, JsonValue]
    semantics_version: Literal["counterfactual-semantics/1.1.0"]
    changes: tuple[Change, ...]
    affected: tuple[dict[str, JsonValue], ...]
    remaining: tuple[Token, ...]
    baseline: dict[str, JsonValue]
    simulated: dict[str, JsonValue] | None
    comparisons: dict[str, JsonValue]
    alternatives: dict[str, JsonValue]
    classification: Literal[
        "necessary",
        "context_redundant",
        "components_changed",
        "conclusion_changed",
        "indeterminate",
    ]
    display: Token
    reasons: tuple[str, ...]
    review: dict[str, JsonValue]
    source_roles: tuple[dict[str, JsonValue], ...]
    limitations: tuple[Token, ...]
    original_counters: dict[str, JsonValue]
    review_status: Literal["pending_memory_review"] = "pending_memory_review"
    identity: Literal["hypothetical_comparison"] = "hypothetical_comparison"


class RecoveryData(Value):
    type: Literal["recovery_experience"] = "recovery_experience"
    request_key: Token
    context: Context
    report_source: Source
    failure: dict[str, JsonValue]
    policy_checksum: Checksum
    command: dict[str, JsonValue]
    before_phase: Token
    observed_phase: Token
    conclusion_status: Token
    stop_reason: Token
    unresolved: tuple[dict[str, JsonValue], ...]
    attempted_alternatives: tuple[Token, ...]
    retained_evidence: tuple[dict[str, JsonValue], ...]
    expert_result_checksum: Checksum
    expert_role: Literal["agent_opinion_not_fact"] = "agent_opinion_not_fact"
    recovery_success: Literal["not_inferred_from_report"] = "not_inferred_from_report"
    review_status: Literal["pending_memory_review"] = "pending_memory_review"


class Advice(Value):
    schema_version: Literal["memory-candidate-advice/1.0.0"] = "memory-candidate-advice/1.0.0"
    producer: Literal["deterministic"] = "deterministic"
    codes: tuple[Token, ...]
    executable: Literal[False] = False
    source_fact: Literal[False] = False


class AbstractData(Value):
    type: Literal["abstract_pattern"] = "abstract_pattern"
    request_key: Token
    contexts: tuple[Context, ...] = Field(min_length=1, max_length=16)
    supporting_episodes: tuple[Token, ...]
    counterexamples: tuple[dict[str, JsonValue], ...]
    experiences: tuple[Dependency, ...]
    missing: tuple[dict[str, JsonValue], ...]
    advice: Advice
    case_count: int = Field(ge=1)
    run_count: int = Field(ge=1)
    evidence_count: int = Field(ge=1)
    independent_patient_count: None = None
    source_roles: tuple[dict[str, JsonValue], ...]
    not_applicable: tuple[Token, ...]
    review_status: Literal["pending_memory_review"] = "pending_memory_review"
    generalization: Literal["unestablished_candidate_only"] = "unestablished_candidate_only"
    accuracy: None = None
    medical_causal_effect: None = None


class DistillationRecord(Record):
    schema_version: Literal["research-memory/1.2.0"] = "research-memory/1.2.0"
    sources: tuple[Source | ExecutionSource, ...] = Field(min_length=1, max_length=64)
    detail: Annotated[PairData | RecoveryData | AbstractData, Field(discriminator="type")]

    @model_validator(mode="after")
    def bindings(self):
        # Override only this version. Old Record/Writer validators and schemas stay intact.
        if self.kind != self.detail.type:
            raise ValueError("memory_detail_kind")
        if len({(s.owner, s.run_id, s.key, s.kind) for s in self.sources}) != len(self.sources):
            raise ValueError("memory_duplicate_source")
        if len({p.key for p in self.parents}) != len(self.parents) or self.key in {
            p.key for p in self.parents
        }:
            raise ValueError("memory_dependency_cycle")
        if not self.parents or any(p.relation != "derived_from" for p in self.parents):
            raise ValueError("memory_distillation_dependencies")
        if (self.case_code, self.run_id) not in {(s.case_code, s.run_id) for s in self.sources}:
            raise ValueError("memory_scope")
        if self.kind != "abstract_pattern" and any(
            s.case_code != self.case_code for s in self.sources
        ):
            raise ValueError("memory_scope")
        if self.kind == "recovery_experience" and any(
            s.run_id != self.run_id for s in self.sources
        ):
            raise ValueError("memory_scope")
        if any(not set(self.applicability) <= set(s.applicability) for s in self.sources):
            raise ValueError("memory_applicability")
        expected = {"counterfactual_pair": "simulation", "abstract_pattern": "abstract_candidate"}
        nature = expected.get(
            self.kind,
            self.detail.report_source.nature if isinstance(self.detail, RecoveryData) else None,
        )
        if self.nature != nature:
            raise ValueError("memory_kind_source")
        if len(canonical(self)) > 48000:
            raise ValueError("memory_distillation_size")
        return self


class DistillationReadResult(ReadResult):
    record: DistillationRecord
