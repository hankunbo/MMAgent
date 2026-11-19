"""Separate research representation. Frozen clinical contracts v1 are never relaxed."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.contracts import CounterfactualPair, Episode, TrajectoryStep
from packages.contracts.models import CaseCode, RunCode
from packages.grounding.models import Checksum, Code, Token, Value, digest, validate

Kind = Literal[
    "episode",
    "graph_snapshot",
    "trajectory_step",
    "counterfactual_pair",
    "recovery_experience",
    "tool_experience",
    "abstract_pattern",
]
State = Literal["candidate", "reviewed", "published", "rejected", "deprecated"]
Nature = Literal[
    "source_fact",
    "agent_opinion",
    "technical_failure",
    "missing_evidence",
    "simulation",
    "abstract_candidate",
]


class Source(Value):
    owner: Literal["research_fact", "planner_event", "simulation", "recovery", "agent_review"]
    case_code: CaseCode
    run_id: RunCode
    key: Token
    object_code: Token
    kind: Token
    version: Annotated[str, Field(min_length=1, max_length=128)]
    checksum: Checksum
    object_checksum: Checksum
    dependencies: tuple[Checksum, ...] = Field(min_length=1, max_length=64)
    semantics_version: Literal["counterfactual-semantics/1.1.0"] | None = None
    purpose: Literal["research_offline"] = "research_offline"
    applicability: tuple[Code, ...] = Field(min_length=1, max_length=32)
    nature: Nature


class Dependency(Value):
    key: Token
    checksum: Checksum
    relation: Literal["derived_from", "revision_of", "episode_member"] = "derived_from"


class Record(Value):
    schema_version: Literal["research-memory/1.0.0"] = "research-memory/1.0.0"
    identity: Literal["research_candidate"] = "research_candidate"
    key: Token
    kind: Kind
    case_code: CaseCode
    run_id: RunCode
    purpose: Literal["research_offline"] = "research_offline"
    applicability: tuple[Code, ...] = Field(min_length=1, max_length=32)
    sources: tuple[Source, ...] = Field(min_length=1, max_length=64)
    parents: tuple[Dependency, ...] = Field(default=(), max_length=64)
    # Structured references only: no report text, hidden labels, clinical notes or vectors.
    nature: Nature

    @model_validator(mode="after")
    def bindings(self):
        if len({(s.owner, s.run_id, s.key, s.kind) for s in self.sources}) != len(self.sources):
            raise ValueError("memory_duplicate_source")
        if len({p.key for p in self.parents}) != len(self.parents) or self.key in {
            p.key for p in self.parents
        }:
            raise ValueError("memory_dependency_cycle")
        if (self.case_code, self.run_id) not in {(s.case_code, s.run_id) for s in self.sources}:
            raise ValueError("memory_scope")
        if self.kind != "abstract_pattern" and any(
            (s.case_code, s.run_id) != (self.case_code, self.run_id) for s in self.sources
        ):
            raise ValueError("memory_scope")
        if any(not set(self.applicability) <= set(s.applicability) for s in self.sources):
            raise ValueError("memory_applicability")
        required = {
            "episode": ("source_fact", "research_fact"),
            "graph_snapshot": ("source_fact", "research_fact"),
            "trajectory_step": ("source_fact", "planner_event"),
            "counterfactual_pair": ("simulation", "simulation"),
            "recovery_experience": ("technical_failure", "recovery"),
            "tool_experience": ("source_fact", "planner_event"),
            "abstract_pattern": ("abstract_candidate", None),
        }
        nature, owner = required[self.kind]
        if self.schema_version == "research-memory/1.1.0":
            owner = "runtime_execution"
        if self.kind == "recovery_experience":
            if self.nature not in {"technical_failure", "missing_evidence", "source_fact"}:
                raise ValueError("memory_kind_source")
            nature = self.nature
        if self.nature != nature or (owner and not any(s.owner == owner for s in self.sources)):
            raise ValueError("memory_kind_source")
        if self.kind == "recovery_experience" and not any(
            s.owner == "recovery" and s.nature == self.nature for s in self.sources
        ):
            raise ValueError("memory_kind_source")
        if self.kind == "tool_experience" and not any(
            s.kind == "tool_execution_event" for s in self.sources
        ):
            raise ValueError("memory_tool_event_required")
        if (
            self.kind == "graph_snapshot"
            and self.schema_version == "research-memory/1.0.0"
            and not any(s.kind == "graph" for s in self.sources)
        ):
            raise ValueError("memory_graph_reference_required")
        if self.kind == "abstract_pattern" and not self.parents:
            raise ValueError("memory_parents_required")
        return self


def validate_relationships(record: Record, parents: dict[str, Record]) -> None:
    """R01 current-validity policy; historical 1.0 payloads retain their exact bytes.

    A member is the parent dependency of its child Episode container.
    Caller-supplied relation labels never confer scope or abstract identity.
    """
    for dependency in record.parents:
        parent = parents[dependency.key]
        if dependency.checksum != digest(parent):
            raise ValueError("memory_parent_checksum")
        relation = dependency.relation
        if relation == "episode_member":
            if record.kind != "episode" or parent.kind not in {
                "graph_snapshot",
                "trajectory_step",
                "counterfactual_pair",
                "recovery_experience",
                "tool_experience",
            }:
                raise ValueError("memory_relation_kind")
        elif relation == "revision_of":
            if record.kind != parent.kind or record.kind == "abstract_pattern":
                raise ValueError("memory_relation_kind")
        elif record.kind != "abstract_pattern" and parent.kind == "abstract_pattern":
            raise ValueError("memory_relation_kind")
        if parent.purpose != record.purpose or (
            not (relation == "derived_from" and record.kind == "abstract_pattern")
            and (parent.case_code, parent.run_id) != (record.case_code, record.run_id)
        ):
            raise ValueError("memory_relation_scope")
        if not set(record.applicability) <= set(parent.applicability):
            raise ValueError("memory_relation_applicability")
    if record.kind == "abstract_pattern":
        inherited = {digest(s) for p in parents.values() for s in p.sources}
        if inherited != {digest(s) for s in record.sources}:
            raise ValueError("memory_derivation_sources")


class Review(Value):
    schema_version: Literal["memory-review/1.0.0"] = "memory-review/1.0.0"
    key: Token
    target: Token
    target_checksum: Checksum
    decision: Literal["approve", "reject", "comment"]
    basis: Literal["engineering_review", "agent_opinion"]
    reason: Code
    source: Source | None = None
    clinical_approval: Literal[False] = False

    @model_validator(mode="after")
    def opinion_binding(self):
        if (
            self.source is not None
            and self.source.owner == "agent_review"
            and self.basis != "agent_opinion"
        ):
            raise ValueError("memory_agent_opinion_not_engineering_review")
        if self.basis == "agent_opinion" and (
            self.source is None or self.source.owner != "agent_review"
        ):
            raise ValueError("memory_review_source_required")
        return self


class EmbeddingIndex(Value):
    schema_version: Literal["memory-index/1.0.0"] = "memory-index/1.0.0"
    key: Token
    target: Token
    target_checksum: Checksum
    model_code: Code
    model_version: Token
    dimension: int = Field(ge=1, le=65536)
    content_checksum: Checksum
    status: Literal["metadata_only"] = "metadata_only"


class ReadResult(Value):
    record: Record
    checksum: Checksum
    state: State
    invalid_reason: Token | None
    current: bool
    eligible_for_retrieval: Literal[False] = False
    eligible_for_planning: Literal[False] = False
    clinical_level: Literal["not_evaluated"] = "not_evaluated"


def read_formal(raw: bytes):
    """Validate exact old bytes without rewriting them or granting current authority."""
    import json

    value = json.loads(raw)
    cls = {
        "Episode": Episode,
        "TrajectoryStep": TrajectoryStep,
        "CounterfactualPair": CounterfactualPair,
    }.get(value.get("contract_type"))
    if cls is None:
        raise ValueError("memory_formal_type")
    return validate(cls, value), raw
