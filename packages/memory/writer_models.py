"""10B additive, reference-only research envelopes; legacy bytes retain 1.0 dispatch."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, validate
from packages.memory.models import ReadResult, Record, Source
from packages.planner_control.models import Graph, Reference

MAX_EVENTS = 256
SEGMENT_EVENTS = 16


class ExecutionSource(Source):
    owner: Literal["runtime_execution"] = "runtime_execution"
    kind: Literal["execution_snapshot"] = "execution_snapshot"
    version: Literal["research-memory-execution/1.0.0"] = "research-memory-execution/1.0.0"


class FieldRef(Value):
    """A saved owner value location, or explicit absence; never caller JSON in a key."""

    source: Checksum | None = None
    path: Annotated[str, Field(pattern=r"^/[A-Za-z0-9_/-]+$", max_length=256)] | None = None
    checksum: Checksum | None = None
    missing_reason: Token | None = None

    @model_validator(mode="after")
    def exactly_one(self):
        if self.missing_reason is not None:
            if any(v is not None for v in (self.source, self.path, self.checksum)):
                raise ValueError("memory_field_ambiguous")
        elif any(v is None for v in (self.source, self.path, self.checksum)):
            raise ValueError("memory_field_missing_reason")
        return self


class Terminal(Value):
    scope: Literal["planner_control", "runtime_lifecycle", "preparation_rejection"]
    runtime_status: Token
    control_phase: Token | None
    lifecycle_phase: Token | None
    technical_status: Token | None
    reason: Token
    cancelled: bool
    execution_confirmed: Literal[True] = True
    evidence: FieldRef


class Step(Value):
    index: int = Field(ge=0, lt=MAX_EVENTS)
    command_key: Token
    original_revision: int = Field(ge=0)
    action: Token
    before: Token
    after: Token
    event: FieldRef
    state: FieldRef
    legal_actions: FieldRef
    selection: FieldRef
    execution: FieldRef
    observation: FieldRef
    feedback: FieldRef
    expert_advice: FieldRef
    expert_decision: FieldRef
    review_result: FieldRef
    controlled_decision: FieldRef
    domain_acceptance: FieldRef
    versions: FieldRef
    status: Token
    reason: Token


class Segment(Value):
    type: Literal["trajectory_segment"] = "trajectory_segment"
    start: int = Field(ge=0, lt=MAX_EVENTS)
    steps: tuple[Step, ...] = Field(min_length=1, max_length=SEGMENT_EVENTS)

    @model_validator(mode="after")
    def ordered(self):
        if tuple(s.index for s in self.steps) != tuple(
            range(self.start, self.start + len(self.steps))
        ):
            raise ValueError("memory_event_order")
        if len({s.command_key for s in self.steps}) != len(self.steps):
            raise ValueError("memory_event_duplicate")
        return self


class Snapshot(Value):
    type: Literal["graph_snapshot"] = "graph_snapshot"
    role: Literal["initial_historical", "final_current"]
    graph: Graph
    evidence: FieldRef


class EpisodeData(Value):
    type: Literal["episode"] = "episode"
    request_key: Token
    input_checksum: Checksum
    terminal: Terminal
    initial_graph: FieldRef
    final_graph: FieldRef
    scores: FieldRef
    versions: tuple[Reference, ...] = Field(max_length=64)
    runtime_versions: FieldRef
    execution_attempts: FieldRef
    runtime_report: FieldRef
    trajectory_count: int = Field(ge=0, le=MAX_EVENTS)
    trajectory_missing_reason: Token | None
    segments: tuple[Token, ...] = Field(max_length=16)
    graph_members: tuple[Token, ...] = Field(max_length=2)
    supplemental_members: tuple[Token, ...] = Field(max_length=16)
    completion: Literal["atomic_candidate"] = "atomic_candidate"
    completeness: Literal["recorded_events_only_missing_fields_explicit"] = (
        "recorded_events_only_missing_fields_explicit"
    )
    extension_state: Literal["only_original_intent_no_dispatch"] = (
        "only_original_intent_no_dispatch"
    )
    cross_run_rule: Literal["independent_episode_original_source_links_only"] = (
        "independent_episode_original_source_links_only"
    )

    @model_validator(mode="after")
    def count(self):
        if len(self.segments) != (self.trajectory_count + SEGMENT_EVENTS - 1) // SEGMENT_EVENTS:
            raise ValueError("memory_segment_count")
        if (self.trajectory_count == 0) != (self.trajectory_missing_reason is not None):
            raise ValueError("memory_trajectory_missing_reason")
        return self


class WriterRecord(Record):
    schema_version: Literal["research-memory/1.1.0"] = "research-memory/1.1.0"
    sources: tuple[Source | ExecutionSource, ...] = Field(min_length=1, max_length=64)
    detail: Annotated[EpisodeData | Segment | Snapshot, Field(discriminator="type")]

    @model_validator(mode="after")
    def detail_kind(self):
        expected = {
            "episode": "episode",
            "trajectory_segment": "trajectory_step",
            "graph_snapshot": "graph_snapshot",
        }
        if self.kind != expected[self.detail.type]:
            raise ValueError("memory_detail_kind")
        if isinstance(self.detail, EpisodeData):
            keys = (
                self.detail.segments + self.detail.graph_members + self.detail.supplemental_members
            )
            if set(keys) != {p.key for p in self.parents} or len(keys) != len(self.parents):
                raise ValueError("memory_episode_members")
            if any(p.relation != "episode_member" for p in self.parents):
                raise ValueError("memory_episode_members")
        return self


class WriterReadResult(ReadResult):
    record: WriterRecord


def parse_record(value):
    version = value.get("schema_version") if isinstance(value, dict) else value.schema_version
    if version in {"research-memory/1.3.0", "research-memory/1.4.0"}:
        from packages.memory.archive_models import ArchiveRecord, ExtendedArchiveRecord

        return validate(
            ExtendedArchiveRecord if version == "research-memory/1.4.0" else ArchiveRecord, value
        )
    if version == "research-memory/1.2.0":
        from packages.memory.distillation_models import DistillationRecord

        return validate(DistillationRecord, value)
    return validate(WriterRecord if version == "research-memory/1.1.0" else Record, value)


def parse_source(value):
    owner = value.get("owner") if isinstance(value, dict) else value.owner
    if owner == "runtime_archive":
        from packages.memory.archive_models import RunArchiveSource

        return validate(RunArchiveSource, value)
    if owner == "case_artifact":
        from packages.memory.archive_models import ArtifactSource

        return validate(ArtifactSource, value)
    return validate(ExecutionSource if owner == "runtime_execution" else Source, value)
