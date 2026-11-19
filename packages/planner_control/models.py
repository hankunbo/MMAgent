"""08A control values; deliberately outside frozen clinical contracts v1."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_serializer, model_validator

from packages.grounding.models import Checksum, Code, Ref, Token, Value, check_seal, digest
from packages.grounding.query_models import METRICS, Component, Dependency, Metric

Count = Annotated[int, Field(ge=0, le=10000)]


class Phase(StrEnum):
    INITIALIZE = "initialize"
    RETRIEVE = "retrieve"
    INSTANTIATE = "instantiate"
    IDENTIFY = "identify"
    SELECT = "select"
    PREPARE = "prepare"
    EXECUTE = "execute"
    VERIFY = "verify"
    REVIEW = "review"
    FALLBACK = "fallback"
    STOPPED = "stopped"


class Reference(Value):
    role: Token
    code: Token
    version: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")]
    checksum: Checksum


class Graph(Reference):
    kind: Literal["research_artifact", "formal_geceg"]
    parent: Reference | None = None

    @model_validator(mode="after")
    def identity(self):
        if not self.code.startswith("ASSET:" if self.kind == "research_artifact" else "GRAPH:"):
            raise ValueError("graph_kind_identity")
        return self


class Profile(Value):
    version: Literal["planner-baseline/1.0.0"] = "planner-baseline/1.0.0"
    baseline_only: Literal[True] = True
    max_steps: Annotated[int, Field(ge=1, le=256)] = 48
    max_actions: Count = 4
    max_branches: Count = 4
    max_retries: Count = 2
    max_no_progress: Count = 4
    seconds: Annotated[int, Field(ge=1, le=3600)] = 300
    expert_required: bool = False


class Binding(Value):
    case_code: Code
    run_id: Code
    run_context_checksum: Checksum
    input_checksum: Checksum
    visible_checksum: Checksum
    graph: Graph
    locks: tuple[Reference, ...]
    profile: Profile
    started_at: datetime
    deadline: datetime

    @model_validator(mode="after")
    def exact(self):
        if not self.case_code.startswith("CASE:") or not self.run_id.startswith("RUN:"):
            raise ValueError("scope_identity")
        if self.started_at.tzinfo is None or self.deadline.tzinfo is None:
            raise ValueError("absolute_deadline_required")
        if self.deadline != self.started_at + timedelta(seconds=self.profile.seconds):
            raise ValueError("deadline_must_inherit_started_at")
        roles = [r.role for r in self.locks]
        if len(roles) != len(set(roles)) or not {
            "knowledge",
            "template",
            "rule",
            "catalog",
            "mapping",
            "execution",
            "ranking",
            "prompt",
            "model",
            "sources",
        }.issubset(roles):
            raise ValueError("exact_version_locks_required")
        return self


class Candidate(Value):
    id: Token
    branch: Token
    operation: Literal["observe", "tool", "new_review", "new_branch"]
    input_refs: tuple[Reference, ...] = ()
    and_ready: bool


class Gap(Value):
    kind: Literal[
        "missing_data",
        "product_demand",
        "mapping",
        "unknown_producer",
        "capability",
        "safety",
        "technical",
        "unavailable_port",
    ]
    target: Token
    reason: Token
    capability_code: Code | None = None
    requirement_ref: Reference | None = None
    dispatch: Literal["recorded_for_11_not_dispatched"] = "recorded_for_11_not_dispatched"

    @model_validator(mode="after")
    def resolved(self):
        if (self.kind == "capability") != (self.capability_code is not None):
            raise ValueError("resolved_capability_only")
        if self.kind == "capability" and self.requirement_ref is None:
            raise ValueError("resolved_requirement_required")
        return self


class Outcome(Value):
    status: Literal["ok", "gap", "unavailable", "rejected", "unknown"] = "ok"
    reason: Token = "confirmed"
    refs: tuple[Reference, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    gaps: tuple[Gap, ...] = ()
    started: Count = 0
    confirmed: Count = 0
    progress: bool = True


Action = Literal[
    "load",
    "retrieve",
    "instantiate",
    "identify",
    "select",
    "prepare",
    "execute",
    "recover",
    "verify",
    "review",
    "fallback",
    "stop",
    "cancel",
]


class Command(Value):
    key: Token
    expected_revision: Count
    action: Action
    input_refs: tuple[Reference, ...] = ()


class Pending(Value):
    command: Command
    request_checksum: Checksum
    execution_key: Token
    launch_claimed: bool = False
    reserved: bool = False


class RankingPolicy(Value):
    version: Literal["planner-ranking/1.0.0"] = "planner-ranking/1.0.0"
    baseline_only: Literal[True] = True
    order: tuple[Metric, ...] = ("directness", "cost", "latency", "failure_risk")
    information_gain_enabled: bool = False
    missing_order: Literal["last"] = "last"
    tie_break: Literal["semantic_action"] = "semantic_action"
    failure_history_first: Literal[True] = True
    max_agent_attempts: Annotated[int, Field(ge=0, le=3)] = 1
    suggestion_seconds: Annotated[int, Field(ge=1, le=120)] = 30
    fallback: Literal["semantic_baseline", "legacy_id_baseline"] = "semantic_baseline"

    @model_validator(mode="after")
    def coherent(self):
        if not self.order or len(set(self.order)) != len(self.order):
            raise ValueError("ranking_order_unique_required")
        return self


ActionKind = Literal["observe", "select_candidate", "request_review", "fallback", "stop"]


class LegalAction(Value):
    schema_version: Literal["planner-action/1.0.0", "planner-research-action/1.0.0"] = (
        "planner-action/1.0.0"
    )
    id: Token
    kind: ActionKind
    target: Reference
    branch: Token
    candidate: Candidate | None
    candidate_ref: Checksum | None
    required_inputs: tuple[Reference, ...]
    dependencies: tuple[Dependency, ...] = ()
    expected_artifact: Ref | None = None
    entrypoint: Literal[
        "ResearchPlannerSteps.invoke",
        "ToolAdmissionService.admit",
        "Planner.step",
        "ResearchReview.run",
    ]
    preconditions: tuple[Token, ...]
    reasons: tuple[Token, ...]
    components: tuple[Component, ...]
    failure_count: Count = 0
    failure_history: tuple[Checksum, ...] = ()
    failure_history_scope: Literal["current_run"] = "current_run"
    validation_level: Token = "unknown"
    review_required: bool = False
    expires_at: datetime

    def semantic_key(self):
        # Display/legacy candidate IDs and estimates never decide semantic ties.
        return (
            self.kind,
            self.target.code,
            self.target.version,
            self.branch,
            self.candidate_ref or "",
            digest(self.required_inputs),
        )

    def identity_checksum(self):
        return digest(
            dict(
                kind=self.kind,
                target=self.target,
                branch=self.branch,
                candidate_ref=self.candidate_ref,
                inputs=self.required_inputs,
                entrypoint=self.entrypoint,
            )
        )

    @model_validator(mode="after")
    def coherent(self):
        if (self.kind == "request_review") != (
            self.schema_version == "planner-research-action/1.0.0"
        ):
            raise ValueError("research_action_version_required")
        if self.failure_count != len(self.failure_history):
            raise ValueError("failure_history_binding")
        if self.id != "action:" + self.identity_checksum()[7:]:
            raise ValueError("action_identity_mismatch")
        if self.expires_at.tzinfo is None or not self.reasons or not self.preconditions:
            raise ValueError("action_conditions_required")
        if tuple(c.name for c in self.components) != METRICS:
            raise ValueError("action_components_required")
        if len(set(self.required_inputs)) != len(self.required_inputs):
            raise ValueError("duplicate_required_input")
        expected = {
            "observe": "ResearchPlannerSteps.invoke",
            "select_candidate": "ToolAdmissionService.admit",
            "fallback": "Planner.step",
            "stop": "Planner.step",
            "request_review": "ResearchReview.run",
        }
        if expected.get(self.kind) != self.entrypoint:
            raise ValueError("action_entrypoint_not_installed")
        if self.kind in {"observe", "select_candidate"}:
            if self.candidate is None or not self.candidate.and_ready:
                raise ValueError("complete_admitted_candidate_required")
            if self.candidate.branch != self.branch:
                raise ValueError("candidate_branch_mismatch")
            if self.candidate.operation != (
                "tool" if self.kind == "select_candidate" else "observe"
            ):
                raise ValueError("candidate_operation_mismatch")
        elif self.candidate is not None or self.candidate_ref is not None:
            raise ValueError("control_action_has_no_candidate")
        if self.kind == "select_candidate" and (
            self.candidate_ref is None or not self.dependencies or self.expected_artifact is None
        ):
            raise ValueError("typed_grounding_candidate_required")
        return self


class ActionRejection(Value):
    target: Token
    category: Literal[
        "empty",
        "search_truncated",
        "missing_data",
        "unknown_producer",
        "missing_implementation",
        "safety",
        "technical",
        "and_unsatisfied",
        "budget",
        "already_attempted",
        "graph_not_legal",
    ]
    reason: Token
    source_checksum: Checksum | None = None


class ActionBudget(Value):
    steps: Count
    reserved: Count
    started: Count
    confirmed: Count
    branches: Count
    retries: Count
    no_progress: Count
    cancelled: bool


class LegalActionSet(Value):
    schema_version: Literal["legal-action-set/1.0.0"] = "legal-action-set/1.0.0"
    binding: Binding
    graph: Graph
    state_checksum: Checksum
    budget: ActionBudget
    policy: RankingPolicy
    safety_checksum: Checksum
    availability_checksum: Checksum
    source_checksums: tuple[Checksum, ...]
    plan_refs: tuple[Reference, ...]
    actions: tuple[LegalAction, ...]
    rejections: tuple[ActionRejection, ...]
    enumeration: Literal["complete", "search_truncated"] = "complete"
    expires_at: datetime
    checksum: Checksum

    @field_serializer("expires_at")
    def serialize_expiry(self, value):
        return value.isoformat()

    @model_validator(mode="after")
    def coherent(self):
        lock = next(r for r in self.binding.locks if r.role == "ranking")
        if (lock.version, lock.checksum) != (self.policy.version, digest(self.policy)):
            raise ValueError("action_set_policy_lock")
        if any(r.role != "plan" for r in self.plan_refs) or len(set(self.plan_refs)) != len(
            self.plan_refs
        ):
            raise ValueError("action_set_plan_refs")
        if self.expires_at.tzinfo is None or self.expires_at > self.binding.deadline:
            raise ValueError("action_set_deadline")
        if self.graph.kind != self.binding.graph.kind:
            raise ValueError("action_set_graph_kind")
        if len({a.id for a in self.actions}) != len(self.actions):
            raise ValueError("duplicate_action")
        if self.actions != tuple(sorted(self.actions, key=lambda a: a.semantic_key())):
            raise ValueError("noncanonical_action_set")
        if any(a.expires_at != self.expires_at for a in self.actions):
            raise ValueError("action_expiry_mismatch")
        if self.source_checksums != tuple(sorted(set(self.source_checksums))):
            raise ValueError("noncanonical_sources")
        check_seal(self)
        return self


class TraceRanking(Value):
    schema_version: Literal["trace-ranking/1.0.0"] = "trace-ranking/1.0.0"
    set_checksum: Checksum
    policy_checksum: Checksum
    ranked_ids: tuple[Token, ...] = Field(min_length=1, max_length=1024)
    reason: Annotated[str, Field(min_length=1, max_length=240, pattern=r"^[^\x00-\x1f]+$")]

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.ranked_ids)) != len(self.ranked_ids):
            raise ValueError("duplicate_ranking")
        return self


class ActionDecision(Value):
    schema_version: Literal["planner-selection/1.0.0"] = "planner-selection/1.0.0"
    action_set: LegalActionSet
    ranking: TraceRanking
    strategy: Literal["trace-ranking/1.0.0", "semantic-baseline/1.0.0", "legacy-id/1.0.0"]
    selected_id: Token
    attempts: Annotated[int, Field(ge=0, le=3)]
    rejected_advice: tuple[Token, ...] = ()
    decided_at: datetime

    @model_validator(mode="after")
    def coherent(self):
        s, r = self.action_set, self.ranking
        if r.set_checksum != s.checksum or r.policy_checksum != digest(s.policy):
            raise ValueError("ranking_binding_mismatch")
        if not set(r.ranked_ids) <= {a.id for a in s.actions}:
            raise ValueError("unknown_action_id")
        if self.selected_id != r.ranked_ids[0]:
            raise ValueError("selection_not_ranked_first")
        if (
            self.decided_at.tzinfo is None
            or not s.binding.started_at <= self.decided_at < s.expires_at
        ):
            raise ValueError("expired_selection")
        if self.strategy != "trace-ranking/1.0.0":
            from .actions import baseline

            if self.ranking != baseline(s, legacy=self.strategy == "legacy-id/1.0.0"):
                raise ValueError("baseline_order_mismatch")
        if self.attempts > s.policy.max_agent_attempts:
            raise ValueError("agent_attempt_budget")
        return self

    @property
    def selected(self):
        return next(a for a in self.action_set.actions if a.id == self.selected_id)


class SelectionPending(Pending):
    schema_version: Literal["planner-selection-pending/1.0.0"] = "planner-selection-pending/1.0.0"
    decision: ActionDecision

    @model_validator(mode="after")
    def coherent(self):
        if self.command.action != "select" or self.launch_claimed or self.reserved:
            raise ValueError("selection_pending_phase")
        return self


class SelectionOutcome(Outcome):
    schema_version: Literal["planner-selection-outcome/1.0.0"] = "planner-selection-outcome/1.0.0"
    decision: ActionDecision


class Event(Value):
    schema_version: Literal["planner-event/1.0.0"] = "planner-event/1.0.0"
    command: Command
    request_checksum: Checksum
    before: Phase
    after: Phase
    graph: Graph
    outcome: Outcome | SelectionOutcome
    at: datetime


class Control(Value):
    schema_version: Literal["planner-control/1.0.0"] = "planner-control/1.0.0"
    binding: Binding
    phase: Phase = Phase.INITIALIZE
    graph: Graph
    revision: Count = 0
    steps: Count = 0
    reserved: Count = 0
    started: Count = 0
    confirmed: Count = 0
    api_requests: Literal[0] = 0
    observed_cost: None = None
    branches: Count = 0
    retries: Count = 0
    no_progress: Count = 0
    cancelled: bool = False
    selected: Candidate | None = None
    candidates: tuple[Candidate, ...] = ()
    attempted: tuple[Token, ...] = ()
    result_refs: tuple[Reference, ...] = ()
    gaps: tuple[Gap, ...] = ()
    pending: Pending | SelectionPending | None = None
    events: tuple[Event, ...] = ()
    memory: tuple[Reference, ...] = ()
    reason: Token | None = None

    @model_validator(mode="after")
    def invariants(self):
        if self.graph.kind != self.binding.graph.kind:
            raise ValueError("graph_kind_immutable")
        if len({e.command.key for e in self.events}) != len(self.events):
            raise ValueError("duplicate_step_key")
        if self.memory:
            raise ValueError("memory_not_installed")
        if self.steps != len(self.events):
            raise ValueError("event_count_mismatch")
        decisions = [e for e in self.events if isinstance(e.outcome, SelectionOutcome)]
        for event in decisions:
            decision = event.outcome.decision
            if (
                event.command.action != "select"
                or decision.action_set.binding != self.binding
                or decision.action_set.graph != event.graph
                or decision.decided_at > event.at
            ):
                raise ValueError("selection_event_scope")
        if decisions and self.phase in {Phase.PREPARE, Phase.EXECUTE, Phase.VERIFY, Phase.REVIEW}:
            if self.selected != decisions[-1].outcome.decision.selected.candidate:
                raise ValueError("selected_state_mismatch")
        if isinstance(self.pending, SelectionPending):
            from .actions import state_checksum

            if (
                self.pending.decision.action_set.binding != self.binding
                or self.pending.decision.action_set.graph != self.graph
                or self.pending.decision.action_set.state_checksum != state_checksum(self)
            ):
                raise ValueError("selection_pending_scope")
        if self.reserved > self.binding.profile.max_actions:
            raise ValueError("reservation_budget")
        return self


def request_checksum(binding, command):
    return digest(dict(binding=binding, command=command))


class ControlReport(Value):
    schema_version: Literal["planner-control-report/1.0.0"] = "planner-control-report/1.0.0"
    run_id: Code
    control_checksum: Checksum
    phase: Phase
    reason: Token | None
    graph: Graph
    deadline: datetime
    pending_execution_key: Token | None
    reserved_actions: Count
    started_processes: Count
    confirmed_workflows: Count
    api_requests: Literal[0] = 0
    observed_cost: None = None
    clinical_effect: None = None
    result_refs: tuple[Reference, ...]
    unresolved: tuple[Gap, ...]
