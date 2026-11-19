"""08C value contracts; requests contain only owner-approved projections."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Code, Token, Value, digest
from packages.planner_control.models import Graph, Reference

Role = Literal["trace", "expert", "source", "vision", "review"]
Text = Annotated[str, Field(min_length=1, max_length=240)]
Error = Literal[
    "schema",
    "refusal",
    "truncated",
    "rate_limit",
    "timeout",
    "authentication",
    "no_result",
    "transport",
    "configuration",
    "budget",
    "deadline",
    "scope",
    "unsupported_image",
    "secret_response",
]


class Scope(Value):
    case_code: Code
    run_id: Code
    graph: Graph
    visible_checksum: Checksum
    input_checksum: Checksum
    material_checksum: Checksum
    policy_checksum: Checksum


class Source(Value):
    reference: Reference
    locator: Text
    purpose: Literal["published_evidence", "research_material"]
    state: Literal["published", "research_only"]
    release: Reference | None = None
    text: Annotated[str, Field(min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def provenance(self):
        if (self.purpose == "published_evidence") != (self.state == "published"):
            raise ValueError("source_purpose")
        if (self.state == "published") != (self.release is not None):
            raise ValueError("source_release")
        return self


class ImageBinding(Value):
    reference: Reference
    media_type: Literal["image/png", "image/jpeg"]
    width: Annotated[int, Field(ge=1, le=32768)]
    height: Annotated[int, Field(ge=1, le=32768)]
    coordinate_system: Literal["pixel_xy_top_left"]
    preprocessing: Reference
    location: tuple[int, int, int, int]

    @model_validator(mode="after")
    def bounds(self):
        x, y, w, h = self.location
        if min(x, y) < 0 or min(w, h) < 1 or x + w > self.width or y + h > self.height:
            raise ValueError("image_location")
        return self


class ObservationInput(Value):
    reference: Reference
    status: Literal["obtained", "missing", "ungradable", "unknown"]
    method: Token
    unit: Token
    value: Annotated[str, Field(max_length=1000)] | None
    image: ImageBinding | None = None


class Material(Value):
    """Installed reader grants visibility; a checksum alone is never authorization."""

    scope: Scope
    goal_code: Token
    goal_summary: Text
    visibility: Literal["approved_visible_projection/1.0.0"]
    sources: tuple[Source, ...] = Field(max_length=16)
    observations: tuple[ObservationInput, ...] = Field(max_length=128)
    policy_ref: Reference
    missing: tuple[Token, ...] = Field(max_length=64)
    memory: tuple[()] = ()

    @model_validator(mode="after")
    def bindings(self):
        if self.policy_ref.checksum != self.scope.policy_checksum:
            raise ValueError("material_policy")
        if self.scope.graph.kind == "formal_geceg" and any(
            s.state != "published" for s in self.sources
        ):
            raise ValueError("formal_requires_published")
        if len({s.reference.code for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate_source")
        if len({o.reference.code for o in self.observations}) != len(self.observations):
            raise ValueError("duplicate_observation")
        raw = self.model_dump(mode="json")
        raw["scope"].pop("material_checksum")
        if digest(raw) != self.scope.material_checksum:
            raise ValueError("material_checksum")
        return self


class TaskDefinition(Value):
    """Visible, installed definition; a missing interpretation never invents a cutoff."""

    reference: Reference
    examination: Text
    applicability_conditions: Text
    interpretation: Text | None
    interpretation_status: Literal["installed", "missing"]
    source_refs: tuple[Reference, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bound(self):
        if (self.interpretation is not None) != (self.interpretation_status == "installed"):
            raise ValueError("task_interpretation_status")
        if len({r.code for r in self.source_refs}) != len(self.source_refs):
            raise ValueError("task_definition_sources")
        if self.reference.checksum != digest(self.model_dump(mode="json", exclude={"reference"})):
            raise ValueError("task_definition_checksum")
        return self


class ReviewTask(Value):
    assessment_id: Token
    node: Token
    goal: Token
    definition: TaskDefinition
    observation: Reference
    allowed_sources: tuple[Reference, ...] = Field(min_length=1, max_length=16)
    method: Token
    unit: Token
    laterality: Literal["LEFT", "RIGHT", "BILATERAL", "UNKNOWN"]
    applicable: bool

    @model_validator(mode="after")
    def sources_unique(self):
        if len({r.code for r in self.allowed_sources}) != len(self.allowed_sources):
            raise ValueError("task_source_duplicates")
        return self


class ReviewTemplate(Value):
    schema_version: Literal["review-task-projection/2.0.0"] = "review-task-projection/2.0.0"
    scope_checksum: Checksum
    policy_checksum: Checksum
    scale: Reference
    tasks: tuple[ReviewTask, ...] = Field(max_length=128)
    fixed_assessment_ids: tuple[Token, ...] = Field(max_length=128)

    @property
    def assessment_ids(self):
        return tuple(t.assessment_id for t in self.tasks)

    @model_validator(mode="after")
    def unique(self):
        ids = self.assessment_ids + self.fixed_assessment_ids
        if len(set(ids)) != len(ids):
            raise ValueError("task_projection_duplicates")
        return self

    def check_material(self, material):
        if (self.scope_checksum, self.policy_checksum) != (
            digest(material.scope),
            material.scope.policy_checksum,
        ):
            raise ValueError("task_projection_scope")
        obs = {o.reference.code: o for o in material.observations}
        sources = {s.reference.code: s.reference for s in material.sources}
        for t in self.tasks:
            o = obs.get(t.observation.code)
            if o is None or (o.reference, o.method, o.unit) != (t.observation, t.method, t.unit):
                raise ValueError("task_projection_observation")
            if (
                t.goal != material.goal_code
                or not t.allowed_sources
                or any(sources.get(r.code) != r for r in t.allowed_sources)
                or set(t.definition.source_refs) != set(t.allowed_sources)
            ):
                raise ValueError("task_projection_sources")
        return self


class Usage(Value):
    status: Literal["observed", "unknown"] = "unknown"
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens: Annotated[int, Field(ge=0)] | None = None
    cost: None = None

    @model_validator(mode="after")
    def observed(self):
        known = any(
            v is not None for v in (self.input_tokens, self.output_tokens, self.total_tokens)
        )
        if known != (self.status == "observed"):
            raise ValueError("usage_observation")
        return self


class RequestRecord(Value):
    schema_version: Literal["agent-request/1.0.0"] = "agent-request/1.0.0"
    request_id: Token
    invocation_id: Token
    retry_index: Annotated[int, Field(ge=0, le=3)]
    role: Role
    scope: Scope
    provider: str = Field(max_length=128)
    model: str = Field(max_length=128)
    endpoint: Literal["chat_completions", "responses"]
    config_checksum: Checksum
    prompt_version: Token
    prompt_checksum: Checksum
    input_checksum: Checksum
    output_checksum: Checksum | None
    remote_request_checksum: Checksum | None
    attempt: Annotated[int, Field(ge=1)]
    simulated: bool
    # None means worker was interrupted before dispatch acknowledgement was observable.
    api_requests: Literal[0, 1] | None
    usage: Usage
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    error: Error | None
    deadline: datetime
    raw_audit: Literal["digest_only_no_raw_retention"] = "digest_only_no_raw_retention"


class CatalogRequestRecord(RequestRecord):
    schema_version: Literal["agent-catalog-request/1.0.0"] = "agent-catalog-request/1.0.0"
    endpoint: Literal["models_get"] = "models_get"
    purpose: Literal["token_model_catalog_probe_not_inference"] = (
        "token_model_catalog_probe_not_inference"
    )


def read_request_record(value):
    from packages.grounding.models import validate

    types = {
        "agent-request/1.0.0": RequestRecord,
        "agent-catalog-request/1.0.0": CatalogRequestRecord,
    }
    cls = types.get(value.get("schema_version"))
    if cls is None:
        raise ValueError("agent_request_record_version")
    return validate(cls, value)


class ExpertProposal(Value):
    schema_version: Literal["expert-proposal/1.0.0"] = "expert-proposal/1.0.0"
    scope_checksum: Checksum
    action: Literal["accept", "reject", "alternative", "safe_stop"]
    action_id: Token | None
    evidence: tuple[Token, ...] = Field(max_length=16)
    reason: Text


class SourceProposal(Value):
    schema_version: Literal["research-relation-candidate/1.0.0"] = (
        "research-relation-candidate/1.0.0"
    )
    scope_checksum: Checksum
    source_code: Token
    quote: Annotated[str, Field(min_length=1, max_length=2000)]
    target: Token
    relation: Literal["requires", "organizes", "derived_from", "research_association"]
    missing: tuple[Token, ...] = Field(max_length=32)
    reason: Text


class VisionProposal(Value):
    schema_version: Literal["visual-observation-candidate/1.0.0"] = (
        "visual-observation-candidate/1.0.0"
    )
    scope_checksum: Checksum
    image_checksum: Checksum
    preprocessing_checksum: Checksum
    location: tuple[int, int, int, int]
    status: Literal["observed", "ungradable", "unknown"]
    finding: Text


class PartialObservation(Value):
    """Task-relative estimates, not image measurements or calibrated risks."""

    coverage: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)
    basis: Literal["disc_arc", "cup_disc_boundaries", "retinal_roi", "vessel_segments"]
    basis_detail: Text
    visible: Text
    unseen: Text
    signal: Literal["present", "absent_in_visible_region", "indeterminate"]
    estimate: float | None
    lower: float | None
    upper: float | None

    @model_validator(mode="after")
    def interval(self):
        import math

        values = (self.estimate, self.lower, self.upper)
        if any(v is not None and not math.isfinite(v) for v in values):
            raise ValueError("partial_nonfinite")
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError("partial_interval_required")
        if self.estimate is not None and not self.lower <= self.estimate <= self.upper:
            raise ValueError("partial_interval_order")
        if self.coverage == 0 and (self.signal != "indeterminate" or self.estimate is not None):
            raise ValueError("partial_unseen_not_observation")
        return self


class PartialVisionProposal(VisionProposal):
    schema_version: Literal["visual-observation-candidate/2.0.0"] = (
        "visual-observation-candidate/2.0.0"
    )
    partial: PartialObservation

    @model_validator(mode="after")
    def observation_status(self):
        if self.status != "observed" and self.partial.signal != "indeterminate":
            raise ValueError("partial_status_signal")
        if self.status == "observed" and self.partial.coverage == 0:
            raise ValueError("partial_observed_requires_visible_region")
        if len(self.observation_value()) > 1000:
            raise ValueError("partial_observation_length")
        return self

    def observation_value(self):
        from packages.grounding.models import canonical

        return canonical(
            dict(
                schema_version="partial-visual-observation/1.0.0",
                finding=self.finding,
                **self.partial.model_dump(mode="json"),
                coverage_origin="vlm_estimate_not_pixel_measurement",
                unseen_risk=None,
            )
        ).decode()


class ReviewProposal(Value):
    schema_version: Literal["research-review-candidate/1.0.0"] = "research-review-candidate/1.0.0"
    scope_checksum: Checksum
    round: Literal["A", "B", "C"]
    # Import the established scale instead of inventing another O/L/A/e definition.
    from packages.evidence_scoring.review_protocol import ReviewItem as _Item

    items: tuple[_Item, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def unique(self):
        if len({i.assessment_id for i in self.items}) != len(self.items):
            raise ValueError("duplicate_review_item")
        return self


class ReasonedReviewProposal(ReviewProposal):
    from packages.evidence_scoring.review_protocol import ReasonedReviewItem as _ReasonedItem

    schema_version: Literal["research-review-candidate/2.0.0"] = "research-review-candidate/2.0.0"
    items: tuple[_ReasonedItem, ...] = Field(min_length=1, max_length=128)


class ActionView(Value):
    from packages.grounding.query_models import Component as _Component

    id: Token
    kind: Literal["observe", "select_candidate", "request_review", "fallback", "stop"]
    target: Reference
    required_inputs: tuple[Reference, ...]
    components: tuple[_Component, ...]
    expires_at: datetime


class ExpertReviewInput(Value):
    schema_version: Literal["research-expert-input/1.0.0"] = "research-expert-input/1.0.0"
    event: Reference
    proposal_checksum: Checksum
    proposed_action_id: Token
    action_set_checksum: Checksum
    policy_checksum: Checksum
    triggers: tuple[Token, ...]
    candidate: ReviewProposal | SourceProposal | PartialVisionProposal | VisionProposal | None = (
        None
    )


class SimulationExpertInput(Value):
    schema_version: Literal["simulation-expert-input/1.0.0"] = "simulation-expert-input/1.0.0"
    identity: Literal["hypothetical_candidate"] = "hypothetical_candidate"
    policy_checksum: Checksum
    report_checksum: Checksum
    proposed_candidate_id: Token
    declared_candidates: tuple[dict, ...] = Field(min_length=1, max_length=128)
    comparisons: dict
    allowed_domain: Literal["synthetic_offline_no_fact_edits"] = "synthetic_offline_no_fact_edits"


class RolePayload(Value):
    role: Role
    material: Material
    locks: tuple[Reference, ...] | None = None
    set_checksum: Checksum | None = None
    policy_checksum: Checksum | None = None
    actions: tuple[ActionView, ...] | None = Field(default=None, max_length=512)
    review_tasks: ReviewTemplate | None = None
    assessment_ids: tuple[Token, ...] | None = Field(default=None, max_length=128)
    scale: Reference | None = None
    round: Literal["A", "B", "C"] | None = None
    expert_review: ExpertReviewInput | None = None
    simulation_review: SimulationExpertInput | None = None

    @model_validator(mode="after")
    def role_view(self):
        if self.simulation_review is not None and (
            self.role != "expert" or self.expert_review is not None
        ):
            raise ValueError("role_simulation_context_forbidden")
        if self.expert_review is not None and self.role != "expert":
            raise ValueError("role_expert_context_forbidden")
        if self.role == "trace" and any(
            v is None for v in (self.locks, self.set_checksum, self.policy_checksum, self.actions)
        ):
            raise ValueError("trace_payload_required")
        if self.role == "expert" and self.actions is None:
            raise ValueError("expert_actions_required")
        if self.role == "review" and any(
            v is None for v in (self.assessment_ids, self.scale, self.round, self.review_tasks)
        ):
            raise ValueError("review_payload_required")
        if self.role == "review":
            self.review_tasks.check_material(self.material)
            if (
                self.assessment_ids != self.review_tasks.assessment_ids
                or self.scale != self.review_tasks.scale
            ):
                raise ValueError("review_task_inventory")
        if self.role not in {"trace", "expert"} and self.actions is not None:
            raise ValueError("role_actions_forbidden")
        if self.role != "trace" and any(
            v is not None for v in (self.locks, self.set_checksum, self.policy_checksum)
        ):
            raise ValueError("role_locks_forbidden")
        if self.role != "review" and any(
            v is not None for v in (self.assessment_ids, self.scale, self.round, self.review_tasks)
        ):
            raise ValueError("role_reviews_forbidden")
        return self


class RequestSummary(Value):
    schema_version: Literal["agent-request-summary/1.0.0"] = "agent-request-summary/1.0.0"
    run_id: Code
    reservations: int = Field(ge=0)
    completed: int = Field(ge=0)
    pending_unknown: int = Field(ge=0)
    api_requests_observed: int = Field(ge=0)
    api_requests_unknown: int = Field(ge=0)
    simulated_attempts: int = Field(ge=0)
    usage_unknown: int = Field(ge=0)
    clinical_effect: None = None
