"""Evidence gaps and executable preparation, distinct from medical conclusions."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value
from packages.llm_provider.models import ReviewTask, RolePayload
from packages.planner_control.models import Reference

Purpose = Literal["disease_relation", "measurement_dependency", "quality", "context"]

Action = Literal[
    "bind_existing_reference",
    "search_knowledge",
    "read_method_provenance",
    "reuse_observation",
    "targeted_vision",
    "review",
    "request_tool_approval",
]


class Requirement(Value):
    assessment_id: Token
    purpose: Purpose
    acquisition: Literal["existing", "image", "external_exam", "tool_gap"]
    required_inputs: tuple[Token, ...] = Field(min_length=1, max_length=12)
    unavailable_inputs: tuple[Token, ...] = Field(default=(), max_length=12)
    missing_source_ids: tuple[Token, ...] = Field(default=(), max_length=16)
    public_topics: tuple[Token, ...] = Field(default=(), max_length=4)
    method_details_required: bool = False
    priority: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="after")
    def consistent(self):
        from packages.knowledge_search.broad import TOPICS

        if not set(self.unavailable_inputs) <= set(self.required_inputs):
            raise ValueError("acquisition_input_inventory")
        if any(t not in TOPICS for t in self.public_topics):
            raise ValueError("acquisition_public_topic")
        if self.unavailable_inputs and self.acquisition != "external_exam":
            raise ValueError("acquisition_unavailable_requires_external_input")
        for values in (self.required_inputs, self.missing_source_ids, self.public_topics):
            if len(values) != len(set(values)):
                raise ValueError("acquisition_duplicate_requirement")
        return self


class GapCard(Value):
    assessment_id: Token
    node: Token
    purpose: Purpose
    observation: Reference
    observation_status: Literal["obtained", "missing", "ungradable", "unknown"]
    interpretation_status: Literal["installed", "missing", "not_independent_disease_rule"]
    blockers: tuple[Token, ...]
    missing_inputs: tuple[Token, ...]
    missing_source_ids: tuple[Token, ...]
    next_actions: tuple[Action, ...]
    public_topics: tuple[Token, ...]
    priority: int
    disposition: Literal[
        "ready_for_review",
        "needs_acquisition",
        "current_input_unavailable",
        "auxiliary_observed",
        "quality_limited",
    ]


class AcquisitionPlan(Value):
    schema_version: Literal["research-acquisition-plan/1.0.0"] = "research-acquisition-plan/1.0.0"
    state_checksum: Checksum
    requirements_checksum: Checksum
    cards: tuple[GapCard, ...]
    # These plans neither create runs nor grant external execution/medical acceptance.
    execution_authorized: Literal[False] = False
    clinical_acceptance: Literal[False] = False


class SearchProgress(Value):
    """Owner-supplied confirmed outcomes; unknown requests are never auto-retried."""

    rounds: int = Field(default=0, ge=0, le=2)
    no_progress: int = Field(default=0, ge=0, le=2)
    pending_unknown: bool = False
    cancelled: bool = False
    deadline_reached: bool = False


class SearchSuggestion(Value):
    schema_version: Literal["acquisition-search-suggestion/1.0.0"] = (
        "acquisition-search-suggestion/1.0.0"
    )
    assessment_id: Token
    topics: tuple[Token, ...] = Field(min_length=1, max_length=4)
    # No case narrative, measurement, URL, free query or credentials cross this boundary.
    purpose: Literal["association", "measurement_method", "limitations"]


class ActionHistory(Value):
    assessment_id: Token
    action: Action
    input_checksum: Checksum
    outcome: Literal["confirmed", "failed", "unknown"]


class TargetedVisionPayload(RolePayload):
    schema_version: Literal["targeted-vision-input/1.0.0"] = "targeted-vision-input/1.0.0"
    role: Literal["vision"] = "vision"
    vision_task: ReviewTask

    @model_validator(mode="after")
    def exact_target(self):
        images = [o for o in self.material.observations if o.image is not None]
        if len(images) != 1 or self.vision_task.observation != images[0].reference:
            raise ValueError("targeted_vision_observation")
        if self.vision_task.goal != self.material.goal_code or not set(
            self.vision_task.allowed_sources
        ) <= {s.reference for s in self.material.sources}:
            raise ValueError("targeted_vision_source_or_goal")
        if (self.vision_task.method, self.vision_task.unit) != (images[0].method, images[0].unit):
            raise ValueError("targeted_vision_method_unit")
        return self
