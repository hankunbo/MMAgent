"""Versioned, non-diagnostic input survey; old Vision contracts are unchanged."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.evidence_acquisition.models import TargetedVisionPayload
from packages.grounding.models import Checksum, Token, Value, canonical, digest

from .models import ExpertReviewInput, RolePayload, VisionProposal

Short = Annotated[str, Field(min_length=1, max_length=100)]
CONTRACT = "input-scope-survey/1.0.0"
PROMPT = "vision-input-scope-1.0.0"
STORAGE_PROMPT = "vision-input-scope-1.1.0"
TEXT_PROMPT = "vision-input-scope-1.2.0"


class ScopeTask(Value):
    schema_version: Literal["input-scope-survey/1.0.0"] = CONTRACT
    component_id: Token
    component_checksum: Checksum
    input_checksum: Checksum
    modality: Token
    representation: Literal["image", "image_series", "volume", "map"]
    # These are questions to inspect, not claims that the region is visible.
    requested_regions: tuple[Token, ...] = Field(min_length=1, max_length=4)
    sampling_limits: Short

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.requested_regions)) != len(self.requested_regions):
            raise ValueError("input_scope_duplicate_region")
        return self


class RegionScope(Value):
    region: Token
    visibility: Literal["visible", "partly_visible", "not_identified", "uncertain"]
    coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    clarity: float = Field(ge=0, le=1, allow_inf_nan=False)
    finding: Short
    limitation: Short

    @model_validator(mode="after")
    def consistent(self):
        if self.visibility == "not_identified" and self.coverage != 0:
            raise ValueError("input_scope_unidentified_coverage")
        if self.visibility in {"visible", "partly_visible"} and self.coverage == 0:
            raise ValueError("input_scope_visible_requires_coverage")
        return self


class ScopeVisionProposal(VisionProposal):
    schema_version: Literal["visual-input-scope-candidate/1.0.0"] = (
        "visual-input-scope-candidate/1.0.0"
    )
    regions: tuple[RegionScope, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def survey(self):
        if len({r.region for r in self.regions}) != len(self.regions):
            raise ValueError("input_scope_duplicate_region")
        visible = any(r.visibility in {"visible", "partly_visible"} for r in self.regions)
        if visible and self.status != "observed":
            raise ValueError("input_scope_visible_status")
        if len(self.observation_value()) > 1000:
            raise ValueError("input_scope_original_observation_bound")
        return self

    def observation_value(self):
        return canonical(
            dict(
                schema_version="visual-input-scope-observation/1.0.0",
                finding=self.finding,
                regions=self.regions,
                estimates="model_estimates_not_pixel_measurements",
                physical_scale="not_established_by_visual_survey",
                disease_direction="not_assessed",
            )
        ).decode()


class ScopeVisionPayload(TargetedVisionPayload):
    schema_version: Literal["input-scope-vision-input/1.0.0"] = "input-scope-vision-input/1.0.0"
    scope_task: ScopeTask

    @model_validator(mode="after")
    def input_binding(self):
        if self.scope_task.input_checksum != self.material.scope.input_checksum:
            raise ValueError("input_scope_task_input")
        return self


class ScopeVisionProposalV2(ScopeVisionProposal):
    """Full findings stay in accepted records; the observation is a bound pointer.

    This is a new output contract. It does not grant an old failed candidate
    acceptance, change its original request, or turn a survey into a diagnosis.
    """

    schema_version: Literal["visual-input-scope-candidate/1.1.0"] = (
        "visual-input-scope-candidate/1.1.0"
    )

    def observation_value(self):
        return canonical(
            dict(
                schema_version="visual-input-scope-observation/1.1.0",
                candidate_checksum=digest(self),
                details_pointer="/candidate",
                region_count=len(self.regions),
                status=self.status,
                estimates="model_estimates_not_pixel_measurements",
                physical_scale="not_established_by_visual_survey",
                disease_direction="not_assessed",
            )
        ).decode()


class RegionScopeV2(RegionScope):
    """Bounded full descriptions; visibility and numeric invariants unchanged."""

    finding: Annotated[str, Field(min_length=1, max_length=512)]
    limitation: Annotated[str, Field(min_length=1, max_length=512)]


class ScopeVisionProposalV3(ScopeVisionProposalV2):
    """Explicit local text-capacity interpretation, not an old wire relaxation."""

    schema_version: Literal["visual-input-scope-candidate/1.2.0"] = (
        "visual-input-scope-candidate/1.2.0"
    )
    regions: tuple[RegionScopeV2, ...] = Field(min_length=1, max_length=4)


def proposal_type(prompt):
    types = {
        PROMPT: ScopeVisionProposal,
        STORAGE_PROMPT: ScopeVisionProposalV2,
        TEXT_PROMPT: ScopeVisionProposalV3,
    }
    if prompt not in types:
        raise ValueError("input_scope_prompt")
    return types[prompt]


def check_regions(candidate, task):
    if tuple(r.region for r in candidate.regions) != task.requested_regions:
        raise ValueError("input_scope_region_inventory")


class ScopeExpertReviewInput(ExpertReviewInput):
    schema_version: Literal["input-scope-expert-review/1.0.0"] = "input-scope-expert-review/1.0.0"
    candidate: ScopeVisionProposal


class ScopeExpertPayload(RolePayload):
    schema_version: Literal["input-scope-expert-input/1.0.0"] = "input-scope-expert-input/1.0.0"
    expert_review: ScopeExpertReviewInput


class ScopeExpertReviewInputV2(ScopeExpertReviewInput):
    schema_version: Literal["input-scope-expert-review/1.1.0"] = "input-scope-expert-review/1.1.0"
    candidate: ScopeVisionProposalV2


class ScopeExpertPayloadV2(ScopeExpertPayload):
    schema_version: Literal["input-scope-expert-input/1.1.0"] = "input-scope-expert-input/1.1.0"
    expert_review: ScopeExpertReviewInputV2


class ScopeExpertReviewInputV3(ScopeExpertReviewInputV2):
    schema_version: Literal["input-scope-expert-review/1.2.0"] = "input-scope-expert-review/1.2.0"
    candidate: ScopeVisionProposalV3


class ScopeExpertPayloadV3(ScopeExpertPayloadV2):
    schema_version: Literal["input-scope-expert-input/1.2.0"] = "input-scope-expert-input/1.2.0"
    expert_review: ScopeExpertReviewInputV3
