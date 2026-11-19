"""Versioned operation references, not another state machine or authorization flag."""

from typing import Literal

from pydantic import Field, model_validator

from agents.expert_agent.models import CommittedReviewReadback
from packages.grounding.models import Checksum, Value, digest

Stage = Literal[
    "sources", "definitions", "requirements", "acquisition", "review", "acceptance", "export"
]


class StagePlan(Value):
    schema_version: Literal["multimodal-stage-plan/1.0.0"] = "multimodal-stage-plan/1.0.0"
    protocol_checksum: str
    implementation: dict[str, str]
    stages: tuple[Stage, ...] = Field(min_length=7, max_length=7)
    deadline: str
    max_new_requests: int = Field(ge=1, le=40)
    budget_owner: Literal["original_campaign_and_RuntimeAgentJournal"]
    acceptance_is_inside_review: Literal[True]


class StageReceipt(Value):
    schema_version: Literal["multimodal-stage-receipt/1.0.0"] = "multimodal-stage-receipt/1.0.0"
    binding: str
    stage: Stage
    run_id: str
    payload: dict


class LocalReviewClosure(Value):
    schema_version: Literal["committed-review-local-closure/1.0.0"] = (
        "committed-review-local-closure/1.0.0"
    )
    original_phase: dict | None
    original_phase_checksum: Checksum
    confirmation: CommittedReviewReadback
    new_requests: Literal[0] = 0
    historical_failure_preserved: Literal[True] = True

    @model_validator(mode="after")
    def binding(self):
        if digest(self.original_phase) != self.original_phase_checksum or (
            self.original_phase is not None
            and (
                self.original_phase.get("run_id") != self.confirmation.run_id
                or self.original_phase.get("target") is not None
            )
        ):
            raise ValueError("local_review_closure_binding")
        return self
