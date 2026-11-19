"""Explicit engineering completion, separate from medical RunStatus and truth."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_serializer, model_validator

from packages.grounding.models import Checksum, Token, Value, check_seal


class TaskDefinition(Value):
    schema_version: Literal["research-task-definition/1.0.0"] = "research-task-definition/1.0.0"
    checksum: Checksum
    task: Literal["single_tool_verified_outputs_and_saved_scores"] = (
        "single_tool_verified_outputs_and_saved_scores"
    )
    goal: Token
    use: Literal["synthetic_offline_planning_prior"] = "synthetic_offline_planning_prior"
    profile: Checksum
    run_id: Token
    case_code: Token
    context_checksum: Checksum
    request_checksum: Checksum
    scope_checksum: Checksum
    plan_checksum: Checksum
    config_checksum: Checksum
    execution_policy: Checksum
    score_policy: Checksum
    requirements: dict
    validity_seconds: int = Field(ge=1, le=86400)

    @model_validator(mode="after")
    def integrity(self):
        check_seal(self)
        return self


class TaskCompletion(Value):
    schema_version: Literal["research-task-completion/1.0.0"] = "research-task-completion/1.0.0"
    checksum: Checksum
    definition: TaskDefinition
    lifecycle_checksum: Checksum
    report_checksum: Checksum
    workflow_checksum: Checksum
    score_checksum: Checksum
    output_checksums: tuple[Checksum, ...] = Field(min_length=1)
    valid_from: datetime
    expires_at: datetime
    medical_success: Literal[False] = False
    owner: Literal["ResearchSingleCaseService.read_task_completion"] = (
        "ResearchSingleCaseService.read_task_completion"
    )

    @field_serializer("valid_from", "expires_at")
    def date_wire(self, value):
        return value.isoformat()

    @model_validator(mode="after")
    def integrity(self):
        check_seal(self)
        if (
            not self.valid_from.tzinfo
            or not self.expires_at.tzinfo
            or (self.expires_at - self.valid_from).total_seconds()
            != self.definition.validity_seconds
        ):
            raise ValueError("task_completion_validity")
        return self
