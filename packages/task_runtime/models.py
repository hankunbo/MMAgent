"""12B bounded dispatch envelopes. No commands, credentials or caller supplied plans."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Code, Token, Value
from packages.planner_control.models import Command


class DispatchRequest(Value):
    run_id: Code
    case_code: Code
    context_checksum: Checksum
    idempotency_key: Token
    expected_revision: Annotated[int, Field(ge=0)]


class Profile(Value):
    schema_version: Literal["research-dispatch-profile/1.0.0"] = "research-dispatch-profile/1.0.0"
    synthetic_only: Literal[True] = True
    resource_pool: Literal["synthetic_case"] = "synthetic_case"
    max_concurrency: Literal[1] = 1
    lease_seconds: Annotated[int, Field(ge=1, le=300)] = 30
    max_claims: Annotated[int, Field(ge=1, le=256)] = 64
    max_failures: Annotated[int, Field(ge=1, le=10)] = 3


class Lease(Value):
    worker: Token
    token: Token
    expires_at: datetime


class Receipt(Value):
    schema_version: Literal["research-dispatch/1.0.0"] = "research-dispatch/1.0.0"
    task_id: Code
    task_type: Literal["RESEARCH_DISPATCH"] = "RESEARCH_DISPATCH"
    kind: Literal["CASE_RUN"] = "CASE_RUN"
    request: DispatchRequest
    profile: Profile
    binding_checksum: Checksum
    deadline: datetime
    revision: Annotated[int, Field(ge=0)] = 0
    state: Literal["queued", "leased", "confirming", "completed", "cancelled", "dead_letter"]
    lease: Lease | None = None
    command: Command | None = None
    claims: Annotated[int, Field(ge=0)] = 0
    failures: Annotated[int, Field(ge=0)] = 0
    cancel_requested: bool = False
    control_checksum: Checksum
    reason: Token = "queued"
    clinical_completion: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        if not self.task_id.startswith("TASK:"):
            raise ValueError("dispatch_task_identity")
        if self.deadline.tzinfo is None or (
            self.lease is not None and self.lease.expires_at.tzinfo is None
        ):
            raise ValueError("dispatch_absolute_time")
        if (self.state == "leased") != (self.lease is not None):
            raise ValueError("dispatch_lease_state")
        if self.state == "cancelled" and not self.cancel_requested:
            raise ValueError("dispatch_cancel_monotone")
        return self


TERMINAL = {"completed", "cancelled", "dead_letter"}
