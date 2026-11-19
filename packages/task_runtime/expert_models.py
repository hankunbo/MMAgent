"""Versioned platform envelope; frozen domain contracts are unchanged."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from packages.grounding.models import Checksum, Code, Token


class ExpertDispatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    api_version: Literal["research-http/1.0.0"]
    idempotency_key: Token
    case_code: Code
    context_checksum: Checksum
    expected_revision: Annotated[int, Field(ge=0, le=10000, strict=True)]
    contract_version: Literal["research-expert-request/2.0.0"]
    expected_version: Checksum
    proposed_action_id: Token
    role: Literal["source", "vision", "review"] | None = None
    abc: bool = False
