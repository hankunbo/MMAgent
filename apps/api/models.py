"""HTTP envelopes, separate from the frozen domain contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from packages.contracts import RunStatus
from packages.grounding.models import Checksum, Code, Token

API_VERSION = "research-http/1.0.0"


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Write(Value):
    api_version: Literal["research-http/1.0.0"]
    idempotency_key: Token


class CaseImport(Write):
    source_code: Code
    dataset_code: Code
    expected_version: Checksum


class ArtifactImport(Write):
    source_code: Code
    expected_version: Checksum


class RequestReference(Value):
    code: Token
    version: Token
    checksum: Checksum


class RunCreate(Write):
    case_code: Code
    run_id: Code
    request: RequestReference
    expected_version: Checksum
    mode: Literal["create_only", "submit"] = "create_only"


class ControlWrite(Write):
    case_code: Code
    context_checksum: Checksum
    expected_revision: Annotated[int, Field(ge=0, le=10000, strict=True)]


class OfflineAppend(Write):
    case_code: Code
    expected_version: Checksum
    finding: Literal["review_requested", "review_observed", "followup_required"]


class ExpertRequest(Write):
    case_code: Code
    expected_version: Checksum
    contract_version: Literal["research-expert-request/1.0.0", "ExpertAssessment/1.0"]


class OfflineVote(Write):
    case_code: Code
    expected_version: Checksum
    decision: Literal["approve", "reject", "needs_more_information"]


class ExtensionConsume(Write):
    case_code: Code
    expected_version: Checksum


class Envelope(Value):
    api_version: Literal["research-http/1.0.0"] = API_VERSION
    trace_id: str
    semantics: Literal["synchronous_result", "historical_read", "unconfirmed"]
    contract_version: str
    data: JsonValue
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )


class ErrorDetail(Value):
    error_code: str
    message: str
    retry: Literal[
        "correct_request",
        "authenticate",
        "request_scope",
        "reread_version",
        "install_owner",
        "restore_qualification",
        "later_same_key",
        "reconcile_same_key",
        "none",
    ]
    automatic_retry: Literal[False] = False
    commit_state: Literal["not_attempted", "unconfirmed", "not_applicable"]


class ErrorEnvelope(Value):
    api_version: Literal["research-http/1.0.0"] = API_VERSION
    trace_id: str
    error: ErrorDetail


class RunView(Value):
    run_id: str
    case_code: str
    status: RunStatus
    context_version: Literal["research-run/1.0.0"]
    context_checksum: Checksum
    synthetic: bool
    created: Literal[True] = True
    dispatch: Literal["not_installed"] = "not_installed"
    main_run_waits_for_extensions: Literal[False] = False
    extensions: JsonValue
