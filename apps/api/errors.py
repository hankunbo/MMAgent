"""Allowlisted error translation: never return exception text or validation input."""

import psycopg
from pydantic import ValidationError

from db.audit.repository import AuditIdempotencyConflictError, ConcurrencyConflictError
from db.runtime.repository.ports import IdempotencyConflictError, NotFoundError
from packages.evidence_details.current_rejection import CurrentResearchRejected

# status, retry policy, anonymous public message
MATRIX = {
    "REQUEST_INVALID": (422, "correct_request", "Request does not match the HTTP contract."),
    "AUTHENTICATION_REQUIRED": (401, "authenticate", "A locally installed identity is required."),
    "SCOPE_FORBIDDEN": (403, "request_scope", "This identity cannot access this scope or action."),
    "OBJECT_NOT_FOUND": (404, "none", "The requested object does not exist."),
    "IDEMPOTENCY_CONFLICT": (409, "none", "The key is already bound to different content."),
    "VERSION_CONFLICT": (409, "reread_version", "The expected version does not match."),
    "STATE_CONFLICT": (409, "reread_version", "The action is unavailable in the current state."),
    "DATA_NOT_READY": (
        409,
        "none",
        "Required data is not available; no negative finding is implied.",
    ),
    "CAPABILITY_NOT_READY": (503, "install_owner", "The required owner is not installed."),
    "QUALIFICATION_WITHDRAWN": (
        409,
        "restore_qualification",
        "Current owner qualification was refused.",
    ),
    "BUDGET_EXHAUSTED": (429, "none", "The original execution budget is exhausted."),
    "OWNER_TIMEOUT": (504, "later_same_key", "The owner did not return within its bound."),
    "OWNER_FAILURE": (503, "later_same_key", "The owner could not complete the operation."),
    "COMMIT_UNKNOWN": (503, "reconcile_same_key", "Commit is unconfirmed; reconcile the same key."),
    "REQUEST_TOO_LARGE": (413, "correct_request", "The baseline request size limit was exceeded."),
    "MEDIA_TYPE_UNSUPPORTED": (415, "correct_request", "The request media type is unsupported."),
    "VERSION_UNSUPPORTED": (422, "correct_request", "This contract version is not supported."),
    "METHOD_NOT_ALLOWED": (405, "correct_request", "The HTTP method is not supported."),
}


class Fault(Exception):
    def __init__(self, code, *, commit_state="not_attempted"):
        self.code, self.commit_state = code, commit_state
        super().__init__(code)


CONFLICT = {
    "expert_key_conflict",
    "dispatch_key_conflict",
    "step_key_content_conflict",
    "research_run_context_conflict",
    "dataset_input_conflict",
    "extension_stage_identity_conflict",
    "agent_idempotency_conflict",
    "research_key_content_conflict",
}
VERSION = {
    "expert_version_changed",
    "dispatch_control_changed",
    "dispatch_binding_changed",
    "control_revision_conflict",
    "planner_binding_conflict",
    "run_context_changed",
    "research_registration_identity",
    "research_registration_mismatch",
}
STATE = {
    "expert_control_state",
    "illegal_state_action",
    "pending_step_requires_same_command",
    "planner_busy",
    "dispatch_busy",
    "dispatch_terminal",
    "dispatch_lease_invalid",
    "dispatch_submit_state",
}
QUALIFICATION = {
    "extension_authority_missing_or_expired",
    "extension_authorization_missing",
    "extension_grant_unavailable",
    "source_withdrawn",
    "synthetic_catalog_unavailable",
    "extension_current_grant_required",
    "extension_grant_expired",
    "extension_catalog_implementation_changed",
}


def translate(error, *, write=False):
    if isinstance(error, Fault):
        return error
    if isinstance(error, NotFoundError):
        return Fault("OBJECT_NOT_FOUND")
    if isinstance(
        error,
        (IdempotencyConflictError, AuditIdempotencyConflictError, psycopg.errors.UniqueViolation),
    ):
        return Fault("IDEMPOTENCY_CONFLICT")
    if isinstance(error, (ConcurrencyConflictError, psycopg.errors.SerializationFailure)):
        return Fault("VERSION_CONFLICT")
    if isinstance(error, CurrentResearchRejected):
        return Fault("QUALIFICATION_WITHDRAWN")
    # Only known, exact owner codes are translated. Arbitrary errors stay technical.
    reason = str(error) if isinstance(error, ValueError) else ""
    if isinstance(error, psycopg.errors.CheckViolation):
        reason = error.diag.message_primary
    for codes, target in (
        (CONFLICT, "IDEMPOTENCY_CONFLICT"),
        (VERSION, "VERSION_CONFLICT"),
        (STATE, "STATE_CONFLICT"),
        (QUALIFICATION, "QUALIFICATION_WITHDRAWN"),
    ):
        if reason in codes:
            return Fault(target)
    if reason in {
        "formal_run_reader_not_installed",
        "dispatch_owner_required",
        "expert_owner_required",
    }:
        return Fault("CAPABILITY_NOT_READY")
    if reason in {
        "dispatch_scope",
        "dispatch_synthetic_only",
        "expert_scope",
        "expert_synthetic_only",
        "expert_action_scope",
    }:
        return Fault("SCOPE_FORBIDDEN")
    if reason == "dispatch_missing":
        return Fault("OBJECT_NOT_FOUND")
    if reason in {"path_graph_report_missing", "gap_application_missing"}:
        return Fault("OBJECT_NOT_FOUND")
    if reason in {"extension_source_budget", "request_budget_exhausted"}:
        return Fault("BUDGET_EXHAUSTED")
    if str(error) == "dataset_commit_unknown_retry_same_key":
        return Fault("COMMIT_UNKNOWN", commit_state="unconfirmed")
    if isinstance(error, (ConnectionError, psycopg.OperationalError, psycopg.InterfaceError)) or (
        write and isinstance(error, (OSError, RuntimeError))
    ):
        return Fault(
            "COMMIT_UNKNOWN" if write else "OWNER_FAILURE",
            commit_state="unconfirmed" if write else "not_applicable",
        )
    if isinstance(error, TimeoutError):
        return Fault(
            "COMMIT_UNKNOWN" if write else "OWNER_TIMEOUT",
            commit_state="unconfirmed" if write else "not_applicable",
        )
    if isinstance(error, (ValidationError, OSError, ValueError, RuntimeError, psycopg.Error)):
        return Fault("OWNER_FAILURE", commit_state="not_applicable")
    return Fault("OWNER_FAILURE", commit_state="not_applicable")
