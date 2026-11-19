"""Application-layer construction of atomic 03B graph-version commits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from db.audit.repository import AuditEvent, AuditEventType, GraphVersionAppend
from packages.contracts import ActorType, GECEGInstance
from packages.graph_engine import (
    AcceptedDeltaEvent,
    GraphDeltaApplicationResult,
    GraphDeltaPlan,
    accepted_delta_event_from_payload,
    accepted_delta_event_to_payload,
    apply_graph_delta,
    canonical_checksum,
    graph_structure_hash,
)

GRAPH_VERSION_AUDIT_PAYLOAD_VERSION = "03B_GRAPH_VERSION_AUDIT_V1"


class GraphVersionCommitError(ValueError):
    """Raised before persistence when application and Audit command are not identical."""


@dataclass(frozen=True, slots=True)
class GraphVersionAuditContext:
    event_id: str
    correlation_code: str
    occurred_at: datetime
    causation_code: str | None = None


@dataclass(frozen=True, slots=True)
class GraphVersionCommit:
    application: GraphDeltaApplicationResult
    command: GraphVersionAppend


def _parse_audit_event(command: GraphVersionAppend) -> AcceptedDeltaEvent:
    payload = command.audit_event.payload
    expected_fields = {
        "payload_version",
        "base_graph_version",
        "new_graph_version",
        "idempotency_key",
        "delta_checksum",
        "validation_checksum",
        "result_checksum",
        "structure_hash",
        "plan_checksum",
        "event_checksum",
        "accepted_delta_event",
    }
    if set(payload) != expected_fields:
        raise GraphVersionCommitError("graph-version Audit payload fields are not canonical")
    if (
        type(payload["payload_version"]) is not str
        or payload["payload_version"] != GRAPH_VERSION_AUDIT_PAYLOAD_VERSION
    ):
        raise GraphVersionCommitError("graph-version Audit payload version mismatch")
    for field in ("base_graph_version", "new_graph_version"):
        if type(payload[field]) is not int:
            raise GraphVersionCommitError(f"graph-version Audit {field} must be an integer")
    for field in (
        "idempotency_key",
        "delta_checksum",
        "result_checksum",
        "structure_hash",
        "plan_checksum",
        "event_checksum",
    ):
        if type(payload[field]) is not str:
            raise GraphVersionCommitError(f"graph-version Audit {field} must be a string")
    if (
        payload["validation_checksum"] is not None
        and type(payload["validation_checksum"]) is not str
    ):
        raise GraphVersionCommitError(
            "graph-version Audit validation_checksum must be a string or null"
        )
    event = accepted_delta_event_from_payload(payload["accepted_delta_event"])
    expected_summary = {
        "base_graph_version": event.base_version,
        "new_graph_version": event.result_version,
        "idempotency_key": event.idempotency_key,
        "delta_checksum": event.delta_checksum,
        "validation_checksum": event.validation_checksum,
        "result_checksum": event.result_checksum,
        "structure_hash": event.result_structure_hash,
        "plan_checksum": event.plan_checksum,
        "event_checksum": event.event_checksum,
    }
    if any(
        type(payload[field]) is not type(value) or payload[field] != value
        for field, value in expected_summary.items()
    ):
        raise GraphVersionCommitError("graph-version Audit summary differs from accepted event")
    return event


def validate_graph_version_commit(
    base: GECEGInstance,
    plan: GraphDeltaPlan,
    commit: GraphVersionCommit,
    *,
    accepted_events: tuple[AcceptedDeltaEvent, ...] = (),
) -> GraphVersionCommit:
    """Rebuild application semantics and machine-check every persistence identity."""

    try:
        canonical_application = apply_graph_delta(
            base,
            plan,
            accepted_events=accepted_events,
        )
        command = GraphVersionAppend.model_validate(commit.command.model_dump(mode="python"))
        persisted_event = _parse_audit_event(command)
    except (TypeError, ValueError) as error:
        raise GraphVersionCommitError("graph-version commit failed canonical validation") from error

    application = commit.application
    audit = command.audit_event
    expected_run_id = canonical_application.instance.scope.run_id
    if expected_run_id is None:
        raise GraphVersionCommitError("persisted graph version requires an explicit run scope")
    if application != canonical_application:
        raise GraphVersionCommitError("application differs from deterministic apply result")
    if command.graph != canonical_application.instance:
        raise GraphVersionCommitError("GraphVersionAppend graph differs from application")
    if canonical_checksum(command.graph) != canonical_application.result_checksum:
        raise GraphVersionCommitError("GraphVersionAppend full result checksum mismatch")
    if graph_structure_hash(command.graph) != canonical_application.structure_hash:
        raise GraphVersionCommitError("GraphVersionAppend structure hash mismatch")
    if persisted_event != canonical_application.accepted_event:
        raise GraphVersionCommitError("Audit accepted event differs from application")
    if (
        command.run_id != expected_run_id
        or command.base_graph_version != persisted_event.base_version
        or audit.event_type is not AuditEventType.GRAPH_VERSION_APPENDED
        or audit.actor is not ActorType.GRAPH_ENGINE
        or audit.case_code != canonical_application.instance.case_code
        or audit.run_id != expected_run_id
        or audit.graph_code != persisted_event.graph_code
        or audit.graph_version != persisted_event.result_version
        or audit.idempotency_key != persisted_event.idempotency_key
    ):
        raise GraphVersionCommitError("Audit scope/version/idempotency differs from application")
    return GraphVersionCommit(application=canonical_application, command=command)


def build_graph_version_commit(
    base: GECEGInstance,
    plan: GraphDeltaPlan,
    context: GraphVersionAuditContext,
    *,
    accepted_events: tuple[AcceptedDeltaEvent, ...] = (),
) -> GraphVersionCommit:
    """Build the sole machine-validated GraphDelta application and append command."""

    application = apply_graph_delta(base, plan, accepted_events=accepted_events)
    event = application.accepted_event
    run_id = application.instance.scope.run_id
    if run_id is None:
        raise GraphVersionCommitError("persisted graph version requires an explicit run scope")
    audit_event = AuditEvent(
        event_id=context.event_id,
        event_type=AuditEventType.GRAPH_VERSION_APPENDED,
        actor=ActorType.GRAPH_ENGINE,
        case_code=application.instance.case_code,
        run_id=run_id,
        graph_code=application.instance.graph_code,
        graph_version=application.instance.graph_version,
        correlation_code=context.correlation_code,
        causation_code=context.causation_code,
        idempotency_key=event.idempotency_key,
        occurred_at=context.occurred_at,
        rule_refs=event.delta.provenance.rule_versions,
        payload={
            "payload_version": GRAPH_VERSION_AUDIT_PAYLOAD_VERSION,
            "base_graph_version": event.base_version,
            "new_graph_version": event.result_version,
            "idempotency_key": event.idempotency_key,
            "delta_checksum": event.delta_checksum,
            "validation_checksum": event.validation_checksum,
            "result_checksum": event.result_checksum,
            "structure_hash": event.result_structure_hash,
            "plan_checksum": event.plan_checksum,
            "event_checksum": event.event_checksum,
            "accepted_delta_event": accepted_delta_event_to_payload(event),
        },
    )
    commit = GraphVersionCommit(
        application=application,
        command=GraphVersionAppend(
            run_id=run_id,
            base_graph_version=event.base_version,
            graph=application.instance,
            audit_event=audit_event,
        ),
    )
    return validate_graph_version_commit(
        base,
        plan,
        commit,
        accepted_events=accepted_events,
    )
