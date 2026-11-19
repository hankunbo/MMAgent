"""Shared persisted transition-reset policy; no rule runtime or persistence imports."""

from __future__ import annotations

from dataclasses import dataclass

from packages.contracts import (
    ActorType,
    GraphDelta,
    GraphNodeStatus,
    GraphUpdateSource,
    UpdateNodeStatusOperation,
    VersionedReference,
    can_transition,
)

from .engine import GraphEngineError, GraphEngineErrorCode, canonical_checksum

TRANSITION_STRATEGY_VERSION = "03C_TRANSITION_RESET_V1"
TRANSITION_RESET_MARKER = "RULE:TRANSITION_RESET_V1"
TRANSITION_RESET_REASON = "RULE:TRANSITION_RESET_REQUIRED"
TRANSITION_RESET_OPERATION_PREFIX = "RULE:TRANSITION_RESET_OP_"
_RESET_PREFIX = "RULE:TRANSITION_RESET_"
_DERIVED_STATES = frozenset(
    {
        GraphNodeStatus.SUPPORTED,
        GraphNodeStatus.REFUTED,
        GraphNodeStatus.EXCLUDED,
        GraphNodeStatus.CONFLICTED,
    }
)


@dataclass(frozen=True, slots=True)
class TransitionResetAuditEntry:
    operation_index: int
    node_code: str
    current_status: GraphNodeStatus
    reset_status: GraphNodeStatus
    derived_status: GraphNodeStatus


@dataclass(frozen=True, slots=True)
class TransitionResetAudit:
    strategy_version: str
    reason_code: str
    entries: tuple[TransitionResetAuditEntry, ...]
    delta_checksum: str
    event_checksum: str | None = None


def transition_reset_reference(code: str) -> VersionedReference:
    """Return the checksum-bound reference for one reserved reset reason."""
    return VersionedReference(
        code=code,
        version=TRANSITION_STRATEGY_VERSION,
        checksum=canonical_checksum(
            {
                "strategy_version": TRANSITION_STRATEGY_VERSION,
                "reason_code": code,
            }
        ),
    )


def transition_reset_metadata_from_delta(delta: GraphDelta) -> TransitionResetAudit:
    """Require a complete, necessary reset whose target follows the frozen policy."""
    try:
        if type(delta) is not GraphDelta:
            raise TypeError("reset audit requires a GraphDelta contract")
        parsed = GraphDelta.model_validate(delta.model_dump(mode="python"))
        if parsed.proposed_by is not ActorType.RULE_ENGINE:
            raise ValueError("reset policy belongs to RULE_ENGINE")
        reasons = tuple(
            str(code) for code in parsed.reason_codes if str(code).startswith(_RESET_PREFIX)
        )
        if len(set(reasons)) != len(reasons):
            raise ValueError("reset reasons must not repeat")
        expected_reasons = {TRANSITION_RESET_MARKER, TRANSITION_RESET_REASON}
        entries = []
        for index, operation in enumerate(parsed.operations):
            if type(operation) is not UpdateNodeStatusOperation:
                raise ValueError("reset Delta may contain only status operations")
            if (
                operation.source is not GraphUpdateSource.DETERMINISTIC_RULE
                or operation.to_status is not GraphNodeStatus.PENDING
            ):
                raise ValueError("reset operation must deterministically transition to PENDING")
            prefix = f"{TRANSITION_RESET_OPERATION_PREFIX}{index:03d}_DERIVED_"
            matches = tuple(code for code in reasons if code.startswith(prefix))
            if len(matches) != 1:
                raise ValueError("reset operation requires exactly one final target")
            derived = GraphNodeStatus(matches[0][len(prefix) :])
            if (
                derived not in _DERIVED_STATES
                or operation.from_status is derived
                or can_transition(operation.from_status, derived, allow_idempotent=False)
                or not can_transition(
                    operation.from_status, GraphNodeStatus.PENDING, allow_idempotent=False
                )
                or not can_transition(GraphNodeStatus.PENDING, derived, allow_idempotent=False)
            ):
                raise ValueError("reset does not represent a necessary legal two-step transition")
            expected_reasons.add(matches[0])
            entries.append(
                TransitionResetAuditEntry(
                    index,
                    str(operation.node_code),
                    operation.from_status,
                    operation.to_status,
                    derived,
                )
            )
        if not entries or set(reasons) != expected_reasons:
            raise ValueError("reset strategy, reason, or operation target is missing or unknown")
        reset_refs = tuple(
            ref
            for ref in parsed.provenance.rule_versions
            if str(ref.code).startswith(_RESET_PREFIX)
        )
        if len(reset_refs) != len(expected_reasons):
            raise ValueError("reset references are missing, duplicated, or extra")
        refs = {str(ref.code): ref for ref in reset_refs}
        if refs != {reason: transition_reset_reference(reason) for reason in expected_reasons}:
            raise ValueError("reset reasons lack canonical checksum-bound provenance")
        return TransitionResetAudit(
            TRANSITION_STRATEGY_VERSION,
            TRANSITION_RESET_REASON,
            tuple(entries),
            canonical_checksum(parsed),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise GraphEngineError(
            GraphEngineErrorCode.TRANSITION_AUDIT_INVALID,
            "transition reset metadata is missing, forged, or inconsistent",
        ) from error


def validate_transition_reset_metadata(delta: GraphDelta) -> TransitionResetAudit | None:
    """Enforce reset auditing at every graph write and accepted-event boundary."""
    if (
        any(str(code).startswith(_RESET_PREFIX) for code in delta.reason_codes)
        or any(str(ref.code).startswith(_RESET_PREFIX) for ref in delta.provenance.rule_versions)
        or any(
            type(operation) is UpdateNodeStatusOperation
            and operation.to_status is GraphNodeStatus.PENDING
            and (
                delta.proposed_by is ActorType.RULE_ENGINE
                or operation.source is GraphUpdateSource.DETERMINISTIC_RULE
            )
            for operation in delta.operations
        )
    ):
        return transition_reset_metadata_from_delta(delta)
    return None
