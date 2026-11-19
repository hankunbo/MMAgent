"""Explicit, conservative migration of legacy design vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .enums import (
    EvidenceValidity,
    GraphNodeStatus,
    PlanAction,
    RunStatus,
    ToolCatalogStatus,
    ToolValidationStatus,
)


class IncompatibleLegacyValue(ValueError):
    """Raised when an old term cannot be safely mapped one-to-one."""


@dataclass(frozen=True, slots=True)
class LegacyNodeStateExpansion:
    graph_node_status: GraphNodeStatus
    evidence_validity: EvidenceValidity | None = None
    note: str = ""


_ALIASES: dict[type[StrEnum], dict[str, StrEnum]] = {
    GraphNodeStatus: {
        "Unknown": GraphNodeStatus.UNKNOWN,
        "Pending": GraphNodeStatus.PENDING,
        "Supported": GraphNodeStatus.SUPPORTED,
        "Refuted": GraphNodeStatus.REFUTED,
        "Conflicted": GraphNodeStatus.CONFLICTED,
        "Excluded": GraphNodeStatus.EXCLUDED,
    },
    RunStatus: {
        "WAITING_DATA": RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
        "COMPLETED_UNCERTAIN": RunStatus.COMPLETED_UNCERTAIN,
    },
    PlanAction: {
        "HUMAN_REVIEW": PlanAction.EXPERT_ASSESSMENT,
    },
    ToolCatalogStatus: {
        "Candidate": ToolCatalogStatus.CANDIDATE,
        "Registered": ToolCatalogStatus.REGISTERED,
        "Deprecated": ToolCatalogStatus.DEPRECATED,
        "Failed": ToolCatalogStatus.FAILED,
    },
    ToolValidationStatus: {
        "Provisional": ToolValidationStatus.SMOKE_TESTED,
        "Validated": ToolValidationStatus.SMOKE_TESTED,
        "validated": ToolValidationStatus.SMOKE_TESTED,
        "provisional": ToolValidationStatus.SMOKE_TESTED,
    },
}

_UNSAFE_ONLINE_TERMS = frozenset(
    {"WAITING_FOR_HUMAN", "HUMAN_REVIEW", "HumanReview", "WAITING_HUMAN"}
)


def parse_compatible(enum_type: type[StrEnum], value: str) -> StrEnum:
    """Parse a canonical code or a documented legacy alias.

    The ordinary compatibility path conservatively maps old ``Validated`` to
    ``smoke_tested`` because it has no complete historical gate evidence. It never
    revives an online human-wait state.
    """

    if not isinstance(value, str):
        raise TypeError("serialized enum code must be a string")
    if enum_type is RunStatus and value in _UNSAFE_ONLINE_TERMS:
        raise IncompatibleLegacyValue(
            f"{value} cannot be migrated into an online run; use safe stop and offline review"
        )
    try:
        return enum_type(value)
    except ValueError:
        try:
            return _ALIASES[enum_type][value]
        except KeyError as exc:
            raise IncompatibleLegacyValue(
                f"no safe {enum_type.__name__} compatibility mapping for {value!r}"
            ) from exc


def expand_legacy_node_state(value: str) -> LegacyNodeStateExpansion:
    """Split ambiguous old node states into their new semantic owners."""

    if value == "Invalid":
        return LegacyNodeStateExpansion(
            graph_node_status=GraphNodeStatus.UNKNOWN,
            evidence_validity=EvidenceValidity.INVALID,
            note="legacy Invalid described evidence validity, not a clinical proposition outcome",
        )
    if value == "HumanReview":
        return LegacyNodeStateExpansion(
            graph_node_status=GraphNodeStatus.INSUFFICIENT_EVIDENCE,
            note="online review is EXPERT_ASSESSMENT; any human review is an offline task",
        )
    parsed = parse_compatible(GraphNodeStatus, value)
    return LegacyNodeStateExpansion(graph_node_status=GraphNodeStatus(parsed))
