"""Deterministic state-machine ownership and transition rules for 01A."""

from __future__ import annotations

from enum import StrEnum

from .enums import (
    ExpertAssessmentStatus,
    GraphNodeStatus,
    OfflineReviewStatus,
    QCStatus,
    RunStatus,
    StepStatus,
    ToolCatalogStatus,
    ToolExecutionStatus,
    ToolExtensionJobStatus,
    ToolValidationStatus,
)


class StateTransitionError(ValueError):
    """Raised for a cross-machine or illegal transition."""


class StateMachine(StrEnum):
    GRAPH_NODE = "GRAPH_NODE"
    RUN = "RUN"
    STEP = "STEP"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    QC = "QC"
    TOOL_CATALOG = "TOOL_CATALOG"
    TOOL_VALIDATION = "TOOL_VALIDATION"
    ONLINE_EXPERT_ASSESSMENT = "ONLINE_EXPERT_ASSESSMENT"
    OFFLINE_REVIEW = "OFFLINE_REVIEW"
    TOOL_EXTENSION_JOB = "TOOL_EXTENSION_JOB"


_MACHINE_BY_TYPE: dict[type[StrEnum], StateMachine] = {
    GraphNodeStatus: StateMachine.GRAPH_NODE,
    RunStatus: StateMachine.RUN,
    StepStatus: StateMachine.STEP,
    ToolExecutionStatus: StateMachine.TOOL_EXECUTION,
    QCStatus: StateMachine.QC,
    ToolCatalogStatus: StateMachine.TOOL_CATALOG,
    ToolValidationStatus: StateMachine.TOOL_VALIDATION,
    ExpertAssessmentStatus: StateMachine.ONLINE_EXPERT_ASSESSMENT,
    OfflineReviewStatus: StateMachine.OFFLINE_REVIEW,
    ToolExtensionJobStatus: StateMachine.TOOL_EXTENSION_JOB,
}


def _terminal_transitions(*states: StrEnum) -> dict[StrEnum, frozenset[StrEnum]]:
    return {state: frozenset() for state in states}


_TRANSITIONS: dict[type[StrEnum], dict[StrEnum, frozenset[StrEnum]]] = {
    GraphNodeStatus: {
        GraphNodeStatus.UNKNOWN: frozenset(
            {GraphNodeStatus.PENDING, GraphNodeStatus.INSUFFICIENT_EVIDENCE}
        ),
        GraphNodeStatus.PENDING: frozenset(
            {
                GraphNodeStatus.UNKNOWN,
                GraphNodeStatus.SUPPORTED,
                GraphNodeStatus.REFUTED,
                GraphNodeStatus.CONFLICTED,
                GraphNodeStatus.EXCLUDED,
                GraphNodeStatus.INSUFFICIENT_EVIDENCE,
            }
        ),
        GraphNodeStatus.SUPPORTED: frozenset(
            {GraphNodeStatus.REFUTED, GraphNodeStatus.CONFLICTED, GraphNodeStatus.PENDING}
        ),
        GraphNodeStatus.REFUTED: frozenset(
            {GraphNodeStatus.SUPPORTED, GraphNodeStatus.CONFLICTED, GraphNodeStatus.PENDING}
        ),
        GraphNodeStatus.CONFLICTED: frozenset(
            {GraphNodeStatus.SUPPORTED, GraphNodeStatus.REFUTED, GraphNodeStatus.PENDING}
        ),
        GraphNodeStatus.EXCLUDED: frozenset({GraphNodeStatus.PENDING}),
        GraphNodeStatus.INSUFFICIENT_EVIDENCE: frozenset({GraphNodeStatus.PENDING}),
    },
    RunStatus: {
        RunStatus.CREATED: frozenset({RunStatus.NORMALIZING, RunStatus.FAILED_TECHNICAL}),
        RunStatus.NORMALIZING: frozenset({RunStatus.PLANNING, RunStatus.FAILED_TECHNICAL}),
        RunStatus.PLANNING: frozenset(
            {
                RunStatus.EXECUTING,
                RunStatus.VERIFYING,
                RunStatus.REPLANNING,
                RunStatus.COMPLETED_UNCERTAIN,
                RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
                RunStatus.FAILED_TECHNICAL,
                RunStatus.STOPPED_UNSAFE,
            }
        ),
        RunStatus.EXECUTING: frozenset(
            {
                RunStatus.VERIFYING,
                RunStatus.REPLANNING,
                RunStatus.FAILED_TECHNICAL,
                RunStatus.STOPPED_UNSAFE,
            }
        ),
        RunStatus.VERIFYING: frozenset(
            {
                RunStatus.REPLANNING,
                RunStatus.COMPLETED_SUPPORTED,
                RunStatus.COMPLETED_REFUTED,
                RunStatus.COMPLETED_UNCERTAIN,
                RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
                RunStatus.FAILED_TECHNICAL,
                RunStatus.STOPPED_UNSAFE,
            }
        ),
        RunStatus.REPLANNING: frozenset(
            {
                RunStatus.PLANNING,
                RunStatus.EXECUTING,
                RunStatus.VERIFYING,
                RunStatus.COMPLETED_UNCERTAIN,
                RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
                RunStatus.FAILED_TECHNICAL,
                RunStatus.STOPPED_UNSAFE,
            }
        ),
        **_terminal_transitions(
            RunStatus.COMPLETED_SUPPORTED,
            RunStatus.COMPLETED_REFUTED,
            RunStatus.COMPLETED_UNCERTAIN,
            RunStatus.COMPLETED_INSUFFICIENT_EVIDENCE,
            RunStatus.FAILED_TECHNICAL,
            RunStatus.STOPPED_UNSAFE,
        ),
    },
    StepStatus: {
        StepStatus.CREATED: frozenset({StepStatus.READY, StepStatus.CANCELLED}),
        StepStatus.READY: frozenset({StepStatus.RUNNING, StepStatus.SKIPPED, StepStatus.CANCELLED}),
        StepStatus.RUNNING: frozenset(
            {
                StepStatus.SUCCEEDED,
                StepStatus.FAILED,
                StepStatus.TIMED_OUT,
                StepStatus.CANCELLED,
            }
        ),
        StepStatus.FAILED: frozenset({StepStatus.RETRY_SCHEDULED}),
        StepStatus.TIMED_OUT: frozenset({StepStatus.RETRY_SCHEDULED}),
        StepStatus.RETRY_SCHEDULED: frozenset({StepStatus.READY, StepStatus.CANCELLED}),
        **_terminal_transitions(
            StepStatus.SUCCEEDED,
            StepStatus.SKIPPED,
            StepStatus.CANCELLED,
        ),
    },
    ToolExecutionStatus: {
        ToolExecutionStatus.REQUESTED: frozenset(
            {
                ToolExecutionStatus.DRY_RUN_VALIDATED,
                ToolExecutionStatus.BLOCKED,
                ToolExecutionStatus.CANCELLED,
            }
        ),
        ToolExecutionStatus.DRY_RUN_VALIDATED: frozenset(
            {ToolExecutionStatus.QUEUED, ToolExecutionStatus.CANCELLED}
        ),
        ToolExecutionStatus.QUEUED: frozenset(
            {ToolExecutionStatus.RUNNING, ToolExecutionStatus.CANCELLED}
        ),
        ToolExecutionStatus.RUNNING: frozenset(
            {
                ToolExecutionStatus.SUCCEEDED,
                ToolExecutionStatus.FAILED,
                ToolExecutionStatus.TIMED_OUT,
                ToolExecutionStatus.CANCELLED,
            }
        ),
        **_terminal_transitions(
            ToolExecutionStatus.SUCCEEDED,
            ToolExecutionStatus.FAILED,
            ToolExecutionStatus.TIMED_OUT,
            ToolExecutionStatus.CANCELLED,
            ToolExecutionStatus.BLOCKED,
        ),
    },
    QCStatus: {
        QCStatus.UNASSESSED: frozenset({QCStatus.PENDING, QCStatus.PASS, QCStatus.FAIL}),
        QCStatus.PENDING: frozenset({QCStatus.PASS, QCStatus.FAIL}),
        **_terminal_transitions(QCStatus.PASS, QCStatus.FAIL),
    },
    ToolCatalogStatus: {
        ToolCatalogStatus.CANDIDATE: frozenset(
            {
                ToolCatalogStatus.REGISTERED,
                ToolCatalogStatus.QUARANTINED,
                ToolCatalogStatus.FAILED,
            }
        ),
        ToolCatalogStatus.REGISTERED: frozenset(
            {
                ToolCatalogStatus.ACTIVE,
                ToolCatalogStatus.QUARANTINED,
                ToolCatalogStatus.FAILED,
                ToolCatalogStatus.DEPRECATED,
            }
        ),
        ToolCatalogStatus.ACTIVE: frozenset(
            {
                ToolCatalogStatus.QUARANTINED,
                ToolCatalogStatus.FAILED,
                ToolCatalogStatus.DEPRECATED,
            }
        ),
        **_terminal_transitions(
            ToolCatalogStatus.QUARANTINED,
            ToolCatalogStatus.FAILED,
            ToolCatalogStatus.DEPRECATED,
        ),
    },
    ToolValidationStatus: {
        ToolValidationStatus.UNVALIDATED: frozenset({ToolValidationStatus.SMOKE_TESTED}),
        ToolValidationStatus.SMOKE_TESTED: frozenset({ToolValidationStatus.BENCHMARK_PASSED}),
        ToolValidationStatus.BENCHMARK_PASSED: frozenset(
            {ToolValidationStatus.AUTO_VALIDATED_RESEARCH}
        ),
        ToolValidationStatus.AUTO_VALIDATED_RESEARCH: frozenset(),
    },
    ExpertAssessmentStatus: {
        ExpertAssessmentStatus.REQUESTED: frozenset(
            {
                ExpertAssessmentStatus.RUNNING,
                ExpertAssessmentStatus.FAILED,
                ExpertAssessmentStatus.TIMED_OUT,
            }
        ),
        ExpertAssessmentStatus.RUNNING: frozenset(
            {
                ExpertAssessmentStatus.COMPLETED,
                ExpertAssessmentStatus.FAILED,
                ExpertAssessmentStatus.INVALID_OUTPUT,
                ExpertAssessmentStatus.TIMED_OUT,
            }
        ),
        **_terminal_transitions(
            ExpertAssessmentStatus.COMPLETED,
            ExpertAssessmentStatus.FAILED,
            ExpertAssessmentStatus.INVALID_OUTPUT,
            ExpertAssessmentStatus.TIMED_OUT,
        ),
    },
    OfflineReviewStatus: {
        OfflineReviewStatus.OFFLINE_REVIEW_PENDING: frozenset(
            {OfflineReviewStatus.IN_REVIEW, OfflineReviewStatus.WITHDRAWN}
        ),
        OfflineReviewStatus.IN_REVIEW: frozenset(
            {
                OfflineReviewStatus.APPROVED,
                OfflineReviewStatus.REJECTED,
                OfflineReviewStatus.WITHDRAWN,
            }
        ),
        **_terminal_transitions(
            OfflineReviewStatus.APPROVED,
            OfflineReviewStatus.REJECTED,
            OfflineReviewStatus.WITHDRAWN,
        ),
    },
    ToolExtensionJobStatus: {
        ToolExtensionJobStatus.REQUESTED: frozenset({ToolExtensionJobStatus.DISCOVERING}),
        ToolExtensionJobStatus.DISCOVERING: frozenset(
            {
                ToolExtensionJobStatus.CHECKING_ELIGIBILITY,
                ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
                ToolExtensionJobStatus.FAILED,
            }
        ),
        ToolExtensionJobStatus.CHECKING_ELIGIBILITY: frozenset(
            {
                ToolExtensionJobStatus.STAGING,
                ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
                ToolExtensionJobStatus.FAILED,
            }
        ),
        ToolExtensionJobStatus.STAGING: frozenset(
            {ToolExtensionJobStatus.ADAPTING, ToolExtensionJobStatus.FAILED}
        ),
        ToolExtensionJobStatus.ADAPTING: frozenset(
            {ToolExtensionJobStatus.SANDBOX_TESTING, ToolExtensionJobStatus.FAILED}
        ),
        ToolExtensionJobStatus.SANDBOX_TESTING: frozenset(
            {ToolExtensionJobStatus.SMOKE_TESTING, ToolExtensionJobStatus.FAILED}
        ),
        ToolExtensionJobStatus.SMOKE_TESTING: frozenset(
            {
                ToolExtensionJobStatus.BENCHMARKING,
                ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
                ToolExtensionJobStatus.FAILED,
            }
        ),
        ToolExtensionJobStatus.BENCHMARKING: frozenset(
            {
                ToolExtensionJobStatus.POLICY_GATE,
                ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
                ToolExtensionJobStatus.FAILED,
            }
        ),
        ToolExtensionJobStatus.POLICY_GATE: frozenset(
            {
                ToolExtensionJobStatus.ACTIVATED_NEXT_RUN,
                ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
                ToolExtensionJobStatus.FAILED,
            }
        ),
        **_terminal_transitions(
            ToolExtensionJobStatus.ACTIVATED_NEXT_RUN,
            ToolExtensionJobStatus.REQUIREMENT_DOCUMENTED,
            ToolExtensionJobStatus.FAILED,
        ),
    },
}


def state_machine_of(value: StrEnum) -> StateMachine:
    """Return the sole state machine that owns an enum value."""

    if not isinstance(value, StrEnum):
        raise TypeError("state value must be a canonical StrEnum member, not an arbitrary string")
    try:
        return _MACHINE_BY_TYPE[type(value)]
    except KeyError as exc:
        raise TypeError(f"{type(value).__name__} is not a transition state") from exc


def can_transition(current: StrEnum, target: StrEnum, *, allow_idempotent: bool = True) -> bool:
    """Return whether a same-machine transition is legal."""

    current_machine = state_machine_of(current)
    target_machine = state_machine_of(target)
    if type(current) is not type(target):
        raise StateTransitionError(
            f"cannot mix {current_machine.value} with {target_machine.value} states"
        )
    if current == target:
        return allow_idempotent
    return target in _TRANSITIONS[type(current)][current]


def require_transition(current: StrEnum, target: StrEnum) -> StrEnum:
    """Validate and return a target state, including idempotent retries."""

    if not can_transition(current, target):
        raise StateTransitionError(
            f"illegal {state_machine_of(current).value} transition: {current} -> {target}"
        )
    return target


def require_graph_node_status(value: object) -> GraphNodeStatus:
    """Prevent run/step/extension states from being assigned to a graph node."""

    if not isinstance(value, GraphNodeStatus):
        raise StateTransitionError("graph node status must come only from GraphNodeStatus")
    return value
