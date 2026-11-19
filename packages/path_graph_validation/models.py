"""09B additive offline envelope; never a contracts-v1 medical acceptance."""

from typing import Any, Literal

from pydantic import Field

from packages.grounding.models import Checksum, Token, Value


class PathGraphSummary(Value):
    schema_version: Literal["path-graph-validation/1.0.0"] = "path-graph-validation/1.0.0"
    domain: Literal["research", "formal_synthetic"]
    purpose: Literal["research_offline", "synthetic_mechanism"]
    case_code: Token
    run_id: Token
    graph: dict[str, Any]
    binding: dict[str, Any]
    paths: tuple[dict[str, Any], ...]
    goals: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    unresolved: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...] = ()
    duplicate_groups: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, Any]
    risk: dict[str, Any]
    review_facts: dict[str, Any]
    original_report: dict[str, Any] | None = None
    original_report_checksum: Checksum | None = None
    legal_candidates: tuple[dict[str, Any], ...] = ()
    stop_candidate: dict[str, Any] | None = None
    clinical_acceptance: Literal[False] = False
    live_status: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )
    checksum: Checksum = Field()


class PathGraphReview(Value):
    from agents.expert_agent.models import ExpertDecision as _Decision
    from packages.planner_control.models import Reference as _Reference

    schema_version: Literal["path-graph-review/1.0.0"] = "path-graph-review/1.0.0"
    summary_checksum: Checksum
    summary: PathGraphSummary
    event: _Reference
    result: dict[str, Any]  # read_result validates 2.0 / exact old 1.0 without rewriting.
    expert_decision: _Decision | None
    disposition: Literal["candidate_only_no_tool_dispatch"]
    checksum: Checksum
