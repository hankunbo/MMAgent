"""Explicit research envelopes, deliberately separate from ExpertAssessment v1."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value, digest
from packages.llm_provider.models import ExpertProposal, RequestSummary, Scope
from packages.planner_control.models import LegalActionSet, Reference


class ReviewPolicy(Value):
    schema_version: Literal["research-expert-policy/1.0.0"] = "research-expert-policy/1.0.0"
    baseline_only: Literal[True] = True
    max_no_progress: int = Field(default=2, ge=1, le=8)
    selection: Literal["first_valid_round_A_B_C"] = "first_valid_round_A_B_C"
    comparison_protocol: Checksum


class CommittedReviewReadback(Value):
    schema_version: Literal["committed-review-readback/1.0.0"] = "committed-review-readback/1.0.0"
    run_id: Token
    event: Reference
    acceptance_key: Token
    parent_checksum: Checksum
    accepted_checksum: Checksum
    report_checksum: Checksum
    intent_checksum: Checksum
    journal_checksum: Checksum


class ReviewFacts(Value):
    conflict: bool = False
    uncertain: bool = False
    high_risk: bool = False
    required: bool = False
    no_progress: int = Field(default=0, ge=0, le=10000)


class ReviewFrame(Value):
    event: Reference
    scope: Scope
    actions: LegalActionSet
    policy: ReviewPolicy
    facts: ReviewFacts

    @model_validator(mode="after")
    def binding(self):
        b = self.actions.binding
        if (b.run_id, b.case_code, b.input_checksum, b.visible_checksum, self.actions.graph) != (
            self.scope.run_id,
            self.scope.case_code,
            self.scope.input_checksum,
            self.scope.visible_checksum,
            self.scope.graph,
        ):
            raise ValueError("review_frame_scope")
        return self


class ExpertDecision(Value):
    schema_version: Literal["research-expert-decision/1.0.0"] = "research-expert-decision/1.0.0"
    event: Reference
    scope: Scope
    proposal_checksum: Checksum
    proposed_action_id: Token
    action_set_checksum: Checksum
    policy_checksum: Checksum
    source_uses: tuple[tuple[Reference, Token], ...]
    triggers: tuple[Token, ...]
    required: bool
    request_ids: tuple[Token, ...]
    suggestion: ExpertProposal | None
    decision: Literal["accept", "reject", "alternative", "safe_stop", "baseline"]
    selected_action_id: Token | None
    reason: Token
    clinical_acceptance: Literal[False] = False


class ReviewResult(Value):
    schema_version: Literal["research-review-result/1.0.0"] = "research-review-result/1.0.0"
    event: Reference
    run_id: Token
    status: Literal["accepted", "accept", "reject", "alternative", "safe_stop", "baseline"]
    reason: Token
    selected_action_id: Token | None
    graph: Reference | None
    decision_checksum: Checksum | None = None
    report_checksum: Checksum | None = None
    successor_checksum: Checksum | None = None
    acceptance_key: Token | None = None
    requests: RequestSummary
    recorded_agent_invocations: int = Field(ge=0)
    original_tool_reservations: int = Field(ge=0)
    original_tool_starts: int = Field(ge=0)
    clinical_acceptance: Literal[False] = False
    new_medical_review: Literal[False] = False

    @model_validator(mode="after")
    def accepted_binding(self):
        if self.requests.run_id != self.run_id:
            raise ValueError("review_result_run")
        if self.status == "accepted" and any(
            v is None
            for v in (
                self.graph,
                self.report_checksum,
                self.successor_checksum,
                self.acceptance_key,
            )
        ):
            raise ValueError("review_result_acceptance_required")
        if self.status != "accepted" and self.graph is not None:
            raise ValueError("review_result_graph_forbidden")
        return self


def triggers(frame, proposed):
    codes = []
    f = frame.facts
    required = (
        f.required or proposed.review_required or frame.actions.binding.profile.expert_required
    )
    for condition, code in (
        (f.conflict, "conflict"),
        (f.uncertain, "uncertain"),
        (f.high_risk, "high_risk"),
        (required, "required"),
        (proposed.kind in {"stop", "fallback"}, "stop_or_fallback"),
        (f.no_progress >= frame.policy.max_no_progress, "no_progress"),
    ):
        if condition:
            codes.append(code)
    return tuple(codes), required or f.conflict or f.high_risk


class ReviewResultV2(ReviewResult):
    """Domain facts and continuation are independent; v1 remains an exact reader."""

    schema_version: Literal["research-review-result/2.0.0"] = "research-review-result/2.0.0"
    domain_status: Literal["accepted", "not_accepted"]
    successor_status: Literal["installed", "not_installed", "unconfirmed"]
    continuation: Literal["continue", "stop"]
    cancelled: bool

    @model_validator(mode="after")
    def control_binding(self):
        if self.domain_status == "accepted" and (
            self.acceptance_key is None or self.report_checksum is None
        ):
            raise ValueError("review_history_required")
        if self.status == "accepted" and (
            self.domain_status != "accepted" or self.successor_status != "installed"
        ):
            raise ValueError("review_install_required")
        if (self.successor_status == "installed") != (self.successor_checksum is not None):
            raise ValueError("review_install_checksum")
        if self.continuation == "continue" and (self.cancelled or self.status == "safe_stop"):
            raise ValueError("review_continuation_forbidden")
        return self


class ReviewResultV3(ReviewResultV2):
    schema_version: Literal["research-review-result/3.0.0"] = "research-review-result/3.0.0"
    execution: Literal["bounded_live_research"] = "bounded_live_research"
    permission_checksum: Checksum
    new_medical_review: bool


def read_result(value):
    from packages.grounding.models import validate

    cls = {
        "research-review-result/1.0.0": ReviewResult,
        "research-review-result/2.0.0": ReviewResultV2,
        "research-review-result/3.0.0": ReviewResultV3,
    }.get(value.get("schema_version"))
    if cls is None:
        raise ValueError("research_review_result_version")
    validate(cls, value)
    return value  # No defaults or upgrades inserted into historical bytes/checksums.


def frame_checksum(frame):
    return digest(frame)
