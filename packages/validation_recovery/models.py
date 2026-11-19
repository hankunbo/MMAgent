"""09C additive research records; no clinical contracts or run status replacement."""

from typing import Literal

from pydantic import model_validator

from packages.grounding.models import Checksum, Token, Value
from packages.path_graph_validation.models import PathGraphSummary
from packages.planner_control.models import Command, Control, Gap, Reference

Category = Literal[
    "missing_input",
    "product_demand",
    "mapping_unresolved",
    "unknown_producer",
    "capability_gap",
    "admission_rejected",
    "execution_failed",
    "qc_failed",
    "medical_negative",
    "valid_refutation",
    "version_conflict",
]


class Failure(Value):
    category: Category
    target: Token
    reason: Token
    sources: tuple[Reference, ...]
    gap: Gap | None = None

    @model_validator(mode="after")
    def distinct(self):
        if not self.sources:
            raise ValueError("failure_owner_reference_required")
        kinds = {
            "missing_input": "missing_data",
            "product_demand": "product_demand",
            "mapping_unresolved": "mapping",
            "unknown_producer": "unknown_producer",
            "capability_gap": "capability",
            "admission_rejected": "safety",
            "execution_failed": "technical",
            "qc_failed": "technical",
        }
        if self.category == "capability_gap" and self.gap is None:
            raise ValueError("resolved_capability_required")
        if self.gap is not None and (
            self.gap.kind != kinds.get(self.category) or self.gap.target != self.target
        ):
            raise ValueError("failure_category_gap_mismatch")
        return self


class BranchRoute(Value):
    branch: Token
    forks: tuple[Token, ...]


class RecoveryPolicy(Value):
    schema_version: Literal["recovery-policy/1.0.0"] = "recovery-policy/1.0.0"
    baseline_only: Literal[True] = True
    routes: tuple[BranchRoute, ...]
    source: Reference

    @model_validator(mode="after")
    def unique(self):
        if len({r.branch for r in self.routes}) != len(self.routes):
            raise ValueError("unique_branch_routes_required")
        if any(not r.forks for r in self.routes):
            raise ValueError("declared_fork_required")
        return self


class ToolRequirement(Value):
    schema_version: Literal["tool-requirement-intent/1.0.0"] = "tool-requirement-intent/1.0.0"
    origin_run_id: Token
    binding_checksum: Checksum
    catalog: Reference
    gap: Gap
    sources: tuple[Reference, ...]
    available_inputs: tuple[Reference, ...]
    attempted_branches: tuple[Token, ...]
    purpose: Literal["research_offline"] = "research_offline"
    extension: Literal["recorded_for_11_not_dispatched"] = "recorded_for_11_not_dispatched"
    replay: Literal["new_run_with_new_locks_and_admission_required"] = (
        "new_run_with_new_locks_and_admission_required"
    )

    @model_validator(mode="after")
    def capability_only(self):
        if self.gap.kind != "capability":
            raise ValueError("resolved_capability_required")
        return self


class RecoveryIncident(Value):
    schema_version: Literal["recovery-incident/1.0.0"] = "recovery-incident/1.0.0"
    key: Token
    control: Control
    command: Command
    failure: Failure
    policy_checksum: Checksum
    action_set_checksum: Checksum | None
    stop_reason: Token | None
    summary: PathGraphSummary | None = None
    requirements: tuple[ToolRequirement, ...] = ()

    @model_validator(mode="after")
    def original_command(self):
        if self.command.key != "recovery:" + self.key or self.command.action not in {
            "review",
            "fallback",
        }:
            raise ValueError("recovery_command_binding")
        if self.control.pending and self.control.pending.command != self.command:
            raise ValueError("recovery_pending_binding")
        if not self.control.pending and self.command.expected_revision != self.control.revision:
            raise ValueError("recovery_command_revision")
        if not any(
            r.role == "recovery_policy" and r.checksum == self.policy_checksum
            for r in self.command.input_refs
        ):
            raise ValueError("recovery_policy_binding")
        return self


class RecoveryStopReport(Value):
    schema_version: Literal["recovery-stop-report/1.0.0"] = "recovery-stop-report/1.0.0"
    incident: RecoveryIncident
    control_observed: Control
    expert_result: dict
    provider_requests: dict
    stop_reason: Token
    unobserved_nodes: tuple[Token, ...]
    retained_evidence: tuple[Reference, ...]
    attempted_alternatives: tuple[Token, ...]
    unresolved: tuple[Gap, ...]
    conclusion_status: Literal["bounded_research_stop"] = "bounded_research_stop"
    scope_limitations: Literal["declared_research_requirements_only_no_clinical_conclusion"] = (
        "declared_research_requirements_only_no_clinical_conclusion"
    )
    final_control: Literal["read_original_Planner_report_after_confirmation"] = (
        "read_original_Planner_report_after_confirmation"
    )
    clinical_effect: None = None


class RecoveryFallbackReport(RecoveryStopReport):
    schema_version: Literal["recovery-fallback-report/1.0.0"] = "recovery-fallback-report/1.0.0"
    conclusion_status: Literal["requires_current_control_confirmation"] = (
        "requires_current_control_confirmation"
    )
