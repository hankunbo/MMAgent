"""Explicit simulation identity and policy; outside frozen contracts v1."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Token, Value
from packages.planner_control.models import Reference

Kind = Literal[
    "remove_evidence", "remove_resource", "reverse", "select_check", "replace_tool", "replace_path"
]


class Intervention(Value):
    kind: Kind
    targets: tuple[Token, ...] = Field(min_length=1, max_length=128)
    alternative: Token | None = None

    @model_validator(mode="after")
    def typed(self):
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("simulation_duplicate_target")
        if self.kind in {"remove_evidence", "remove_resource", "reverse"} and self.alternative:
            raise ValueError("simulation_unexpected_alternative")
        return self


class Candidate(Value):
    id: Token
    interventions: tuple[Intervention, ...] = Field(min_length=1, max_length=16)


class SimulationPolicy(Value):
    schema_version: Literal["counterfactual-policy/1.0.0"] = "counterfactual-policy/1.0.0"
    source: Reference
    case_code: Token
    run_id: Token
    fact_checksum: Checksum
    rule: Reference
    synthetic: Literal[True]
    applicability: Literal["synthetic_same_case_input_exact_definition_unit"] = (
        "synthetic_same_case_input_exact_definition_unit"
    )
    allowed: dict[Kind, tuple[Token, ...]]
    alternatives: dict[Token, Checksum] = Field(default_factory=dict)
    forbidden_combinations: tuple[tuple[Kind, ...], ...] = ()
    max_candidates: int = Field(ge=1, le=128)
    max_search: int = Field(ge=1, le=256)
    tolerance: float = Field(ge=0, le=1)
    coverage_retention: float = Field(ge=0, le=1)
    preserve_roles: bool
    metrics: tuple[Literal["score", "support", "refutation", "conflict", "coverage"], ...]
    expert_required: bool = False
    expected_cost: float | None = Field(default=None, ge=0)
    expected_failure_risk: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def bounded(self):
        if not self.allowed or not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("simulation_policy_empty")
        if any(not v or len(set(v)) != len(v) for v in self.allowed.values()):
            raise ValueError("simulation_allowed_targets")
        return self


class SimulationReport(Value):
    schema_version: Literal["counterfactual-report/1.0.0", "counterfactual-report/1.1.0"] = (
        "counterfactual-report/1.1.0"
    )
    simulation_id: Token
    identity: Literal["hypothetical_candidate"] = "hypothetical_candidate"
    case_code: Token
    run_id: Token
    policy_checksum: Checksum
    fact_checksum: Checksum
    candidate: Candidate
    baseline: dict
    simulated: dict | None
    affected: tuple[dict, ...]
    remaining: tuple[Token, ...]
    comparisons: dict
    alternatives: dict
    classification: Literal[
        "context_redundant",
        "components_changed",
        "conclusion_changed",
        "necessary",
        "indeterminate",
    ]
    reasons: tuple[str, ...]
    counters: dict
    applicability: Literal["frozen_snapshot_and_rule_only"] = "frozen_snapshot_and_rule_only"
    clinical_acceptance: Literal[False] = False
    disease_probability: None = None
    medical_causal_effect: None = None
    memory_status: Literal["candidate_not_published"] = "candidate_not_published"
    checksum: Checksum
