"""Explicit application identity, separate from contracts v1 ToolOnboardingJob."""

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from packages.grounding.models import Checksum, Code, Token, Value, digest

EXTENSIBLE = frozenset(
    {
        "capability_unregistered",
        "implementation_missing",
        "validation_insufficient",
        "input_incompatible",
        "output_incompatible",
        "environment_unavailable",
    }
)


class QueuePolicy(Value):
    schema_version: Literal["tool-gap-policy/1.0.0"] = "tool-gap-policy/1.0.0"
    baseline_only: Literal[True] = True
    allowed: tuple[Token, ...] = tuple(sorted(EXTENSIBLE))
    purposes: tuple[Literal["debug", "shadow", "decisive_research"], ...] = (
        "debug",
        "shadow",
        "decisive_research",
    )
    lease_seconds: int = Field(default=60, ge=1, le=300)
    max_claims: int = Field(default=3, ge=1, le=10)
    lifetime_seconds: int = Field(default=86400, ge=1, le=604800)

    @model_validator(mode="after")
    def narrow(self):
        if not set(self.allowed) <= EXTENSIBLE or len(set(self.allowed)) != len(self.allowed):
            raise ValueError("extension_policy_categories")
        return self


class Finding(Value):
    kind: Token
    source_checksum: Checksum
    capability: dict[str, JsonValue] | None = None
    queueable: bool = False
    reason: Token

    @model_validator(mode="after")
    def resolved(self):
        if self.queueable and (self.kind not in EXTENSIBLE or self.capability is None):
            raise ValueError("resolved_extension_required")
        return self


class Application(Value):
    schema_version: Literal["tool-gap-application/1.0.0"] = "tool-gap-application/1.0.0"
    run_id: Code
    case_code: Code
    key: Token
    owner: Literal["05", "06_research", "06_formal", "06_execution", "08", "09"]
    purpose: Literal["debug", "shadow", "decisive_research", "research_offline"]
    requirement: dict[str, JsonValue]
    catalog: dict[str, JsonValue]
    policy: QueuePolicy
    material: dict[str, JsonValue]
    findings: tuple[Finding, ...]
    control: dict[str, JsonValue] | None = None
    intent: dict[str, JsonValue] | None = None
    missing: tuple[Token, ...] = (
        "selected_candidate",
        "evaluation_dataset",
        "threshold_policy",
        "onboarding_job",
    )
    replay: Literal["new_run_new_locks_05_grounding_06_admission"] = (
        "new_run_new_locks_05_grounding_06_admission"
    )
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"] = (
        "live_domain_acceptance_not_enabled"
    )

    @property
    def queueable(self):
        return any(f.queueable for f in self.findings)

    @property
    def checksum(self):
        return digest(self)


class Receipt(Value):
    schema_version: Literal["tool-gap-receipt/1.0.0"] = "tool-gap-receipt/1.0.0"
    task_id: Code
    application: Application
    revision: int = Field(ge=0)
    state: Literal["documented", "queued", "leased", "acknowledged", "failed", "cancelled"]
    claims: int = Field(ge=0, le=10)
    created_at: datetime
    expires_at: datetime
    lease_until: datetime | None = None
    lease_token: Token | None = None
    worker: Token | None = None
    reason: Token
    completed: Literal[False] = False
    available_next_run: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        if any(t.tzinfo is None for t in (self.created_at, self.expires_at)):
            raise ValueError("queue_absolute_time_required")
        if self.claims > self.application.policy.max_claims:
            raise ValueError("claim_budget")
        if self.state in {"leased", "acknowledged"} and (
            not self.lease_token or not self.worker or self.lease_until is None
        ):
            raise ValueError("lease_identity_required")
        if self.state in {"queued", "leased", "acknowledged"} and not self.application.queueable:
            raise ValueError("nonextensible_application")
        return self
