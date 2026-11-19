"""Additive stage bounds inside the original request journal, not a second ledger."""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Code, Token, Value, validate


class StageAllocation(Value):
    schema_version: Literal["bounded-live-allocation/1.0.0"] = "bounded-live-allocation/1.0.0"
    session_id: Token
    stage: Literal["public", "case"]
    public_run_id: Code
    public_used: int = Field(ge=0, le=12)
    public_journal_checksum: Checksum | None
    configuration_checksums: tuple[Checksum, ...]

    @model_validator(mode="after")
    def exact(self):
        if not 1 <= len(self.configuration_checksums) <= 3:
            raise ValueError("stage_config_inventory")
        if self.stage == "public" and (
            self.public_used or self.public_journal_checksum is not None
        ):
            raise ValueError("public_allocation")
        if self.stage == "case" and (not self.public_used or self.public_journal_checksum is None):
            raise ValueError("public_proof_required")
        return self


class PriorCaseUsage(Value):
    run_id: Code
    configuration_checksum: Checksum
    control_checksum: Checksum
    journal_checksum: Checksum
    used: int = Field(ge=1, le=40)


class StageAllocationV2(StageAllocation):
    """Explicit charged, terminal previous cases; never reclassify them as public probes."""

    schema_version: Literal["bounded-live-allocation/2.0.0"] = "bounded-live-allocation/2.0.0"
    stage: Literal["case"]
    prior_cases: tuple[PriorCaseUsage, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def bounded_history(self):
        if (
            len({p.run_id for p in self.prior_cases}) != len(self.prior_cases)
            or self.public_run_id in {p.run_id for p in self.prior_cases}
            or self.public_used + sum(p.used for p in self.prior_cases) >= 40
        ):
            raise ValueError("prior_case_budget_binding")
        return self


class HistoricalRunUsage(PriorCaseUsage):
    stage: Literal["public", "case"]


class StageAllocationV3(StageAllocation):
    schema_version: Literal["bounded-live-allocation/3.0.0"] = "bounded-live-allocation/3.0.0"
    history: tuple[HistoricalRunUsage, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def history_binding(self):
        ids = [p.run_id for p in self.history]
        if len(set(ids)) != len(ids) or self.public_run_id in ids:
            raise ValueError("historical_run_identity")
        if self.public_used + sum(p.used for p in self.history) >= 40:
            raise ValueError("historical_budget_exhausted")
        return self


def read_allocation(value):
    raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    model = {
        "bounded-live-allocation/1.0.0": StageAllocation,
        "bounded-live-allocation/2.0.0": StageAllocationV2,
        "bounded-live-allocation/3.0.0": StageAllocationV3,
    }.get(raw.get("schema_version"))
    if model is None:
        raise ValueError("stage_allocation_version")
    return validate(model, raw)


def prior_case_used(allocation):
    return sum(p.used for p in getattr(allocation, "prior_cases", ())) + sum(
        p.used for p in getattr(allocation, "history", ())
    )
