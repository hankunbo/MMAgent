"""Complete input contracts for new discovery; legacy values remain in models.py."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.grounding.models import (
    ArtifactConstraint,
    Checksum,
    Input,
    PortRef,
    Token,
    Value,
    digest,
    validate,
)

from . import models as v10


class InputRequirement(Value):
    port: PortRef | None = None
    constraint: ArtifactConstraint | None = None
    required: bool | None = None
    cardinality_min: int | None = Field(default=None, ge=0, le=64)
    cardinality_max: int | None = Field(default=None, ge=1, le=64)

    @model_validator(mode="after")
    def coherent(self):
        if self.port is not None and self.port.direction != "input":
            raise ValueError("input direction required")
        lo, hi = self.cardinality_min, self.cardinality_max
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("required input/cardinality mismatch")
        if lo is not None and self.required is not None and self.required != (lo > 0):
            raise ValueError("required input/cardinality mismatch")
        if self.complete:
            validate(Input, self.model_dump(mode="json"))
        return self

    @property
    def complete(self):
        return all(getattr(self, f) is not None for f in type(self).model_fields)


class Semantics(v10.Semantics):
    schema_version: Literal["tool-candidate-semantics/1.1.0"] = "tool-candidate-semantics/1.1.0"
    inputs: tuple[InputRequirement, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def ports_unique(self):
        ports = [digest(i.port) for i in self.inputs if i.port is not None]
        if len(ports) != len(set(ports)):
            raise ValueError("discovery_duplicate_input_port")
        return self

    @property
    def complete(self):
        return bool(self.inputs) and all(i.complete for i in self.inputs)


class DiscoveryQuery(v10.DiscoveryQuery):
    schema_version: Literal["tool-discovery-query/1.1.0"] = "tool-discovery-query/1.1.0"
    contracts: tuple[Semantics, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def owner_contract_complete(self):
        if any(not c.complete for c in self.contracts):
            raise ValueError("discovery_owner_input_contract_incomplete")
        return self


class PortBinding(Value):
    source: PortRef
    target: PortRef

    @model_validator(mode="after")
    def input_ports(self):
        if self.source.direction != "input" or self.target.direction != "input":
            raise ValueError("discovery_mapping_input_direction")
        return self


class InputMapping(Value):
    requirement_contract_checksum: Checksum
    bindings: tuple[PortBinding, ...] = Field(min_length=1, max_length=64)
    omitted_optional: tuple[PortRef, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def one_to_one(self):
        sources = [digest(b.source) for b in self.bindings]
        targets = [digest(b.target) for b in self.bindings]
        omitted = [digest(p) for p in self.omitted_optional]
        if (
            len(set(sources + omitted)) != len(sources + omitted)
            or len(set(targets)) != len(targets)
            or any(p.direction != "input" for p in self.omitted_optional)
        ):
            raise ValueError("discovery_mapping_not_one_to_one")
        return self


class Provenance(v10.Provenance):
    schema_version: Literal["tool-candidate-provenance/1.1.0"] = "tool-candidate-provenance/1.1.0"
    semantics: Semantics | None = None
    input_mappings: tuple[InputMapping, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def mapping_contracts_unique(self):
        keys = [m.requirement_contract_checksum for m in self.input_mappings]
        if len(keys) != len(set(keys)):
            raise ValueError("discovery_mapping_contract_conflict")
        return self


class ContractMatch(Value):
    status: Literal["exact", "compatible", "missing", "incompatible"]
    requirement_contract_checksum: Checksum
    reasons: tuple[Token, ...]
    bindings: tuple[PortBinding, ...] = ()
    omitted_optional: tuple[PortRef, ...] = ()


class CandidateObservation(v10.CandidateObservation):
    records: tuple[
        Annotated[Provenance | v10.Provenance, Field(discriminator="schema_version")], ...
    ] = Field(min_length=1, max_length=128)
    input_match: ContractMatch


class DiscoveryResult(v10.DiscoveryResult):
    schema_version: Literal["tool-discovery-result/1.1.0"] = "tool-discovery-result/1.1.0"
    query: DiscoveryQuery
    candidates: tuple[CandidateObservation, ...] = Field(max_length=128)


class DiscoveryPlan(v10.DiscoveryPlan):
    schema_version: Literal["tool-discovery-plan/1.1.0"] = "tool-discovery-plan/1.1.0"
    query: DiscoveryQuery
