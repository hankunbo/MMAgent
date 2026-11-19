"""Existing 05A lock value relocated for pure query bindings; schema unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from .models import Checksum, Policy, Ref, Value, check_seal, require_kind, unique
from .validation import GroundingError


class GroundingLock(Value):
    schema_version: Literal["grounding-lock/1.0.0"] = "grounding-lock/1.0.0"
    run_id: str = Field(pattern=r"^RUN:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
    graph_code: str = Field(pattern=r"^GRAPH:(ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
    graph_version: int = Field(ge=1)
    graph_checksum: Checksum
    catalog_ref: Ref
    mapping_refs: tuple[Ref, ...] = Field(min_length=1, max_length=128)
    release_ref: Ref
    template_ref: Ref
    graph_template_ref: Ref
    knowledge_refs: tuple[Ref, ...] = Field(min_length=1, max_length=512)
    policy: Policy
    checksum: Checksum

    @model_validator(mode="after")
    def shape(self):
        require_kind(self.catalog_ref, "catalog")
        require_kind(self.release_ref, "release")
        require_kind(self.template_ref, "template")
        require_kind(self.graph_template_ref, "template")
        for r in self.mapping_refs:
            require_kind(r, "mapping")
        for r in self.knowledge_refs:
            require_kind(
                r,
                "evidence_requirement",
                "clinical_claim",
                "source_version",
                "applicability_condition",
            )
        unique(self.mapping_refs)
        unique(self.knowledge_refs)
        if len(self.mapping_refs) > self.policy.max_mappings:
            raise GroundingError("mapping_count_limit")
        check_seal(self)
        return self
