"""Explicit limited-use scope, independent of content lifecycle and clinical evaluation."""

from typing import Literal

from pydantic import Field, model_validator

from packages.contracts import VersionedReference
from packages.graph_engine import canonical_checksum
from packages.knowledge_claims.models import Model

RESEARCH_PURPOSE = "limited_research_evidence_planning"


class ResearchScope(Model):
    version: Literal["knowledge-research-scope/1.0.0"] = "knowledge-research-scope/1.0.0"
    purpose: Literal["limited_research_evidence_planning"]
    definition_ref: VersionedReference
    context_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    allowed_outputs: tuple[
        Literal["evidence_requirements", "insufficiency", "limitations"], ...
    ] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def exact(self):
        if self.definition_ref.checksum is None or not self.definition_ref.code.startswith(
            "POLICY:DISEASE_PACK:"
        ):
            raise ValueError("KNO_B03_DEFINITION_BINDING")
        if len(set(self.allowed_outputs)) != len(self.allowed_outputs):
            raise ValueError("KNO_B03_OUTPUT_SCOPE")
        return self


class KnowledgeUseRequest(Model):
    version: Literal["knowledge-use-request/1.0.0"] = "knowledge-use-request/1.0.0"
    purpose: Literal["limited_research_evidence_planning", "clinical"]
    policy_ref: VersionedReference
    scope_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact(self):
        if self.policy_ref.checksum is None:
            raise ValueError("KNO_B03_POLICY_REQUIRED")
        return self


def context_checksum(context):
    return canonical_checksum(context.model_dump(mode="json", exclude={"use"}))


def validate_use(scope, policy_ref, context):
    from db.knowledge.repository import KnowledgeConflictError

    use = getattr(context, "use", None)
    if use is None or use.purpose != RESEARCH_PURPOSE:
        raise KnowledgeConflictError("KNO_B03_USE_REQUIRED")
    if use.policy_ref != policy_ref or use.scope_checksum != canonical_checksum(scope):
        raise KnowledgeConflictError("KNO_B03_POLICY_BINDING")
    if context_checksum(context) != scope.context_checksum:
        raise KnowledgeConflictError("KNO_B03_CONTEXT_SCOPE")
