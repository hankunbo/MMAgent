"""Versioned technical context, outside frozen medical contracts."""

from typing import Literal

from pydantic import model_validator

from packages.grounding.models import Checksum, Value, check_seal, digest
from packages.tool_admission.models import ResearchRequest


class ResearchRunContext(Value):
    schema_version: Literal["research-run/1.0.0"] = "research-run/1.0.0"
    checksum: Checksum
    request: ResearchRequest
    synthetic: bool
    research_only: Literal[True] = True
    decisive_evidence: Literal[False] = False
    scope_checksum: Checksum

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        if self.scope_checksum != digest(self.request.scope):
            raise ValueError("research_context_scope_checksum")
        return self

    @property
    def run_id(self):
        return self.request.lock.run_id

    def validate_plan(self, plan):
        from packages.tool_workflow.models import PlanBundle, ResearchContext

        bundle = PlanBundle.model_validate_json(plan.to_bundle_json())
        d = bundle.plan
        if (
            type(bundle.context) is not ResearchContext
            or bundle.context.request != self.request
            or d.scope != self.request.scope
            or d.run_id != self.run_id
            or d.catalog_ref != self.request.lock.catalog_ref
            or d.purpose != self.request.purpose
            or any(s.deployment.allowlist.synthetic_only != self.synthetic for s in d.steps)
        ):
            raise ValueError("research_context_plan_mismatch")
        return d
