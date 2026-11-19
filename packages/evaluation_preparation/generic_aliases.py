"""Exact source aliases for new generic review material, never response repair."""

from packages.evidence_acquisition.source_aliases import alias_sources
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate

from .comprehensive_materials import ComprehensivePhase


def alias_review(state, *, previous=()):
    aliases, manifest = alias_sources(state.material.sources, previous=previous)
    mapping = {
        old.reference.code: new.reference.model_dump(mode="json")
        for old, new in zip(state.material.sources, aliases, strict=True)
    }
    raw = state.model_dump(mode="json")
    for part in (raw["policy"]["items"], raw["policy"]["requirements"]):
        for item in part:
            if "source_codes" in item:
                item["source_codes"] = [mapping[c]["code"] for c in item["source_codes"]]
            definition = item["definition"]
            definition["source_refs"] = [mapping[r["code"]] for r in definition["source_refs"]]
            body = {k: v for k, v in definition.items() if k != "reference"}
            definition["reference"].update(
                version="research-source-alias/1.0.0", checksum=digest(body)
            )
    material = raw["material"]
    material["sources"] = [s.model_dump(mode="json") for s in aliases]
    material["policy_ref"]["checksum"] = material["scope"]["policy_checksum"] = digest(
        raw["policy"]
    )
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    result = validate(DomainState, raw)
    project_review_tasks(result)
    return result, manifest


class GenericPhase(ComprehensivePhase):
    def project_current(self):
        state = super().project_current()
        if self.target is not None:
            return state
        state, self.source_alias_manifest = alias_review(state)
        return state
