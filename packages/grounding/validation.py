"""05A mapping composition and machine policy; no ranking or execution planning."""

from pydantic import Field, model_validator

from packages.retrieval.knowledge.models import RetrievalContext

from .compatibility import tool_compatibility
from .models import Candidate, CatalogSnapshot, Mapping, Policy, Value, canonical, validate
from .ports import KnowledgeProjection, KnowledgeReader, ToolCatalogReader, ToolSafety


class GroundingError(ValueError):
    """Stable boundary error carrying a machine-readable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _base(
    mapping: Mapping,
    policy: Policy,
    knowledge: KnowledgeReader,
    catalogs: ToolCatalogReader,
    context: RetrievalContext,
    *,
    admission: bool,
):
    mapping = validate(Mapping, mapping)
    policy = validate(Policy, policy)
    context = validate(RetrievalContext, context)
    if mapping.policy_ref != policy.reference():
        raise GroundingError("policy_version_mismatch")
    if len(canonical(mapping)) > policy.max_bytes:
        raise GroundingError("mapping_size_limit")
    if (
        len(mapping.candidates) > policy.max_candidates
        or len(mapping.accepted_artifacts) > policy.max_artifacts
        or any(len(c.inputs) > policy.max_inputs for c in mapping.candidates)
    ):
        raise GroundingError("mapping_count_limit")
    from .models import CatalogSnapshot

    catalog = validate(CatalogSnapshot, catalogs.read_snapshot(mapping.catalog_ref))
    if catalog.reference() != mapping.catalog_ref:
        raise GroundingError("catalog_version_mismatch")
    if policy.synthetic_only and not (mapping.synthetic and catalog.synthetic):
        raise GroundingError("synthetic_policy_required")
    projection = validate(
        KnowledgeProjection, knowledge.read(mapping, context, replay=not admission)
    )
    for key in (
        "requirement_ref",
        "claim_refs",
        "release_ref",
        "template_ref",
        "source_refs",
        "applicability_refs",
    ):
        if getattr(mapping, key) != getattr(projection, key):
            raise GroundingError("knowledge_reference_mismatch")
    if projection.mode != ("published" if admission else "replay"):
        raise GroundingError("knowledge_channel_mismatch")
    if policy.synthetic_only and not projection.synthetic:
        raise GroundingError("synthetic_knowledge_required")
    # Extra alternatives express explicitly reviewed Grounding equivalence; original stays present.
    if projection.artifact_type_code not in {
        a.artifact_ref.code for a in mapping.accepted_artifacts
    }:
        raise GroundingError("knowledge_artifact_missing")
    safety = (
        validate(ToolSafety, catalogs.current_safety(mapping.catalog_ref)) if admission else None
    )
    if safety and safety.catalog_ref != mapping.catalog_ref:
        raise GroundingError("safety_catalog_mismatch")
    return mapping, catalog, safety


class PathAdmission(Value):
    candidate: Candidate
    reasons: tuple[str, ...]


class MappingPaths(Value):
    record: Mapping
    catalog: CatalogSnapshot
    safety: ToolSafety
    paths: tuple[PathAdmission, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def complete_paths(self):
        if (
            self.record.catalog_ref != self.catalog.reference()
            or self.safety.catalog_ref != self.record.catalog_ref
            or len(self.paths) != len(self.record.candidates)
            or {p.candidate for p in self.paths} != set(self.record.candidates)
        ):
            raise ValueError("incomplete or misbound path evaluation")
        return self


def evaluate_paths(mapping, policy, knowledge, catalogs, context) -> MappingPaths:
    """Current Published/global binding first; each OR path then evaluated independently."""
    mapping, catalog, safety = _base(mapping, policy, knowledge, catalogs, context, admission=True)
    return MappingPaths(
        record=mapping,
        catalog=catalog,
        safety=safety,
        paths=tuple(
            PathAdmission(candidate=c, reasons=tool_compatibility(c, catalog, policy, safety))
            for c in sorted(mapping.candidates, key=lambda c: c.code)
        ),
    )


def verify(mapping, policy, knowledge, catalogs, context, *, admission):
    """05A management admission remains all-or-nothing; replay never authorizes queries."""
    mapping, catalog, safety = _base(
        mapping, policy, knowledge, catalogs, context, admission=admission
    )
    for constraint in mapping.accepted_artifacts:
        if constraint.artifact_ref not in catalog.artifacts:
            raise GroundingError("artifact_outside_catalog")
        terms = (*constraint.scope_terms, constraint.modality.code, constraint.unit.code)
        if any(t and t not in policy.allowed_terms for t in terms):
            raise GroundingError("unknown_scope_term")
    for candidate in mapping.candidates:
        reasons = tool_compatibility(candidate, catalog, policy, safety, admission=admission)
        if reasons:
            raise GroundingError(reasons[0])
    return mapping
