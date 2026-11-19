"""Narrow read-only ports; current safety decisions are distinct from frozen content."""

from typing import Literal, Protocol

from pydantic import Field

from packages.retrieval.knowledge.models import RetrievalContext

from .models import CatalogSnapshot, Mapping, Ref, Value


class KnowledgeProjection(Value):
    requirement_ref: Ref
    claim_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    release_ref: Ref
    template_ref: Ref
    source_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    applicability_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    graph_template_ref: Ref | None
    target_node_code: str | None = None
    artifact_type_code: str = Field(min_length=1, max_length=128)
    synthetic: bool
    mode: Literal["published", "replay"]


class KnowledgeReader(Protocol):
    def read(
        self, mapping: Mapping, context: RetrievalContext, *, replay: bool
    ) -> KnowledgeProjection:
        """Exact references, release/template membership, source and applicability checks."""
        ...


class ToolSafety(Value):
    catalog_ref: Ref
    disabled_implementations: tuple[Ref, ...] = Field(max_length=128)
    revision: int = Field(ge=0)


class ToolCatalogReader(Protocol):
    def read_snapshot(self, reference: Ref) -> CatalogSnapshot:
        """Resolve immutable content-bound catalog, never current rows as old catalog."""
        ...

    def current_safety(self, reference: Ref) -> ToolSafety:
        """Current revocations can reject old candidates, never replace them."""
        ...


class EstimateReader(Protocol):
    def read_estimate(self, catalog_ref: Ref, estimate_ref: Ref):
        """Return query_models.RankingEstimate with exact catalog/content binding.

        Optional separate port: null Mapping estimate_ref needs no reader. Technical
        errors propagate. Values are source-attributed estimates, not clinical facts.
        """
        ...
