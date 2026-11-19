"""Public Grounding 05A domain API."""

from .models import (
    ArtifactConstraint,
    Candidate,
    CatalogSnapshot,
    CatalogTool,
    GroundingRecord,
    History,
    Input,
    Mapping,
    MappingSnapshot,
    Policy,
    PortRef,
    Ref,
    Review,
    State,
    Term,
    canonical,
    digest,
    seal,
    validate,
)
from .ports import KnowledgeProjection, KnowledgeReader, ToolCatalogReader, ToolSafety
from .validation import GroundingError, verify

__all__ = [
    "ArtifactConstraint",
    "Candidate",
    "CatalogSnapshot",
    "CatalogTool",
    "GroundingRecord",
    "History",
    "Input",
    "Mapping",
    "MappingSnapshot",
    "Policy",
    "PortRef",
    "Ref",
    "Review",
    "State",
    "Term",
    "canonical",
    "digest",
    "seal",
    "validate",
    "KnowledgeProjection",
    "KnowledgeReader",
    "ToolCatalogReader",
    "ToolSafety",
    "GroundingError",
    "verify",
]

# 05B remains importable offline without Runtime/DB/Agent dependencies.
from .compatibility import artifact_compatibility, constraint_compatibility, tool_compatibility
from .locks import GroundingLock
from .query import evaluate_query, normalized_request
from .query_models import (
    ArtifactMetadata,
    CaseScope,
    CatalogProducer,
    Component,
    EnvironmentFact,
    ExtensionSuggestion,
    Finding,
    InputImpact,
    PathSource,
    ProductDemand,
    QueryConfig,
    QueryRequest,
    QueryResult,
    RankingEstimate,
)

__all__ += [
    "artifact_compatibility",
    "constraint_compatibility",
    "tool_compatibility",
    "GroundingLock",
    "evaluate_query",
    "normalized_request",
    "ArtifactMetadata",
    "CaseScope",
    "Component",
    "EnvironmentFact",
    "QueryConfig",
    "QueryRequest",
    "QueryResult",
    "RankingEstimate",
    "CatalogProducer",
    "ExtensionSuggestion",
    "Finding",
    "InputImpact",
    "PathSource",
    "ProductDemand",
]
