"""04B7 transient requests and explicitly separated retrieval outcomes."""

from typing import Literal

from pydantic import Field

from packages.knowledge_index.models import Filters, IndexConfig
from packages.retrieval.knowledge.scope import Terms


class RetrievalContext(Filters):
    version: Literal["retrieval-context/1.0.0"] = "retrieval-context/1.0.0"
    disease_packs: Terms = ()
    disease_stages: Terms = ()
    sexes: Terms = ()
    exams: Terms = ()
    times: Terms = ()
    lateralities: Terms = ()
    devices: Terms = ()
    required_artifacts: tuple[Literal["template", "validation_rule", "stop_rule"], ...] = Field(
        default=(), max_length=3
    )
    age_years: int | None = Field(default=None, ge=0, le=150)


class RetrievalConfig(IndexConfig):
    """The exact frozen IndexConfig is the ranking configuration (including version)."""


Outcome = Literal["matched", "empty_result", "no_applicable_knowledge", "knowledge_gap"]


class RetrievalInfrastructureError(RuntimeError):
    """Storage, database or ranking provider failed; never medical negative evidence."""


class RetrievalIntegrityError(ValueError):
    """The exact published/database/index/object identity failed verification."""
