"""Read-only Planner port. Candidate capabilities belong to the offline application."""

from typing import Protocol

from .models import RetrievalContext, RetrievalInfrastructureError, RetrievalIntegrityError
from .scope import ClinicalScope


class PublishedKnowledgePort(Protocol):
    def retrieve_published_knowledge(self, run_id, query, context, config, **providers): ...


__all__ = [
    "ClinicalScope",
    "PublishedKnowledgePort",
    "RetrievalContext",
    "RetrievalInfrastructureError",
    "RetrievalIntegrityError",
]
