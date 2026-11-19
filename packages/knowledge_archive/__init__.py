"""Normalization ports; the only persistence entry is KnowledgeRepository.archive_hit."""

from .models import ArchiveMetadata, Bibliography, FetchInput

__all__ = ["ArchiveMetadata", "Bibliography", "FetchInput"]
