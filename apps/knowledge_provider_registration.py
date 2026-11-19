"""The only 04B1 write: register descriptive SourceProvider metadata as Draft."""

from db.knowledge import (
    KnowledgeReference,
    KnowledgeSnapshot,
    KnowledgeStatus,
    ProviderType,
    SourceProvider,
    seal,
)
from db.knowledge.repository import KnowledgeConflictError, KnowledgeRepository
from packages.knowledge_connectors.configuration import ProviderConfig, validate_config


def register_source_provider(
    config: ProviderConfig,
    repository: KnowledgeRepository,
    *,
    supersedes: KnowledgeReference | None = None,
    synthetic: bool = False,
    rate_policy=None,
) -> KnowledgeSnapshot:
    from packages.knowledge_search.provider import configuration_proof, default_rate_policy

    config = validate_config(config)
    if type(synthetic) is not bool:
        raise TypeError("synthetic must be an explicit boolean")
    provider_type = {
        "pubmed": ProviderType.LITERATURE,
        "europe_pmc": ProviderType.LITERATURE,
        "crossref": ProviderType.METADATA,
        "guideline": ProviderType.GUIDELINE,
        "web_fallback": ProviderType.WEB_FALLBACK,
    }[config.adapter]
    unresolved = ()
    if config.adapter == "guideline" and config.additional_institution:
        unresolved = ("KNO-008",)
    record = seal(
        SourceProvider,
        configuration_proof=configuration_proof(config, rate_policy or default_rate_policy()),
        code=config.provider_code,
        version=config.provider_version,
        name=config.name,
        provider_type=provider_type,
        homepage_url=config.homepage_url,
        enabled=config.enabled,
        allowlist_version=config.allowlist_version,
        synthetic=synthetic,
        supersedes=supersedes,
        unresolved_decisions=unresolved,
    )
    snapshot = repository.put(record)
    if snapshot.status is not KnowledgeStatus.DRAFT or snapshot.withdrawn:
        raise KnowledgeConflictError("04B1 registration requires a non-withdrawn Draft version")
    return snapshot
