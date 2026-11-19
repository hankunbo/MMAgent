"""Actual 04B1 NG81 discovery/fetch; no fabricated archive or publication.

Default: deterministic local catalog search. --fetch: bounded original-site HTTP.
The JSON is a discovery receipt, not a persisted SourceVersion or medical rule.
"""

import argparse
import json
from pathlib import Path

from packages.knowledge_connectors import (
    ConnectorError,
    GuidelineSiteConnector,
    PinnedHTTPSTransport,
    SafeHTTPClient,
    SourceLocator,
    StructuredQuery,
    load_configuration,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/knowledge_sources_glaucoma.toml"


def query_plan():
    from packages.knowledge_search import CodedTerm, KnowledgeQueryPlan, ProviderBudget

    return KnowledgeQueryPlan(
        goal="GOAL:KNOWLEDGE:EVIDENCE",
        claim="CLAIM:KNOWLEDGE:ASSOCIATION",
        evidence_need="diagnostic_criteria",
        diseases=(CodedTerm(code="CONCEPT:DISEASE:GLAUCOMA", label="glaucoma"),),
        population=("research cohort",),
        source_types=("guideline",),
        providers=(
            ProviderBudget(
                provider_code="SOURCE:PROVIDER:NICE:GLAUCOMA",
                page_size=10,
                max_hits=10,
                max_pages=1,
                max_requests=1,
                max_retries=0,
            ),
        ),
        total_max_hits=10,
        total_max_requests=1,
        max_concurrency=1,
    )


def persist_discovery(repository):
    """Only Provider Draft and original 04B2 Query/Run/Hit ports; no rule publishing."""
    from apps.knowledge_federated_search import KnowledgeSearchService
    from apps.knowledge_provider_registration import register_source_provider
    from packages.knowledge_search import FederatedSearch, load_rate_policy

    configuration = load_configuration(CONFIG)
    policy = load_rate_policy(CONFIG.with_name("knowledge_search.toml"))
    provider = register_source_provider(
        configuration.providers[0], repository, synthetic=False, rate_policy=policy
    )

    def factory(config, governor):
        return SafeHTTPClient(
            config, PinnedHTTPSTransport(), clock=governor.clock, governor=governor
        )

    with FederatedSearch(configuration, policy, factory) as executor:
        receipt = KnowledgeSearchService(executor, repository, (provider,)).search(
            query_plan(), idempotency_key="ng81-curated-discovery-20260915-v1"
        )
    return {
        "status": receipt.status,
        "cache_hit": receipt.cache_hit,
        "records": [x.model_dump(mode="json") for x in receipt.records],
        "database_written": True,
        "source_archived": False,
        "published": False,
    }


def discover(*, fetch=False):
    configuration = load_configuration(CONFIG)
    config = configuration.providers[0]
    client = SafeHTTPClient(config, PinnedHTTPSTransport())
    connector = GuidelineSiteConnector(config, client)
    page = connector.search(StructuredQuery(text="glaucoma", page_size=10))
    result = {
        "configuration_checksum": config.checksum(),
        "discovery_mode": "curated_catalog_not_web_search",
        "page": page.model_dump(mode="json"),
        "fetch": [],
        "database_written": False,
        "published": False,
    }
    if fetch:
        for record in page.records:
            try:
                doc = connector.fetch(SourceLocator(record_id=record.record_id))
                result["fetch"].append(
                    {
                        "record_id": record.record_id,
                        "status": "fetched_not_archived",
                        "metadata": doc.response.metadata.model_dump(mode="json"),
                        "byte_count": len(doc.response.body),
                    }
                )
            except ConnectorError as error:
                result["fetch"].append(
                    {
                        "record_id": record.record_id,
                        "status": "failed",
                        "category": error.category.value,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Register Draft discovery through the existing local Knowledge editor",
    )
    args = parser.parse_args()
    result = discover(fetch=args.fetch)
    if args.persist:
        import psycopg

        from apps.research_single_case import credential_boundary
        from db.knowledge.repository import KnowledgeRepository, KnowledgeRepositoryError

        try:
            repository = KnowledgeRepository(
                credential_boundary().resolve_target("knowledge_editor"), capability="editor"
            )
            result["persistence"] = persist_discovery(repository)
            result["database_written"] = True
        except (OSError, psycopg.Error, KnowledgeRepositoryError):
            result["persistence"] = {"status": "unavailable", "database_written": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = any(x["status"] == "failed" for x in result["fetch"])
    return 2 if failed or (args.persist and not result["database_written"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
