"""Bounded NEI discovery/archive via existing Knowledge owners; never publish rules."""

import json
from pathlib import Path

from apps.knowledge_federated_search import KnowledgeSearchService
from apps.knowledge_provider_registration import register_source_provider
from apps.knowledge_source_tools import configuration, repository
from apps.patient_service_assembly import OBJECT_ROOTS
from apps.vectorengine_single_case import write_once
from db.knowledge.models import KnowledgeKind, KnowledgeReference
from packages.knowledge_archive import Bibliography, FetchInput
from packages.knowledge_connectors import (
    PinnedHTTPSTransport,
    SafeHTTPClient,
    SourceLocator,
    WebSearchFallbackConnector,
)
from packages.knowledge_connectors.configuration import ConnectorConfiguration, ProviderConfig
from packages.knowledge_search import (
    CodedTerm,
    FederatedSearch,
    KnowledgeQueryPlan,
    ProviderBudget,
    load_rate_policy,
)
from packages.object_store import LocalObjectStore, checksum

DESTINATION = Path("docs/acceptance/13-unified-evidence-v1/nei-risk-blocks.json")


def ingest():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    repo = repository("editor")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    if DESTINATION.exists():
        value = json.loads(DESTINATION.read_bytes())
        from types import SimpleNamespace

        from apps.vectorengine_round_materials import RoundCaseEvidence

        reader = object.__new__(RoundCaseEvidence)
        reader.original = SimpleNamespace(knowledge=repo, knowledge_objects=store)
        reader.read_source(value)
        return value
    data = configuration("nei").model_dump(mode="json")
    data.update(
        provider_code="SOURCE:PROVIDER:NEI:GLAUCOMA_INFO",
        catalog=[
            dict(
                record_id="NEI-GLAUCOMA",
                title="Glaucoma risk information",
                url=data["homepage_url"],
            )
        ],
    )
    config = ProviderConfig.model_validate_json(json.dumps(data))
    configs = ConnectorConfiguration(config_version="1.0.0", providers=(config,))
    policy = load_rate_policy(Path("configs/knowledge_search.toml"))
    provider = register_source_provider(config, repo, synthetic=False, rate_policy=policy)
    plan = KnowledgeQueryPlan(
        goal="GOAL:KNOWLEDGE:EVIDENCE",
        claim="CLAIM:KNOWLEDGE:ASSOCIATION",
        evidence_need="diagnostic_criteria",
        diseases=(CodedTerm(code="CONCEPT:DISEASE:GLAUCOMA", label="glaucoma"),),
        population=("research cohort",),
        source_types=("web",),
        providers=(
            ProviderBudget(
                provider_code=config.provider_code,
                page_size=1,
                max_hits=1,
                max_pages=1,
                max_requests=1,
                max_retries=0,
            ),
        ),
        total_max_hits=1,
        total_max_requests=1,
        max_concurrency=1,
    )

    def client(c, governor):
        return SafeHTTPClient(c, PinnedHTTPSTransport(), clock=governor.clock, governor=governor)

    with FederatedSearch(configs, policy, client) as executor:
        receipt = KnowledgeSearchService(executor, repo, (provider,), synthetic=False).search(
            plan, idempotency_key="unified-evidence-nei-20261005-v1"
        )
    hit = next(r.record for r in receipt.records if r.record.kind is KnowledgeKind.SEARCH_HIT)
    prior = [
        r
        for r in repo.archive_catalog()["observations"]
        if r["hit_ref"] == hit.reference().model_dump(mode="json")
        and r["pipeline_version"] == "1.2.0"
    ]
    if len(prior) > 1:
        raise ValueError("nei_multiple_archives_require_selection")
    if prior:
        archived = prior[0]
    else:
        connector = WebSearchFallbackConnector(
            config, SafeHTTPClient(config, PinnedHTTPSTransport())
        )
        fetched = connector.fetch(SourceLocator(url=config.homepage_url))
        archived = repo.archive_hit(
            hit.reference(),
            FetchInput(
                pipeline_version="1.2.0",
                document=fetched,
                metadata_response_checksum=fetched.response.response_checksum,
                bibliography=Bibliography(
                    title="Glaucoma",
                    language="en",
                    institutions=("National Eye Institute",),
                    canonical_url=config.homepage_url,
                ),
            ),
            store,
        )
    write_once(DESTINATION.with_name("nei-approved-fetch-receipt.json"), archived)
    # Archive creates canonical documents; exact bytes, not a copied web-search summary.
    from db.knowledge import models as k
    from packages.knowledge_index.models import IndexConfig

    context = k.seal(
        k.ApplicabilityCondition,
        code="EVIDENCE:NEI:RESEARCH_CONTEXT",
        version="1.0.0",
        population=("research cohort",),
        conditions=("glaucoma",),
        scope_note="Research risk context; not a diagnosis or published rule",
    )
    repo.put(context)
    index = repo.build_document_index(
        archived,
        store,
        IndexConfig(
            mode="fts_only",
            fts_weight=1.0,
            vector_weight=0.0,
            vector_unavailable_reason="not_required_for_exact_risk_passage",
        ),
        applicability_refs=(context.reference(),),
        targets=("glaucoma_authoring",),
    )
    write_once(DESTINATION.with_name("nei-index.json"), index)
    refs = [index["document_ref"]] if index.get("document_ref") else []
    if not refs:
        raise ValueError("nei_archive_document_missing")
    for reference in refs:
        record = repo.get(KnowledgeReference.model_validate_json(json.dumps(reference))).record
        raw = store.read_bounded(record.object_uri, 4194304)
        if checksum(raw) != record.content_checksum:
            raise ValueError("nei_archive_checksum")
        text = raw.decode("utf-8")
        blocks = json.loads(text)["blocks"]
        indices = [
            i
            for i, block in enumerate(blocks)
            if any(
                "at risk for glaucoma" in title.lower()
                for title in block["position"]["section_path"]
            )
        ]
        passage = "\n".join(blocks[i]["text"] for i in indices)
        if "60" not in passage or "family" not in passage.lower():
            continue
        result = dict(
            id="NEI-GLAUCOMA-RISK",
            reference=reference,
            text=passage,
            text_checksum=checksum(passage.encode()),
            locator=config.homepage_url,
            selector=dict(kind="json_blocks", indices=indices),
            archive=archived,
        )
        write_once(DESTINATION, result)
        return result
    raise ValueError("nei_exact_risk_passage_not_found")


if __name__ == "__main__":
    result = ingest()
    print(
        json.dumps(
            {k: result[k] for k in ("id", "reference", "text_checksum", "locator")}, indent=2
        )
    )
