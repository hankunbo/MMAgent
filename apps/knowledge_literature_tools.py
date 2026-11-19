"""Live literature CLI, using the original Knowledge registration/search ports.

Catalog discovery, network fetch and persisted search are deliberately separate.
No clinical claims are published, and no patient inputs are accepted here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from apps.knowledge_federated_search import KnowledgeSearchService
from apps.knowledge_provider_registration import register_source_provider
from apps.knowledge_source_tools import repository
from db.knowledge import KnowledgeStatus, ProviderType, SourceProvider, seal
from db.knowledge.repository import KnowledgeNotFoundError, KnowledgeRepositoryError
from packages.knowledge_connectors import (
    ConnectorConfiguration,
    ConnectorError,
    ErrorCategory,
    ExplicitProxyHTTPSTransport,
    PinnedHTTPSTransport,
    ProviderConfig,
    ProviderRegistry,
    ProxyPolicy,
    SafeHTTPClient,
    SourceLocator,
    load_configuration,
)
from packages.knowledge_connectors.configuration import validate_contact_email
from packages.knowledge_search import FederatedSearch, configuration_proof, load_rate_policy

ROOT = Path(__file__).resolve().parents[1]
CONTACT_FILE = ROOT / ".local/knowledge_contact.json"
PROVIDERS = {
    "pubmed": "SOURCE:PROVIDER:PUBMED",
    "europepmc": "SOURCE:PROVIDER:EUROPEPMC",
    "crossref": "SOURCE:PROVIDER:CROSSREF",
    "aao": "SOURCE:PROVIDER:AAO",
    "cochrane": "SOURCE:PROVIDER:COCHRANE:GLAUCOMA",
}


def contact_environment(name):
    """Local operator contact; values never enter provider identity or results."""
    value = os.environ.get(name)
    if value is not None or name != "MMAGENT_NCBI_EMAIL":
        return value
    try:
        with CONTACT_FILE.open(encoding="utf-8") as source:
            raw = source.read(4097)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ConnectorError(ErrorCategory.CONFIGURATION) from None
    try:
        if len(raw) > 4096:
            raise ValueError
        data = json.loads(raw)
        if (
            type(data) is not dict
            or set(data) != {"version", "ncbi_email"}
            or data["version"] != "knowledge-contact/1.0.0"
        ):
            raise ValueError
        return validate_contact_email(data["ncbi_email"])
    except (ValueError, TypeError, KeyError):
        raise ConnectorError(ErrorCategory.CONFIGURATION) from None


def configuration():
    baseline = load_configuration(ROOT / "configs/knowledge_sources.toml")
    curated = load_configuration(ROOT / "configs/knowledge_sources_curated_literature.toml")
    return ConnectorConfiguration(
        config_version="1.0.0",
        providers=tuple(
            ProviderConfig.model_validate(c.model_copy(update={"provider_version": "2.1.0"}))
            if c.adapter == "crossref"
            else c
            for c in baseline.providers
            if c.adapter in ("pubmed", "europe_pmc", "crossref")
        )
        + tuple(c for c in curated.providers if c.provider_code != PROVIDERS["aao"])
        + (aao_current(),),
    )


def policy():
    return load_rate_policy(ROOT / "configs/knowledge_search.toml")


def provider_config(key):
    code = PROVIDERS[key]
    return next(c for c in configuration().providers if c.provider_code == code)


def provider_record(config):
    kind = {
        "pubmed": ProviderType.LITERATURE,
        "europe_pmc": ProviderType.LITERATURE,
        "crossref": ProviderType.METADATA,
        "guideline": ProviderType.GUIDELINE,
        "web_fallback": ProviderType.WEB_FALLBACK,
    }[config.adapter]
    return seal(
        SourceProvider,
        code=config.provider_code,
        version=config.provider_version,
        name=config.name,
        provider_type=kind,
        homepage_url=config.homepage_url,
        enabled=config.enabled,
        allowlist_version=config.allowlist_version,
        configuration_proof=configuration_proof(config, policy()),
        synthetic=False,
        supersedes=provider_record(crossref_baseline()).reference()
        if config.adapter == "crossref" and config.provider_version == "2.1.0"
        else provider_record(aao_pdf_baseline()).reference()
        if config.provider_code == PROVIDERS["aao"] and config.provider_version == "2.2.1"
        else provider_record(aao_baseline()).reference()
        if config.provider_code == PROVIDERS["aao"] and config.provider_version == "2.2.0"
        else None,
    )


def crossref_baseline():
    return next(
        c
        for c in load_configuration(ROOT / "configs/knowledge_sources.toml").providers
        if c.adapter == "crossref"
    )


def aao_baseline():
    return next(
        c
        for c in load_configuration(
            ROOT / "configs/knowledge_sources_curated_literature.toml"
        ).providers
        if c.provider_code == PROVIDERS["aao"]
    )


def aao_pdf_baseline():
    return load_configuration(ROOT / "configs/knowledge_sources_aao_pdf.toml").providers[0]


def aao_current():
    # 2.2.0 was registered during verification. Preserve its exact bytes/reference;
    # 2.2.1 probes the PDF, not an HTML landing page rejected by the PDF MIME policy.
    baseline = aao_pdf_baseline()
    return ProviderConfig.model_validate(
        baseline.model_copy(
            update={
                "provider_version": "2.2.1",
                "health_endpoint": baseline.catalog[0].url,
            }
        )
    )


def register(repo):
    for config in configuration().providers:
        if config.adapter == "crossref" and config.provider_version == "2.1.0":
            register_source_provider(
                crossref_baseline(), repo, synthetic=False, rate_policy=policy()
            )
        if config.provider_code == PROVIDERS["aao"] and config.provider_version == "2.2.1":
            register_source_provider(aao_baseline(), repo, synthetic=False, rate_policy=policy())
            register_source_provider(
                aao_pdf_baseline(),
                repo,
                synthetic=False,
                rate_policy=policy(),
                supersedes=provider_record(aao_baseline()).reference(),
            )
        register_source_provider(
            config,
            repo,
            supersedes=provider_record(config).supersedes,
            synthetic=False,
            rate_policy=policy(),
        )
    return listing(repo)


def active_provider(repo, key):
    config = provider_config(key)
    expected = provider_record(config)
    snapshot = repo.get(expected.reference())
    if (
        snapshot.record != expected
        or snapshot.withdrawn
        or snapshot.status is not KnowledgeStatus.DRAFT
        or not snapshot.record.enabled
    ):
        raise ValueError("registered_literature_provider_unavailable")
    return config, snapshot


def configured():
    result = []
    for key in PROVIDERS:
        config = provider_config(key)
        result.append(
            {
                "provider": key,
                "name": config.name,
                "version": config.provider_version,
                "provider_ref": provider_record(config).reference().model_dump(mode="json"),
                "discovery_mode": "curated_catalog" if config.catalog else "remote_api",
                "catalog_entries": [c.model_dump(mode="json") for c in config.catalog],
                "contact_environment": config.contact_email_env,
                "contact_configured": bool(contact_environment(config.contact_email_env))
                if config.contact_email_env
                else None,
                "fetch_content": {
                    "pubmed": "abstract_record_xml",
                    "europepmc": "open_access_full_text_by_pmcid",
                    "crossref": "bibliographic_metadata_not_full_text",
                    "aao": "official_academy_pdf_with_bounded_page_text",
                    "cochrane": "public_review_summary_not_subscription_full_text",
                }[key],
                "live_network_status": "not_checked_by_listing",
            }
        )
    return result


def listing(repo):
    result = configured()
    for item in result:
        try:
            _, snapshot = active_provider(repo, item["provider"])
            item.update(registered=True, enabled=True, synthetic=snapshot.record.synthetic)
        except KnowledgeNotFoundError:
            item.update(registered=False, enabled=False)
        except ValueError:
            item.update(registered=True, enabled=False)
    return result


def client(config, governor=None, *, transport="proxy"):
    wire = (
        ExplicitProxyHTTPSTransport(proxy_policy(config))
        if transport == "proxy"
        else PinnedHTTPSTransport()
    )
    if transport not in ("proxy", "direct"):
        raise ValueError("unsupported_transport")
    return SafeHTTPClient(
        config,
        wire,
        clock=governor.clock if governor else None,
        governor=governor,
        environment=contact_environment,
    )


def proxy_policy(config):
    # NCBI and official AAO PDF downloads use the verified domain CONNECT route.
    # This is explicit assembly, not an error-triggered or direct-route fallback.
    return ProxyPolicy(
        proxy_environment="HTTPS_PROXY",
        target_resolution="proxy_hostname"
        if config.adapter == "pubmed" or config.provider_code == PROVIDERS["aao"]
        else "local_public_ip_pinned",
    )


def normalized_topic(value):
    value = value.strip().casefold()
    return {"青光眼": "glaucoma", "rnfl": "rnfl"}.get(value, value)


def duplicate_groups(hits):
    """Link exact DOI/PMID/PMCID/URL identities, retaining every provider hit."""
    groups = []
    for hit in hits:
        identities = {
            (
                i["identifier_type"],
                i["value"].casefold() if i["identifier_type"] == "DOI" else i["value"],
            )
            for i in hit["identifiers"]
        }
        matches = [g for g in groups if identities & g[0]]
        merged_ids, merged_hits = identities, [hit]
        for group in matches:
            merged_ids |= group[0]
            merged_hits.extend(group[1])
            groups.remove(group)
        groups.append((merged_ids, merged_hits))
    return [
        {
            "title": members[0]["title"],
            "identifiers": [
                {"identifier_type": kind, "value": value} for kind, value in sorted(ids)
            ],
            "hits": members,
        }
        for ids, members in groups
    ]


def search(
    repo,
    keys,
    topic,
    *,
    limit=8,
    key=None,
    transport="proxy",
    client_factory=None,
    topics=None,
    request_guard=None,
    timeout_ms=30000,
    public_target=None,
    public_target_vocabulary=None,
):
    from packages.knowledge_search.broad import broad_plan
    from packages.knowledge_search.targeted import targeted_plan

    if not keys or len(set(keys)) != len(keys) or not 1 <= limit <= 20:
        raise ValueError("invalid_provider_selection_or_limit")
    selected = tuple(active_provider(repo, k) for k in keys)
    normalized = normalized_topic(topic)
    builder = broad_plan if public_target is None else targeted_plan
    if public_target_vocabulary is not None:
        from packages.knowledge_search.registered_targets import registered_plan

        if public_target_vocabulary != "registered-task-public/1.0.0" or public_target is None:
            raise ValueError("public_target_vocabulary_binding")
        builder = registered_plan
    plan = builder(
        tuple(c.provider_code for c, _ in selected),
        topics=topics
        if topics is not None
        else (() if normalized == "glaucoma" else (normalized,)),
        limit=limit,
        timeout_ms=timeout_ms,
        **({} if public_target is None else {"target": public_target}),
    )
    request_key = key or "literature-" + uuid4().hex
    config = ConnectorConfiguration(config_version="1.0.0", providers=tuple(c for c, _ in selected))
    factory = client_factory or (lambda c, g: client(c, g, transport=transport))
    with FederatedSearch(config, policy(), factory, request_guard=request_guard) as executor:
        service = KnowledgeSearchService(
            executor, repo, tuple(s for _, s in selected), synthetic=False
        )
        receipt = service.search(plan, idempotency_key=request_key)
    runs, hits, queries = [], [], []
    for snapshot in receipt.records:
        record = snapshot.record
        kind = record.kind.value
        if kind == "search_query":
            queries.append({"provider": record.provider_ref.code, "text": record.query_text})
        elif kind == "search_run":
            runs.append(
                {
                    "provider": record.provider_ref.code,
                    "reference": record.reference().model_dump(mode="json"),
                    "outcome": record.execution.model_dump(mode="json"),
                }
            )
        elif kind == "search_hit":
            identities = dict.fromkeys(
                (i.identifier_type, i.value)
                for i in (*record.identifiers, *record.discovery.discovery_identifiers)
            )
            hits.append(
                {
                    "provider": record.provider_ref.code,
                    "reference": record.reference().model_dump(mode="json"),
                    "title": record.title,
                    "identifiers": [
                        {"identifier_type": str(kind), "value": value} for kind, value in identities
                    ],
                }
            )
    grouped = duplicate_groups(hits)
    result = {
        "status": receipt.status,
        "request_key": request_key,
        "cache_hit": receipt.cache_hit,
        "database_written": not receipt.cache_hit,
        "topic": normalized_topic(topic),
        "transport": transport if client_factory is None else "injected_test_client",
        "proxy_policies": {
            c.provider_code: {
                **proxy_policy(c).receipt(),
                "checksum": proxy_policy(c).checksum(),
            }
            for c, _ in selected
        }
        if transport == "proxy" and client_factory is None
        else {},
        "queries": queries,
        "runs": runs,
        "provider_hit_count": len(hits),
        "unique_identity_group_count": len(grouped),
        "results": grouped,
        "classification": "discovery_candidates_not_medical_conclusions",
        "preferences_applied_as_filters": False,
    }
    if topics is not None:
        result["topic"] = None
        result["topics"] = list(topics)
        result["topic_combination"] = "and_between_topics_or_within_synonyms"
    return result


def fetch(
    repo,
    key,
    record_id,
    *,
    transport="proxy",
    client_factory=None,
    max_chars=4000,
    page_start=1,
    page_count=5,
):
    config, snapshot = active_provider(repo, key)
    if not 1 <= max_chars <= 16000:
        raise ValueError("invalid_excerpt_limit")
    if not 1 <= page_start <= 200 or not 1 <= page_count <= 10:
        raise ValueError("invalid_pdf_page_range")
    factory = client_factory or (lambda c: client(c, transport=transport))
    registry = ProviderRegistry(
        ConnectorConfiguration(config_version="1.0.0", providers=(config,)), factory
    )
    if key == "cochrane":
        entries = [e for e in config.catalog if e.record_id == record_id]
        if len(entries) != 1:
            raise ValueError("unknown_curated_record")
        locator = SourceLocator(url=entries[0].url)
    else:
        locator = SourceLocator(record_id=record_id)
    result = {
        "provider": key,
        "provider_ref": snapshot.record.reference().model_dump(mode="json"),
        "record_id": record_id,
        "archived": False,
        "proxy_policy": proxy_policy(config).receipt()
        if transport == "proxy" and client_factory is None
        else None,
    }
    try:
        document = registry.get(config.provider_code).fetch(locator)
    except ConnectorError as error:
        return {
            **result,
            "status": "failed",
            "category": error.category.value,
            "metadata": error.metadata.model_dump(mode="json") if error.metadata else None,
        }
    response = document.response
    text = response.body.decode("utf-8", errors="replace")
    if response.mime == "text/html":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = (soup.find("main") or soup).get_text(" ", strip=True)
    elif response.mime in ("application/xml", "text/xml"):
        from defusedxml import ElementTree
        from defusedxml.common import DefusedXmlException

        try:
            tree = ElementTree.fromstring(response.body)
        except (ElementTree.ParseError, DefusedXmlException):
            return {
                **result,
                "status": "failed",
                "category": ErrorCategory.PARSE.value,
                "metadata": response.metadata.model_dump(mode="json"),
            }
        text = " ".join(" ".join(tree.itertext()).split())
    elif response.mime == "application/pdf":
        from packages.knowledge_index.pdf_preview import preview

        parsed = preview(
            response.body, page_start=page_start, page_count=page_count, max_chars=max_chars
        )
        return {
            **result,
            "status": "succeeded" if parsed["status"] == "succeeded" else "failed",
            "mime": response.mime,
            "byte_count": response.byte_count,
            "checksum": response.response_checksum,
            "metadata": response.metadata.model_dump(mode="json"),
            **parsed,
        }
    return {
        **result,
        "status": "succeeded",
        "mime": response.mime,
        "byte_count": response.byte_count,
        "checksum": response.response_checksum,
        "metadata": response.metadata.model_dump(mode="json"),
        "text_extracted": response.mime != "application/pdf",
        "excerpt": text[:max_chars],
        "excerpt_truncated": len(text) > max_chars,
    }


def sweep(repo, keys, *, topics, limit=5, key=None, transport="proxy"):
    from packages.knowledge_search import PrivacyError
    from packages.knowledge_search.broad import TOPICS

    if not topics or len(set(topics)) != len(topics) or any(t not in TOPICS for t in topics):
        raise ValueError("invalid_topic_batch")
    if not keys or len(set(keys)) != len(keys) or any(k not in PROVIDERS for k in keys):
        raise ValueError("invalid_provider_selection")
    batch_key = key or "literature-batch-" + uuid4().hex
    results = []
    for topic in topics:
        for provider in keys:
            request_key = f"{batch_key}:per-source-v1:{topic}:{provider}"
            try:
                result = search(
                    repo,
                    (provider,),
                    topic,
                    limit=limit,
                    key=request_key,
                    transport=transport,
                )
            except PrivacyError:
                # A rejected response must not cancel independent sources or be
                # disguised as an empty result/persisted run. Preserve the core
                # service's zero-write rejection and never print rejected text.
                result = {
                    "topic": topic,
                    "provider": provider,
                    "request_key": request_key,
                    "status": "rejected_metadata",
                    "reason": "public_bibliographic_metadata_policy_rejected",
                    "database_written": False,
                    "results": [],
                    "provider_hit_count": 0,
                }
            results.append(result)
    # Deduplicate for display across topics as well as providers. All original
    # hits and exact stored searches remain available under topic_results.
    all_hits = [hit for r in results for g in r["results"] for hit in g["hits"]]
    statuses = {r["status"] for r in results}
    if statuses <= {"failed", "error", "timeout", "cancelled", "rejected_metadata"}:
        status = "failed"
    elif statuses <= {"succeeded", "empty"}:
        status = "succeeded" if all_hits else "empty"
    else:
        status = "partial"
    return {
        "batch_key": batch_key,
        "batch_mode": "independent_topic_provider/1.0.0",
        "status": status,
        "topic_count": len(topics),
        "search_task_count": len(results),
        "provider_hit_count": len(all_hits),
        "unique_identity_group_count": len(duplicate_groups(all_hits)),
        "topic_results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("configured", "register", "list", "topics"):
        sub.add_parser(name).add_argument("--output", type=Path)
    lookup = sub.add_parser("search")
    lookup.add_argument("--providers", default="europepmc,crossref")
    topic_selection = lookup.add_mutually_exclusive_group()
    topic_selection.add_argument("--topic", default="glaucoma")
    topic_selection.add_argument(
        "--topics", help="Comma-separated approved public topics; AND semantics"
    )
    lookup.add_argument("--limit", type=int, default=8)
    lookup.add_argument("--key")
    lookup.add_argument("--transport", choices=("proxy", "direct"), default="proxy")
    lookup.add_argument("--output", type=Path)
    batch = sub.add_parser("sweep")
    batch.add_argument("--providers", default="europepmc,crossref")
    batch.add_argument("--topics", default="all")
    batch.add_argument("--limit", type=int, default=5)
    batch.add_argument("--key")
    batch.add_argument("--transport", choices=("proxy", "direct"), default="proxy")
    batch.add_argument("--output", type=Path)
    get = sub.add_parser("fetch")
    get.add_argument("provider", choices=tuple(PROVIDERS))
    get.add_argument("--record-id", required=True)
    get.add_argument("--max-chars", type=int, default=4000)
    get.add_argument("--page-start", type=int, default=1)
    get.add_argument("--page-count", type=int, default=5)
    get.add_argument("--transport", choices=("proxy", "direct"), default="proxy")
    get.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Reject an existing export before any search/registration side effects.
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("output must be a new file in an existing directory")
    try:
        if args.command == "configured":
            result = configured()
        elif args.command == "topics":
            from packages.knowledge_search.broad import TOPICS

            result = TOPICS
        else:
            repo = repository("editor")
            if args.command == "register":
                result = register(repo)
            elif args.command == "list":
                result = listing(repo)
            elif args.command == "search":
                result = search(
                    repo,
                    tuple(args.providers.split(",")),
                    args.topic,
                    limit=args.limit,
                    key=args.key,
                    transport=args.transport,
                    topics=tuple(args.topics.split(",")) if args.topics is not None else None,
                )
            elif args.command == "sweep":
                from packages.knowledge_search.broad import TOPICS

                result = sweep(
                    repo,
                    tuple(args.providers.split(",")),
                    topics=tuple(TOPICS) if args.topics == "all" else tuple(args.topics.split(",")),
                    limit=args.limit,
                    key=args.key,
                    transport=args.transport,
                )
            else:
                result = fetch(
                    repo,
                    args.provider,
                    args.record_id,
                    transport=args.transport,
                    max_chars=args.max_chars,
                    page_start=args.page_start,
                    page_count=args.page_count,
                )
    except (KnowledgeRepositoryError, ConnectorError, ValueError, KeyError) as error:
        # Do not print connection DSNs, identity environment values or full input.
        print(json.dumps({"status": "error", "category": type(error).__name__}))
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
        print(json.dumps({"output": str(args.output), "status": "saved"}))
    else:
        print(rendered)
    if isinstance(result, dict) and result.get("status") in ("failed", "error"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
