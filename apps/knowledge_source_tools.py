"""Operator CLI for DB-registered fixed public-page tools.

Uses SourceProvider/Repository and the existing bounded HTTP transport. For WEB
pages, web_fallback is the existing transport identity, NOT an installed JSON
search gateway. Only explicit fixed-page fetch is exposed, never federated search.
"""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

from apps.glaucoma_knowledge_intake import CONFIG
from db.knowledge import KnowledgeStatus, ProviderType, SourceProvider, seal
from db.knowledge.repository import (
    KnowledgeNotFoundError,
    KnowledgeRepository,
    KnowledgeRepositoryError,
)
from packages.knowledge_connectors import (
    ConnectorError,
    ExplicitProxyHTTPSTransport,
    PinnedHTTPSTransport,
    ProviderConfig,
    ProxyPolicy,
    SafeHTTPClient,
    load_configuration,
)
from packages.knowledge_search.provider import configuration_proof, default_rate_policy

ROOT = Path(__file__).resolve().parents[1]
NICE_PROXY = ProxyPolicy(proxy_environment="HTTPS_PROXY")
SITES = {
    "dalhousie": (
        "Dalhousie glaucoma and myopia research",
        "medicine.dal.ca",
        "/departments/department-sites/ophthalmology/research/glaucoma-research/research/projects/myopia.html",
        "WEB",
    ),
    "nice": (
        "NICE NG81 recommendations",
        "www.nice.org.uk",
        "/guidance/ng81/chapter/Recommendations",
        "GUIDELINE",
    ),
    "nei": (
        "NEI glaucoma information",
        "www.nei.nih.gov",
        "/eye-health-information/eye-conditions-and-diseases/glaucoma",
        "WEB",
    ),
    "egs": (
        "EGS optic disc size education",
        "www.eugs.org",
        "/news/optic-disc-size-can-be-quickly-assessed-at-the-slit-lamp-2007-05-31-tip-of-the-month",
        "WEB",
    ),
}

SEARCH_TERMS = {
    "glaucoma",
    "myopia",
    "oct",
    "optic disc",
    "visual field",
    "rnfl",
    "intraocular pressure",
    "corneal thickness",
    "gonioscopy",
    "disc hemorrhage",
}


def configuration(key):
    name, host, path, kind = SITES[key]
    data = load_configuration(CONFIG).providers[0].model_dump(mode="json")
    url = "https://" + host + path
    data.update(
        provider_code="SOURCE:PROVIDER:PAGE:" + key.upper(),
        provider_version="1.1.0" if key == "nice" else "1.0.0",
        name=name,
        homepage_url=url,
        allowed_hosts=[host],
        record_url_prefix="https://" + host + "/",
        search_endpoint=url,
        fetch_endpoint=url,
        health_endpoint=url,
        catalog=[],
        adapter="guideline" if kind == "GUIDELINE" else "web_fallback",
        source_type=kind,
        additional_institution=key != "nice",
    )
    return ProviderConfig.model_validate_json(json.dumps(data))


def record(key):
    config = configuration(key)
    return seal(
        SourceProvider,
        code=config.provider_code,
        version=config.provider_version,
        name=config.name,
        provider_type=ProviderType.GUIDELINE
        if config.source_type == "GUIDELINE"
        else ProviderType.WEB_FALLBACK,
        homepage_url=config.homepage_url,
        enabled=True,
        allowlist_version=config.allowlist_version,
        configuration_proof=configuration_proof(config, default_rate_policy()),
        synthetic=False,
    )


def register(repository):
    records = tuple(record(key) for key in sorted(SITES))
    snapshots = repository.put_many(
        records, idempotency_key="public-page-tools-explicit-nice-proxy-20260915-v2"
    )
    return [s.model_dump(mode="json") for s in snapshots]


def listing(repository):
    results = []
    for key in sorted(SITES):
        try:
            snapshot = repository.get(record(key).reference())
        except KnowledgeNotFoundError:
            results.append({"key": key, "registered": False})
            continue
        results.append(
            {
                "key": key,
                "registered": True,
                "fetch_enabled": snapshot.record.enabled
                and not snapshot.withdrawn
                and snapshot.status is KnowledgeStatus.DRAFT,
                "operations": [
                    "fetch_fixed_page",
                    *(["bounded_site_search"] if key == "dalhousie" else []),
                ],
                "search_engine": False,
                "snapshot": snapshot.model_dump(mode="json"),
            }
        )
    return results


def transport_receipt(key):
    if key == "nice":
        return {**NICE_PROXY.receipt(), "policy_checksum": NICE_PROXY.checksum()}
    return {"transport": "direct-pinned-https/1.0.0", "proxy_environment": None}


def page_client(key, config):
    transport = ExplicitProxyHTTPSTransport(NICE_PROXY) if key == "nice" else PinnedHTTPSTransport()
    return SafeHTTPClient(config, transport)


def fetch(repository, key, *, client_factory=None):
    expected = record(key)
    snapshot = repository.get(expected.reference())
    if (
        snapshot.record != expected
        or snapshot.withdrawn
        or not snapshot.record.enabled
        or snapshot.status is not KnowledgeStatus.DRAFT
    ):
        raise ValueError("registered_page_tool_not_available")
    config = configuration(key)
    client = client_factory(config) if client_factory else page_client(key, config)
    result = {
        "key": key,
        "provider_ref": expected.reference().model_dump(mode="json"),
        "config_checksum": config.checksum(),
        "operation": "fetch_fixed_page",
        "source_version_created": False,
        "medical_claims_created": False,
        "transport_policy": transport_receipt(key),
        "transport_used": "injected_client" if client_factory else "configured",
    }
    try:
        response = client.get(
            config.fetch_endpoint, operation="fetch", expected_mime=config.http.allowed_mime_types
        )
        result.update(
            status="succeeded",
            metadata=response.metadata.model_dump(mode="json"),
            byte_count=len(response.body),
            content_sha256=hashlib.sha256(response.body).hexdigest(),
        )
        return result, response.body
    except ConnectorError as error:
        result.update(
            status="failed",
            category=error.category.value,
            metadata=error.metadata.model_dump(mode="json") if error.metadata else None,
        )
        return result, None


def repository(capability):
    from apps.research_single_case import credential_boundary

    return KnowledgeRepository(
        credential_boundary().resolve_target("knowledge_" + capability), capability=capability
    )


def search_site(repository, key, query, *, max_pages=6, client_factory=None):
    """Read known institutional links; not Google CSE or exhaustive university search."""
    from bs4 import BeautifulSoup

    if key != "dalhousie" or query not in SEARCH_TERMS or not 1 <= max_pages <= 12:
        raise ValueError("unsupported_site_or_general_medical_query")
    config = configuration(key)
    expected = record(key)
    client = (client_factory or (lambda c: SafeHTTPClient(c, PinnedHTTPSTransport())))(config)
    pending, seen, pages, matches = [config.fetch_endpoint], set(), [], []
    prefix = "/departments/department-sites/ophthalmology/"
    deadline = time.monotonic() + 60
    while pending and len(pages) < max_pages and time.monotonic() < deadline:
        snapshot = repository.get(expected.reference())
        if (
            snapshot.record != expected
            or snapshot.withdrawn
            or snapshot.status is not KnowledgeStatus.DRAFT
        ):
            raise ValueError("registered_page_tool_not_available")
        url = pending.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            response = client.get(url, operation="fetch", expected_mime=("text/html",))
        except ConnectorError as error:
            pages.append({"url": url, "status": "failed", "category": error.category.value})
            continue
        soup = BeautifulSoup(response.body, "html.parser")
        for anchor in soup.find_all("a", href=True):
            link = urldefrag(urljoin(url, anchor["href"]))[0]
            parts = urlsplit(link)
            if (
                parts.scheme == "https"
                and parts.netloc == "medicine.dal.ca"
                and parts.path.startswith(prefix)
                and not parts.query
                and link not in seen
                and link not in pending
                and len(pending) < 128
            ):
                pending.append(link)
        for tag in soup.find_all(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        selected = [p for p in paragraphs if query.casefold() in p.casefold()]
        pages.append(
            {"url": url, "status": "fetched", "metadata": response.metadata.model_dump(mode="json")}
        )
        if selected:
            matches.append(
                {
                    "url": url,
                    "title": soup.title.get_text(strip=True) if soup.title else None,
                    "excerpts": [p[:400] for p in selected[:3]],
                    "content_sha256": hashlib.sha256(response.body).hexdigest(),
                }
            )
    return {
        "status": "partial" if any(p["status"] == "failed" for p in pages) else "succeeded",
        "provider_ref": expected.reference().model_dump(mode="json"),
        "query": query,
        "mode": "bounded_link_search_not_google_button",
        "scope": "Dalhousie ophthalmology pages",
        "all_university_searched": False,
        "truncated": bool(pending),
        "pages": pages,
        "matches": matches,
        "medical_rules_generated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("configured", "register", "list", "show", "fetch", "search", "withdraw")
    )
    parser.add_argument("key", nargs="?", choices=tuple(SITES))
    parser.add_argument("--query", choices=sorted(SEARCH_TERMS))
    parser.add_argument("--max-pages", type=int, choices=range(1, 13), default=6)
    args = parser.parse_args(argv)
    if args.action in {"show", "fetch", "search", "withdraw"} and args.key is None:
        parser.error("this action requires a source key")
    if args.action == "configured":
        print(
            json.dumps(
                {
                    "database_checked": False,
                    "tools": [
                        {
                            "key": key,
                            "configuration": configuration(key).model_dump(mode="json"),
                            "interface": "fixed-page-http/1.0.0",
                            "transport_policy": transport_receipt(key),
                            "search_engine": False,
                        }
                        for key in sorted(SITES)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        repo = repository("publisher" if args.action == "withdraw" else "editor")
        if args.action == "register":
            result = {"database_written": True, "records": register(repo)}
        elif args.action == "list":
            result = {"database_checked": True, "tools": listing(repo)}
        elif args.action == "show":
            result = repo.get(record(args.key).reference()).model_dump(mode="json")
        elif args.action == "withdraw":
            result = repo.withdraw(
                record(args.key).reference(), "operator withdrew fixed public-page tool"
            ).model_dump(mode="json")
        else:
            if args.action == "search":
                result, body = (
                    search_site(repo, args.key, args.query, max_pages=args.max_pages),
                    None,
                )
            else:
                result, body = fetch(repo, args.key)
            parent = ROOT / ".local/knowledge-page-calls"
            parent.mkdir(mode=0o700, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=args.key + "-", dir=parent))
            if body is not None:
                with (output / "body.bin").open("xb") as f:
                    f.write(body)
            with (output / "receipt.json").open("x") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["receipt_path"] = str(output / "receipt.json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") in {"failed", "partial"} else 0
    except (OSError, KnowledgeRepositoryError, ValueError):
        print(
            json.dumps(
                {
                    "action": args.action,
                    "status": "not_completed",
                    "reason": "local_database_or_registered_tool_unavailable",
                    "database_success_not_confirmed": True,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
