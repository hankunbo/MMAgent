"""Fixed public citation follow-up through registered Knowledge connectors.

Uses existing Source/SourceVersion/DocumentChunk storage. This is direct citation
retrieval, not a fabricated SearchHit, publication, or a new search provider.
"""

import json
from pathlib import Path

from apps.glaucoma_evidence_bindings import dump, ref, write_new
from apps.glaucoma_evidence_plan import checksum
from apps.knowledge_literature_tools import active_provider, client
from db.knowledge import models as k
from packages.knowledge_connectors import ConnectorConfiguration, ProviderRegistry, SourceLocator
from packages.object_store import checksum as bytes_checksum

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/experiments/glaucoma-associations-v1"
CITATIONS = {
    "caliber": ("pubmed", "23062656", "Retinal vessel caliber is associated"),
    "geometry": ("pubmed", "23009894", "Retinal vascular geometry and glaucoma"),
    "density": ("pubmed", "39850540", "Associations of Retinal Microvascular Density"),
    "null_geometry": ("pubmed", "37185916", "Retinal vessel architecture and geometry"),
    "tilt_lee": ("pubmed", "26505465", "Implications of Optic Disc Tilt"),
    "tilt_sawada": ("pubmed", "28114308", "Association of Myopic Deformation of Optic Disc"),
    "aao": ("aao", "AAO_POAG_PPP_2026", "Primary Open-Angle Glaucoma"),
}


def extract(document, provider, record_id):
    if provider == "pubmed":
        from defusedxml import ElementTree

        root = ElementTree.fromstring(document.response.body)
        articles = root.findall(".//PubmedArticle")
        if len(articles) != 1 or articles[0].findtext(".//PMID") != record_id:
            raise ValueError("citation_identity_mismatch")
        article = articles[0]
        title = " ".join(article.find(".//ArticleTitle").itertext()).strip()
        blocks = []
        for i, part in enumerate(article.findall(".//Abstract/AbstractText")):
            text = " ".join(" ".join(part.itertext()).split())
            blocks.append((part.get("Label") or "Abstract", None, i + 1, text))
        if not blocks:
            raise ValueError("citation_abstract_missing")
        return title, blocks
    from packages.knowledge_index.pdf_preview import preview

    blocks = []
    for page in range(24, 28):
        result = preview(document.response.body, page_start=page, page_count=1, max_chars=16000)
        if result["status"] != "succeeded" or result["excerpt_truncated"]:
            raise ValueError("citation_pdf_page_incomplete")
        blocks.append(("AAO POAG PPP", page, 1, result["excerpt"]))
    return "AAO Primary Open-Angle Glaucoma PPP (2025 approval / 2026 edition)", blocks


def verify(bundle, repo=None, store=None):
    if bundle["schema"] != "association-citation-set/1.0.0" or set(bundle["sources"]) != set(
        CITATIONS
    ):
        raise ValueError("association_citation_set")
    for key, source in bundle["sources"].items():
        provider, identifier, title = CITATIONS[key]
        if (source["provider"], source["record_id"]) != (provider, identifier):
            raise ValueError("association_citation_identity")
        if title.casefold() not in source["title"].casefold() or not source["chunks"]:
            raise ValueError("association_citation_title")
        for item in (source, *source["chunks"]):
            reference = ref(item["reference"])
            record = k.parse_record_json(reference.kind, json.dumps(item["record"]))
            if record.reference() != reference:
                raise ValueError("association_citation_record")
            if record.size_bytes > 4_194_304:
                raise ValueError("association_citation_size")
            if repo is not None:
                current = repo.get(reference)
                if current.withdrawn or current.record != record:
                    raise ValueError("association_citation_withdrawn")
                if isinstance(record, k.SourceVersion):
                    origin = repo.get(record.source_ref)
                    if origin.withdrawn or origin.record.title != source["title"]:
                        raise ValueError("association_citation_origin")
                    if provider == "pubmed" and not any(
                        i.identifier_type is k.SourceIdentifierType.PMID and i.value == identifier
                        for i in origin.record.identifiers
                    ):
                        raise ValueError("association_citation_origin_identity")
                    _, active = active_provider(repo, provider)
                    if active.record.reference() != record.provider_ref:
                        raise ValueError("association_citation_provider")
            if store is not None:
                body = store.read_bounded(record.object_uri, max(record.size_bytes, 1))
                if (
                    bytes_checksum(body) != record.content_checksum
                    or len(body) != record.size_bytes
                ):
                    raise ValueError("association_citation_bytes")
            if "text" in item and (
                bytes_checksum(item["text"].encode()) != record.content_checksum
                or record.source_version_ref != ref(source["reference"])
            ):
                raise ValueError("association_citation_text")
    if bundle["checksum"] != checksum({k: v for k, v in bundle.items() if k != "checksum"}):
        raise ValueError("association_citation_checksum")


def retrieve(repo, store, destination=DIRECTORY):
    sources = {}
    for key, (provider, identifier, expected_title) in CITATIONS.items():
        path = destination / (key + ".json")
        if path.exists():
            sources[key] = json.loads(path.read_text())
            continue
        config, registered = active_provider(repo, provider)
        connector = ProviderRegistry(
            ConnectorConfiguration(config_version="1.0.0", providers=(config,)), client
        ).get(config.provider_code)
        document = connector.fetch(SourceLocator(record_id=identifier))
        title, blocks = extract(document, provider, identifier)
        if expected_title.casefold() not in title.casefold():
            raise ValueError("citation_unexpected_title")
        raw = document.response.body
        sha = bytes_checksum(raw)
        if sha != document.response.response_checksum:
            raise ValueError("citation_transport_checksum")
        source_code = "SOURCE:GLAUCOMA_CITATION:" + key.upper()
        url = (
            "https://pubmed.ncbi.nlm.nih.gov/" + identifier + "/"
            if provider == "pubmed"
            else config.catalog[0].url
        )
        source = k.seal(
            k.Source,
            code=source_code,
            version="1.0.0",
            title=title,
            source_type=k.SourceType.ARTICLE if provider == "pubmed" else k.SourceType.GUIDELINE,
            identifiers=(
                k.SourceIdentity(identifier_type=k.SourceIdentifierType.PMID, value=identifier),
            )
            if provider == "pubmed"
            else (k.SourceIdentity(identifier_type=k.SourceIdentifierType.URL, value=url),),
        )
        version = k.seal(
            k.SourceVersion,
            code=source_code + ":CONTENT:" + sha[7:].upper(),
            version="1.0.0",
            source_ref=source.reference(),
            provider_ref=registered.record.reference(),
            discovery_refs=(),
            content_checksum=sha,
            object_uri=store.promote(store.stage(raw), sha),
            media_type=document.response.mime,
            size_bytes=len(raw),
            fetched_at=document.response.metadata.requested_at,
            allowlist_approved=True,
        )
        text = "\n\n".join(b[3] for b in blocks)
        body = text.encode()
        digest = bytes_checksum(body)
        parsed = k.seal(
            k.SourceDocument,
            code=source_code + ":TEXT:" + digest[7:].upper(),
            version="1.0.0",
            source_version_ref=version.reference(),
            content_checksum=digest,
            object_uri=store.promote(store.stage(body), digest),
            media_type="text/plain",
            size_bytes=len(body),
            parser_version="1.0.0",
        )
        chunks, offset = [], 0
        for section, page, paragraph, passage in blocks:
            body = passage.encode()
            digest = bytes_checksum(body)
            chunk = k.seal(
                k.DocumentChunk,
                code=source_code + ":CHUNK:" + digest[7:].upper(),
                version="1.0.0",
                source_version_ref=version.reference(),
                document_ref=parsed.reference(),
                content_checksum=digest,
                object_uri=store.promote(store.stage(body), digest),
                media_type="text/plain",
                size_bytes=len(body),
                parser_version="1.0.0",
                chunking_policy_version="1.0.0",
                location=k.SourceLocation(
                    section=section,
                    page=page,
                    paragraph=paragraph,
                    span_start=offset,
                    span_end=offset + len(passage),
                ),
            )
            chunks.append(
                {"reference": dump(chunk.reference()), "record": dump(chunk), "text": passage}
            )
            offset += len(passage) + 2
        records = (
            source,
            version,
            parsed,
            *(
                k.parse_record_json(k.KnowledgeKind.DOCUMENT_CHUNK, json.dumps(c["record"]))
                for c in chunks
            ),
        )
        repo.put_many(
            records, idempotency_key="association-citation:" + checksum([dump(r) for r in records])
        )
        entry = {
            "provider": provider,
            "provider_ref": dump(registered.record.reference()),
            "record_id": identifier,
            "title": title,
            "url": url,
            "retrieval": "registered_connector_fixed_identifier_not_search",
            "representation": "abstract" if provider == "pubmed" else "pdf_pages_24_27",
            "reference": dump(version.reference()),
            "record": dump(version),
            "http": dump(document.response.metadata),
            "chunks": chunks,
            "indexed": False,
            "published": False,
        }
        write_new(path, entry)
        sources[key] = entry
    value = {"schema": "association-citation-set/1.0.0", "sources": sources}
    value["checksum"] = checksum(value)
    verify(value, repo, store)
    write_new(destination / "sources.json", value)
    return value
