"""Bounded exact-citation acquisition using the installed Knowledge connectors.

Two public documents only; no patient data, new provider, publication or search
gateway. Every reused object is revalidated through the original archive owner.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

from apps.glaucoma_association_sources import extract
from apps.knowledge_literature_tools import active_provider, client
from apps.vectorengine_single_case import write_once
from db.knowledge import models as k
from packages.grounding.models import digest
from packages.knowledge_connectors import (
    ConnectorConfiguration,
    PinnedHTTPSTransport,
    ProviderRegistry,
    SafeHTTPClient,
    SourceLocator,
    load_configuration,
)
from packages.object_store import checksum

DESTINATION = Path("docs/experiments/active-acquisition-v1")
EGS_URL = (
    "https://eugs.org/news/rim-disc-ratio-is-a-more-sensitive-parameter-than-cup-disc-ratio"
    "-2011-06-30-tip-of-the-month"
)
SPECS = {
    "phene": ("pubmed", "31561879", "Deep Learning and Glaucoma Specialists"),
    "rim": ("egs", EGS_URL, "Rim-disc ratio"),
}


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.current, self.inside = [], [], False

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.inside, self.current = True, []

    def handle_data(self, data):
        if self.inside:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.inside:
            self.parts.append(" ".join("".join(self.current).split()))
            self.inside = False


def fetch(repo, key):
    provider, identifier, expected = SPECS[key]
    if provider == "pubmed":
        config, registered = active_provider(repo, provider)
        factory = client
        locator = SourceLocator(record_id=identifier)
    else:
        config = load_configuration(Path("configs/knowledge_sources_egs.toml")).providers[0]
        matches = list(
            {
                row["record"]["provider_ref"]["checksum"]: row["record"]["provider_ref"]
                for row in repo.archive_catalog()["records"]
                if row["record"].get("provider_ref", {}).get("code") == config.provider_code
                and row["record"]["provider_ref"]["version"] == config.provider_version
            }.values()
        )
        if len(matches) != 1:
            raise ValueError("active_egs_registered_provider_required")
        registered = repo.get(k.KnowledgeReference.model_validate_json(json.dumps(matches[0])))
        if registered.withdrawn or not registered.record.enabled:
            raise ValueError("active_provider_withdrawn")
        from packages.knowledge_search.provider import configuration_proof, default_rate_policy

        if registered.record.configuration_proof != configuration_proof(
            config, default_rate_policy()
        ):
            raise ValueError("active_provider_configuration")

        def factory(c):
            return SafeHTTPClient(c, PinnedHTTPSTransport())

        locator = SourceLocator(url=identifier)
    connector = ProviderRegistry(
        ConnectorConfiguration(config_version="1.0.0", providers=(config,)), factory
    ).get(config.provider_code)
    document = connector.fetch(locator)
    if provider == "pubmed":
        title, blocks = extract(document, provider, identifier)
    else:
        page = document.response.body.decode("utf-8")
        parser = Paragraphs()
        parser.feed(page)
        selected = [p for p in parser.parts if "rim" in p.lower() or "glaucoma" in p.lower()]
        if "0.1" not in " ".join(selected) or "diameter" not in " ".join(selected).lower():
            raise ValueError("active_rim_passage_missing")
        title = "Rim-disc ratio is a more sensitive parameter than cup-disc ratio"
        blocks = [
            ("EGS tip of the month 2011-06-30", None, n + 1, p) for n, p in enumerate(selected)
        ]
    if expected.casefold() not in title.casefold():
        raise ValueError("active_citation_title")
    return registered, document, title, blocks


def archive(repo, store, key, destination=DESTINATION):
    path = destination / (key + ".json")
    if path.exists():
        entry = json.loads(path.read_text())
        verify(repo, store, entry)
        return entry
    registered, document, title, blocks = fetch(repo, key)
    provider, identifier, _ = SPECS[key]
    url = (
        "https://pubmed.ncbi.nlm.nih.gov/" + identifier + "/"
        if provider == "pubmed"
        else identifier
    )
    raw = document.response.body
    raw_sha = checksum(raw)
    if raw_sha != document.response.response_checksum:
        raise ValueError("active_transport_checksum")
    code = "SOURCE:ACTIVE_CITATION:" + key.upper()
    source = k.seal(
        k.Source,
        code=code,
        version="1.0.0",
        title=title,
        source_type=k.SourceType.ARTICLE if provider == "pubmed" else k.SourceType.WEB,
        identifiers=(
            k.SourceIdentity(
                identifier_type=k.SourceIdentifierType.PMID
                if provider == "pubmed"
                else k.SourceIdentifierType.URL,
                value=identifier,
            ),
        ),
    )
    version = k.seal(
        k.SourceVersion,
        code=code + ":CONTENT:" + raw_sha[7:].upper(),
        version="1.0.0",
        source_ref=source.reference(),
        provider_ref=registered.record.reference(),
        discovery_refs=(),
        content_checksum=raw_sha,
        object_uri=store.promote(store.stage(raw), raw_sha),
        media_type=document.response.mime,
        size_bytes=len(raw),
        fetched_at=document.response.metadata.requested_at,
        allowlist_approved=True,
    )
    text = "\n\n".join(b[3] for b in blocks)
    body, sha = text.encode(), checksum(text.encode())
    parsed = k.seal(
        k.SourceDocument,
        code=code + ":TEXT:" + sha[7:].upper(),
        version="1.0.0",
        source_version_ref=version.reference(),
        content_checksum=sha,
        object_uri=store.promote(store.stage(body), sha),
        media_type="text/plain",
        size_bytes=len(body),
        parser_version="1.0.0",
    )
    chunks, offset = [], 0
    for section, page, paragraph, passage in blocks:
        body, sha = passage.encode(), checksum(passage.encode())
        chunk = k.seal(
            k.DocumentChunk,
            code=code + ":CHUNK:" + sha[7:].upper(),
            version="1.0.0",
            source_version_ref=version.reference(),
            document_ref=parsed.reference(),
            content_checksum=sha,
            object_uri=store.promote(store.stage(body), sha),
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
        chunks.append(chunk)
        offset += len(passage) + 2
    records = (source, version, parsed, *chunks)
    repo.put_many(records, idempotency_key="active-citation:" + digest(records))
    entry = dict(
        schema_version="active-citation-archive/1.0.0",
        key=key,
        url=url,
        title=title,
        provider_ref=registered.record.reference(),
        records=records,
        chunks=[
            dict(reference=c.reference(), text=b[3]) for c, b in zip(chunks, blocks, strict=True)
        ],
        http=document.response.metadata,
        published=False,
        clinical_validation=False,
    )
    write_once(path, entry)
    entry = json.loads(path.read_text())
    verify(repo, store, entry)
    return entry


def verify(repo, store, entry):
    if entry["schema_version"] != "active-citation-archive/1.0.0" or entry["key"] not in SPECS:
        raise ValueError("active_archive_identity")
    provider_ref = k.KnowledgeReference.model_validate_json(json.dumps(entry["provider_ref"]))
    records = tuple(
        k.parse_record_json(k.KnowledgeKind(raw["kind"]), json.dumps(raw))
        for raw in entry["records"]
    )
    refs = (provider_ref,) + tuple(r.reference() for r in records)
    snapshots = (
        repo.get_many(refs) if hasattr(repo, "get_many") else tuple(repo.get(r) for r in refs)
    )
    provider = snapshots[0]
    if provider.withdrawn or not provider.record.enabled:
        raise ValueError("active_provider_withdrawn")
    for record, current in zip(records, snapshots[1:], strict=True):
        if current.withdrawn or current.status.value == "deprecated" or current.record != record:
            raise ValueError("active_source_withdrawn_or_changed")
        if hasattr(record, "object_uri"):
            body = store.read_bounded(record.object_uri, 4194304)
            if checksum(body) != record.content_checksum or len(body) != record.size_bytes:
                raise ValueError("active_source_bytes")
    for chunk in entry["chunks"]:
        matches = [
            r for r in records if r.reference().model_dump(mode="json") == chunk["reference"]
        ]
        if len(matches) != 1:
            raise ValueError("active_chunk_identity")
        record = matches[0]
        if checksum(chunk["text"].encode()) != record.content_checksum:
            raise ValueError("active_source_excerpt")


def main():
    from apps.knowledge_source_tools import repository
    from apps.patient_service_assembly import OBJECT_ROOTS
    from packages.object_store import LocalObjectStore

    DESTINATION.mkdir(parents=True, exist_ok=True)
    repo = repository("editor")
    store = LocalObjectStore(OBJECT_ROOTS["knowledge"])
    for key in SPECS:
        result = archive(repo, store, key)
        print(
            json.dumps(
                dict(
                    key=key,
                    title=result["title"],
                    chunks=len(result["chunks"]),
                    database_written=True,
                )
            )
        )


if __name__ == "__main__":
    main()
