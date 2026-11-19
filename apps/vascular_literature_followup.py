"""Archive two fixed methods papers with the registered Europe PMC connector."""

import json
from pathlib import Path

from defusedxml import ElementTree

from apps.glaucoma_evidence_bindings import dump, write_new
from apps.knowledge_literature_tools import active_provider, client
from apps.knowledge_source_tools import repository
from apps.patient_service_assembly import OBJECT_ROOTS
from db.knowledge import models as k
from packages.knowledge_connectors import ConnectorConfiguration, ProviderRegistry, SourceLocator
from packages.object_store import LocalObjectStore, checksum

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/experiments/glaucoma-vascular-followup-v5"
CITATIONS = {
    "density": ("PMC11754513", "Associations of Retinal Microvascular Density"),
    "tortuosity": ("PMC12786152", "An improved tortuosity measurement"),
}


def verify(entry, repo, store):
    for raw in entry["records"]:
        record = k.parse_record_json(k.KnowledgeKind(raw["kind"]), json.dumps(raw))
        saved = repo.get(record.reference())
        if saved.withdrawn or saved.record != record:
            raise ValueError("vascular_source_unavailable")
        if hasattr(record, "object_uri"):
            data = store.read_bounded(record.object_uri, 4_194_304)
            if len(data) != record.size_bytes or checksum(data) != record.content_checksum:
                raise ValueError("vascular_source_bytes")


def retrieve():
    repo, store = repository("editor"), LocalObjectStore(OBJECT_ROOTS["knowledge"])
    config, provider = active_provider(repo, "europepmc")
    connector = ProviderRegistry(
        ConnectorConfiguration(config_version="1.0.0", providers=(config,)), client
    ).get(config.provider_code)
    entries = {}
    for key, (identifier, expected) in CITATIONS.items():
        path = DIRECTORY / (key + "-archive.json")
        if path.exists():
            entry = json.loads(path.read_bytes())
            verify(entry, repo, store)
            entries[key] = entry
            continue
        doc = connector.fetch(SourceLocator(record_id=identifier))
        raw = doc.response.body
        if len(raw) > 4_194_304 or checksum(raw) != doc.response.response_checksum:
            raise ValueError("vascular_source_response")
        xml = ElementTree.fromstring(raw)
        title = " ".join(xml.find(".//article-title").itertext())
        if expected not in title or identifier.removeprefix("PMC") not in [
            x.text for x in xml.findall(".//article-id")
        ]:
            raise ValueError("vascular_source_identity")
        code = "SOURCE:VASCULAR_METHODS:" + identifier
        source = k.seal(
            k.Source,
            code=code,
            version="1.0.0",
            title=title,
            source_type=k.SourceType.ARTICLE,
            identifiers=(
                k.SourceIdentity(
                    identifier_type=k.SourceIdentifierType.URL,
                    value="https://pmc.ncbi.nlm.nih.gov/articles/" + identifier + "/",
                ),
            ),
        )
        sha = checksum(raw)
        version = k.seal(
            k.SourceVersion,
            code=code + ":CONTENT:" + sha[7:].upper(),
            version="1.0.0",
            source_ref=source.reference(),
            provider_ref=provider.record.reference(),
            discovery_refs=(),
            content_checksum=sha,
            object_uri=store.promote(store.stage(raw), sha),
            media_type=doc.response.mime,
            size_bytes=len(raw),
            fetched_at=doc.response.metadata.requested_at,
            allowlist_approved=True,
        )
        # Archive source body in objects, only bounded metadata in DB/receipt.
        candidates = list(xml.findall(".//body/sec")) + list(xml.findall(".//table-wrap"))
        texts = [" ".join(" ".join(part.itertext()).split()) for part in candidates[:40]]
        parsed_bytes = "\n\n".join(texts).encode()
        parsed_sha = checksum(parsed_bytes)
        parsed = k.seal(
            k.SourceDocument,
            code=code + ":TEXT:" + parsed_sha[7:].upper(),
            version="1.0.0",
            source_version_ref=version.reference(),
            content_checksum=parsed_sha,
            object_uri=store.promote(store.stage(parsed_bytes), parsed_sha),
            media_type="text/plain",
            size_bytes=len(parsed_bytes),
            parser_version="1.0.0",
        )
        records = [source, version, parsed]
        sections = []
        offset = 0
        for index, part in enumerate(candidates[:40]):
            text = texts[index]
            if not text:
                continue
            data = text.encode()
            if len(data) > 150000:
                raise ValueError("vascular_section_budget")
            digest = checksum(data)
            section = (
                part.findtext("title") or part.findtext("label") or part.get("id") or str(index)
            )
            chunk = k.seal(
                k.DocumentChunk,
                code=code + ":CHUNK:" + str(index) + ":" + digest[7:].upper(),
                version="1.0.0",
                source_version_ref=version.reference(),
                document_ref=parsed.reference(),
                content_checksum=digest,
                object_uri=store.promote(store.stage(data), digest),
                media_type="text/plain",
                size_bytes=len(data),
                parser_version="1.0.0",
                chunking_policy_version="1.0.0",
                location=k.SourceLocation(
                    section=section,
                    paragraph=index + 1,
                    span_start=offset,
                    span_end=offset + len(text),
                ),
            )
            records.append(chunk)
            sections.append(
                {"section": section, "reference": dump(chunk.reference()), "text": text}
            )
            offset += len(text) + 2
        repo.put_many(tuple(records), idempotency_key="vascular-methods:" + identifier + ":" + sha)
        entry = {
            "title": title,
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/" + identifier + "/",
            "reference": dump(version.reference()),
            "records": [dump(r) for r in records],
            "http": dump(doc.response.metadata),
            "section_count": len(sections),
            "published": False,
            "reading_scope": "methods_and_selected_tables_not_all_references",
        }
        verify(entry, repo, store)
        write_new(path, entry)
        # Open-access excerpts for independent checking, not patient information.
        write_new(DIRECTORY / (key + "-sections.json"), sections)
        entries[key] = entry
    write_new(
        DIRECTORY / "source-verification.json",
        {
            key: {
                "reference": entry["reference"],
                "section_count": entry["section_count"],
                "verified": True,
            }
            for key, entry in entries.items()
        },
    )
    print(
        json.dumps(
            {k: {"archived": True, "sections": v["section_count"]} for k, v in entries.items()}
        )
    )


if __name__ == "__main__":
    retrieve()
