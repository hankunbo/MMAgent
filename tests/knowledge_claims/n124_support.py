"""Offline unit stubs around actual parsed bytes; not repository source evidence."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from packages.knowledge_claims.evidence import Evidence
from packages.knowledge_claims.models import Reference
from packages.knowledge_claims.n124 import OVERVIEW_URL, URL
from packages.knowledge_index.models import ParsePolicy
from packages.knowledge_index.parsers import parse
from packages.object_store import checksum

SOURCES = Path(".local/07-p1a-review-v1/sources")


@lru_cache
def unit_evidence():
    def bound(name, query, url):
        raw = (SOURCES / (name + ".html")).read_bytes()
        parsed = parse(raw, "text/html", language="en").document
        block = next(b for b in parsed.blocks if b.text.startswith(query))
        refs = {
            kind: Reference(
                kind=kind, code="TEST:N124:" + kind, version="1.0.0", checksum="sha256:" + "a" * 64
            )
            for kind in ("source", "source_version", "source_document", "document_chunk")
        }
        source = SimpleNamespace(
            content_checksum=checksum(raw),
            source_ref=refs["source"],
            reference=lambda: refs["source_version"],
            archive_metadata=SimpleNamespace(bibliography=SimpleNamespace(canonical_url=url)),
        )
        document = SimpleNamespace(
            reference=lambda: refs["source_document"],
            content_checksum=checksum(parsed.model_dump_json().encode()),
            object_uri="object://knowledge/objects/" + "0" * 32,
        )
        chunk = SimpleNamespace(
            reference=lambda: refs["document_chunk"],
            content_checksum=checksum(block.text.encode()),
            derivation=SimpleNamespace(
                position=block.position, fidelity_policy="text-fidelity/2.0.0"
            ),
        )
        return Evidence(
            chunk,
            document,
            source,
            parsed,
            block,
            block.text,
            {
                "key": "sha256:" + "b" * 64,
                "derivation_config": {"parse_policy": ParsePolicy().model_dump(mode="json")},
            },
            raw,
        )

    main = bound("nice-recommendations", "Obtain an optic nerve", URL)
    scope = bound("nice-overview", "This guideline covers", OVERVIEW_URL)
    return replace(main, source_scope=scope)
