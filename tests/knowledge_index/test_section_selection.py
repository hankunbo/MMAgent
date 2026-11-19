"""Explicit partial indexes preserve original positions, not silent truncation."""

import json

import pytest

from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import SectionChunkPolicy, parse_chunk_policy
from packages.knowledge_index.parsers import parse
from packages.object_store import checksum

HTML = (
    b"<html><h1>Guide</h1><h2>A</h2><p>Keep condition.</p><ul>"
    b"<li>Keep exception.</li></ul><h2>B</h2><p>Other clause.</p></html>"
)


def test_sections_preserve_every_selected_block_and_position():
    document = parse(HTML, "text/html").document
    policy = SectionChunkPolicy(source_checksum=checksum(HTML), section_paths=(("Guide", "A"),))
    selected = chunk(document, policy)
    assert selected == tuple(
        b for b in document.blocks if b.position.section_path == ("Guide", "A")
    )
    assert any("exception" in b.text for b in selected)
    assert all("Other" not in b.text for b in selected)
    assert parse_chunk_policy(policy) == policy
    assert parse_chunk_policy(json.loads(policy.model_dump_json())) == policy
    with pytest.raises(ValueError, match="chunk_count_limit"):
        chunk(document, policy.model_copy(update={"max_chunks": 1}))


def test_missing_duplicate_and_heading_only_sections_rejected():
    document = parse(HTML, "text/html").document
    for paths in ((("Guide", "missing"),), (("Guide",),)):
        with pytest.raises(ValueError, match="missing_or_heading_only"):
            chunk(document, SectionChunkPolicy(source_checksum=checksum(HTML), section_paths=paths))
    with pytest.raises(ValueError, match="duplicate_section"):
        SectionChunkPolicy(source_checksum=checksum(HTML), section_paths=(("A",), ("A",)))
