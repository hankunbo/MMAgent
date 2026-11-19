"""Explicit prose-only indexing preserves full original parse and table omission."""

import pytest

from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import Block, ChunkPolicy, ParsedDocument, Position


def test_long_table_does_not_discard_other_complete_prose():
    from packages.knowledge_index.models import ProseChunkPolicy, canonical, parse_chunk_policy

    blocks = tuple(
        Block(
            text=text,
            position=Position(
                section_path=("Article",), kind=kind, ordinal=i, span_start=0, span_end=len(text)
            ),
        )
        for i, (kind, text) in enumerate(
            [
                ("paragraph", "Complete source-grounded observation."),
                ("table", "x" * 5000),
                ("paragraph", "Complete limitations and alternative explanation."),
            ],
            1,
        )
    )
    doc = ParsedDocument(
        parser="xml",
        parser_version="2.1.0",
        policy_checksum="sha256:x",
        language="en",
        title="Test",
        blocks=blocks,
        fidelity_policy="text-fidelity/2.1.0",
    )
    with pytest.raises(ValueError, match="table_chunk_limit"):
        chunk(doc, ChunkPolicy(max_chars=3000))
    policy = ProseChunkPolicy(source_checksum="sha256:" + "a" * 64, max_chars=3000)
    selected = chunk(doc, parse_chunk_policy(__import__("json").loads(canonical(policy))))
    assert selected == (blocks[0], blocks[2])
    assert doc.blocks == blocks and len(blocks[1].text) == 5000
    assert b"non-table-blocks/1.0.0" in canonical(policy)


def test_table_only_is_explicitly_unusable_not_empty_success():
    from packages.knowledge_index.models import ProseChunkPolicy

    doc = ParsedDocument(
        parser="xml",
        parser_version="2.1.0",
        policy_checksum="sha256:x",
        language="en",
        title="Test",
        blocks=(
            Block(
                text="table",
                position=Position(
                    section_path=("A",), kind="table", ordinal=1, span_start=0, span_end=5
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="prose_selection_empty"):
        chunk(doc, ProseChunkPolicy(source_checksum="sha256:" + "a" * 64))
