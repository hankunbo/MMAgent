"""Bounded whole-block windows, never silent loss of source paragraphs."""

import json

import pytest

from packages.knowledge_index import chunking
from packages.knowledge_index.models import Block, ChunkPolicy, ParsedDocument, Position

CHECKSUM = "sha256:" + "a" * 64


def document(count=262):
    return ParsedDocument(
        parser="xml",
        parser_version="2.3.0",
        policy_checksum=CHECKSUM,
        language="en",
        title="Full source",
        blocks=tuple(
            Block(
                text=f"Clause {i}: consider alternative explanations.",
                position=Position(
                    section_path=("Article",),
                    kind="paragraph",
                    ordinal=i,
                    span_start=0,
                    span_end=len(f"Clause {i}: consider alternative explanations."),
                ),
            )
            for i in range(1, count + 1)
        ),
    )


def test_overflow_windows_preserve_every_original_block_and_position():
    from packages.knowledge_index.models import parse_chunk_policy

    doc = document()
    before = doc.model_dump_json()
    with pytest.raises(ValueError, match="^chunk_count_limit$"):
        chunking.chunk(doc, ChunkPolicy(max_chars=3000, max_chunks=250))
    windows = chunking.plan_windows(
        doc, source_checksum=CHECKSUM, policy=ChunkPolicy(max_chars=3000, max_chunks=250)
    )
    assert len(windows) == 2
    restored = tuple(parse_chunk_policy(json.loads(w.model_dump_json())) for w in windows)
    assert tuple(b for w in restored for b in chunking.chunk(doc, w)) == doc.blocks
    assert [len(chunking.chunk(doc, w)) for w in windows] == [250, 12]
    assert doc.model_dump_json() == before


def test_window_budget_rejects_without_returning_a_truncated_plan():
    with pytest.raises(ValueError, match="^source_window_limit$"):
        chunking.plan_windows(
            document(1001),
            source_checksum=CHECKSUM,
            policy=ChunkPolicy(max_chars=3000, max_chunks=250),
        )


def test_windows_count_long_paragraph_fragments_not_only_blocks():
    doc = document(3)
    long = doc.blocks[1].model_copy(
        update={
            "text": "x" * 6100,
            "position": doc.blocks[1].position.model_copy(update={"span_end": 6100}),
        }
    )
    doc = doc.model_copy(update={"blocks": (doc.blocks[0], long, doc.blocks[2])})
    policy = ChunkPolicy(max_chars=3000, max_chunks=3)
    windows = chunking.plan_windows(doc, source_checksum=CHECKSUM, policy=policy)
    assert [len(chunking.chunk(doc, w)) for w in windows] == [1, 3, 1]
    assert all(b.position.ordinal in {1, 2, 3} for w in windows for b in chunking.chunk(doc, w))


def test_window_endpoints_and_ordinal_order_are_checked():
    from packages.knowledge_index.models import WindowChunkPolicy

    doc = document(3)
    with pytest.raises(ValueError, match="window_endpoint"):
        chunking.chunk(
            doc, WindowChunkPolicy(source_checksum=CHECKSUM, first_ordinal=1, last_ordinal=4)
        )
    backwards = doc.model_copy(update={"blocks": tuple(reversed(doc.blocks))})
    with pytest.raises(ValueError, match="window_ordinal_order"):
        chunking.plan_windows(backwards, source_checksum=CHECKSUM, policy=ChunkPolicy())


def test_prose_window_exclusion_is_explicit_and_does_not_rewrite_table():
    from packages.knowledge_index.models import ProseChunkPolicy

    doc = document(3)
    table = doc.blocks[1].model_copy(
        update={
            "text": "x" * 5000,
            "position": doc.blocks[1].position.model_copy(
                update={"kind": "table", "span_end": 5000}
            ),
        }
    )
    doc = doc.model_copy(update={"blocks": (doc.blocks[0], table, doc.blocks[2])})
    windows = chunking.plan_windows(
        doc,
        source_checksum=CHECKSUM,
        policy=ProseChunkPolicy(source_checksum=CHECKSUM, max_chars=3000, max_chunks=1),
    )
    assert all(w.tables == "excluded" for w in windows)
    assert tuple(b for w in windows for b in chunking.chunk(doc, w)) == (
        doc.blocks[0],
        doc.blocks[2],
    )
    assert doc.blocks[1] == table
