"""Unsectioned public prose has a locator, without inventing a source heading."""

import pytest

from db.knowledge import models as m
from db.knowledge.repository.index import _source_location
from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import ChunkPolicy, ParsePolicy, Position
from packages.knowledge_index.parsers import parse


def test_jats_body_paragraph_keeps_empty_original_heading_and_exact_offsets():
    raw = (
        b"<article><body><sec><title/><p>Glaucoma is not established by this synthetic "
        b"paragraph.</p></sec></body></article>"
    )
    result = parse(
        raw,
        "application/xml",
        language="en",
        policy=ParsePolicy(version="xml-jats-citations/1.0.0"),
    )
    assert result.status == "parsed"
    blocks = chunk(result.document, ChunkPolicy())
    assert blocks and any(not " / ".join(b.position.section_path) for b in blocks)
    for block in blocks:
        position = block.position
        before = position.model_dump(mode="json")
        location = _source_location(position)
        assert location.section == "[unsectioned document]"
        assert (location.paragraph, location.span_start, location.span_end) == (
            position.ordinal,
            position.span_start,
            position.span_end,
        )
        assert position.model_dump(mode="json") == before
        assert position.span_end - position.span_start == len(block.text)


@pytest.mark.parametrize("sections", [("Methods",), ("Results", "Limitations"), ("x" * 150,)])
def test_existing_nonempty_locator_retains_exact_legacy_bytes(sections):
    position = Position(
        section_path=sections, kind="paragraph", ordinal=7, page=2, span_start=11, span_end=25
    )
    old = m.SourceLocation(
        section=" / ".join(sections)[:128], page=2, paragraph=7, span_start=11, span_end=25
    )
    assert _source_location(position).model_dump_json() == old.model_dump_json()
