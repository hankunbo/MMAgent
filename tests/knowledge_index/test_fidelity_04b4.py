"""Semantic regressions reproduced against the original 04B4 parsers first."""

import pytest

from packages.knowledge_index.parsers import parse


@pytest.mark.parametrize("format", ["xml", "html"])
def test_inline_decimal_and_word_form(format):
    if format == "xml":
        body = (
            "<article><p>Threshold 0.<italic>5</italic> units; "
            "non<bold>negative</bold>.</p></article>"
        )
    else:
        body = "<html><body><p>Threshold 0.<b>5</b> units; non<b>negative</b>.</p></body></html>"
    result = parse(body.encode(), "application/xml" if format == "xml" else "text/html")
    assert result.status == "parsed"
    assert result.document.blocks[0].text == "Threshold 0.5 units; nonnegative."


@pytest.mark.parametrize("format", ["xml", "html"])
def test_table_shape_is_recoverable(format):
    root = "article" if format == "xml" else "html"
    mime = "application/xml" if format == "xml" else "text/html"
    a = (
        f"<{root}><table><tr><td>A</td><td>B</td></tr>"
        f"<tr><td>1</td><td>2</td></tr></table></{root}>"
    )
    b = f"<{root}><table><tr><td>A</td><td>B</td><td>1</td><td>2</td></tr></table></{root}>"
    first, second = (parse(body.encode(), mime).document for body in (a, b))
    assert first != second
    assert first.blocks[0].table.rows == 2
    assert first.blocks[0].table.columns == 2
    assert second.blocks[0].table.rows == 1
    assert second.blocks[0].table.columns == 4


def document(markup, format="xml"):
    root = "article" if format == "xml" else "html"
    result = parse(
        f"<{root}>{markup}</{root}>".encode(), "application/xml" if format == "xml" else "text/html"
    )
    assert result.status == "parsed", result
    return result.document


@pytest.mark.parametrize("format", ["xml", "html"])
@pytest.mark.parametrize(
    "markup,expected",
    [
        (
            "<p>0.<b><i>5</i></b> mg/<i>dL</i>; non<b>negative</b>; 不<b>推荐</b>。</p>",
            "0.5 mg/dL; nonnegative; 不推荐。",
        ),
        (
            "<p>x &lt;<b>=</b> 0.<b>5</b>; ≥<i>2</i> mm; no <b>benefit</b>.</p>",
            "x <= 0.5; ≥2 mm; no benefit.",
        ),
        (
            "<p>a  <b>b </b> c\t<i>d</i>e; 中文<b>ABC</b>。<br/>next</p>",
            "a  b  c\tde; 中文ABC。\nnext",
        ),
        ("<div><p>first<b>word</b></p><p>second <i>word</i></p></div>", "firstword\nsecond word"),
        ("<ul><li>no<b>thing</b></li><li>not <i>shown</i></li></ul>", "nothing\nnot shown"),
    ],
)
def test_inline_boundaries_and_exact_canonical_slices(format, markup, expected):
    from packages.knowledge_index.chunking import chunk
    from packages.knowledge_index.models import FIDELITY_VERSION, ChunkPolicy

    doc = document(markup, format)
    canonical_text = "\n".join(b.text for b in doc.blocks)
    assert canonical_text == expected
    assert doc.fidelity_policy == FIDELITY_VERSION
    for block in (*doc.blocks, *chunk(doc, ChunkPolicy())):
        assert canonical_text[block.position.span_start : block.position.span_end] == block.text


@pytest.mark.parametrize("format", ["xml", "html"])
def test_titles_share_inline_rules(format):
    markup = (
        "<sec><title>non<b>negative</b> 0.<i>5</i></title><p>value</p></sec>"
        if format == "xml"
        else (
            "<head><title>nonnegative 0.5</title></head>"
            "<h1>non<b>negative</b> 0.<i>5</i></h1><p>value</p>"
        )
    )
    doc = document(markup, format)
    assert doc.title == "nonnegative 0.5"
    assert doc.blocks[-1].position.section_path == ("nonnegative 0.5",)


@pytest.mark.parametrize("format", ["xml", "html"])
@pytest.mark.parametrize("tag", ["sup", "sub", "math"])
def test_numeric_markup_is_explicitly_unsupported(format, tag):
    root = "article" if format == "xml" else "html"
    result = parse(
        f"<{root}><p>10<{tag}>2</{tag}> mg</p></{root}>".encode(),
        "application/xml" if format == "xml" else "text/html",
    )
    assert (result.status, result.reason, result.document) == (
        "unsupported",
        "numeric_markup_unsupported",
        None,
    )


GRID = (
    '<table><tr><th>Group</th><th scope="col">Threshold (mg)</th></tr>'
    '<tr><th scope="row">nonnegative</th><td>0.<b>5</b></td></tr></table>'
)


@pytest.mark.parametrize("format", ["xml", "html"])
def test_grid_headers_cells_and_whole_table_chunks(format):
    from packages.knowledge_index.chunking import chunk
    from packages.knowledge_index.models import ChunkPolicy, digest

    doc = document("<p>before</p>" + GRID + "<p>after</p>", format)
    assert doc == document("<p>before</p>" + GRID + "<p>after</p>", format)
    blocks = chunk(doc, ChunkPolicy(max_chars=64, overlap_chars=8))
    assert len(blocks) == 3
    block = blocks[1]
    assert block.table == doc.blocks[1].table
    assert block.position == doc.blocks[1].position
    assert block.position.table_capability.checksum == digest(block.table)
    assert block.position.table_capability.numeric_evidence is True
    value = block.table.cells[-1]
    assert value.text == "0.5"
    assert value.headers == ((0, 1), (1, 0))
    assert block.text[value.span_start : value.span_end] == "0.5"
    canonical_text = "\n".join(b.text for b in doc.blocks)
    for cell in block.table.cells:
        assert (
            canonical_text[
                block.position.span_start + cell.span_start : block.position.span_start
                + cell.span_end
            ]
            == cell.text
        )
    assert canonical_text[block.position.span_start : block.position.span_end] == block.text
    # A large table is rejected as a unit; it never creates orphan value chunks.
    big = document(GRID.replace("nonnegative", "nonnegative" * 6), format)
    with pytest.raises(ValueError, match="table_chunk_limit"):
        chunk(big, ChunkPolicy(max_chars=64, overlap_chars=8))


@pytest.mark.parametrize("format", ["xml", "html"])
@pytest.mark.parametrize(
    "markup,reason",
    [
        ('<table><tr><td colspan="2">0.5</td></tr></table>', "merged_cells_unsupported"),
        (
            "<table><tr><td><table><tr><td>0.5</td></tr></table></td></tr></table>",
            "nested_table_unsupported",
        ),
        (
            "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td></tr></table>",
            "nonrectangular_table_unsupported",
        ),
        (
            '<table><tr><th id="h">dose</th><td headers="h">0.5</td></tr></table>',
            "explicit_header_ids_unsupported",
        ),
    ],
)
def test_unavailable_structure_is_machine_readable(format, markup, reason):
    block = document(markup, format).blocks[0]
    assert block.table is None
    assert block.position.table_status == "unavailable"
    assert block.position.table_capability.reason == reason
    assert block.position.table_capability.numeric_evidence is False


@pytest.mark.parametrize("format", ["xml", "html"])
def test_grid_limits_are_explicit(format):
    root = "article" if format == "xml" else "html"
    body = f"<{root}><table>" + "<tr><td>x</td></tr>" * 65 + f"</table></{root}>"
    result = parse(body.encode(), "application/xml" if format == "xml" else "text/html")
    assert (result.status, result.reason) == ("failed", "resource_limit")


def test_legacy_parsed_bytes_and_versions_are_unchanged():
    from pathlib import Path

    from packages.knowledge_index.models import ParsedDocument, canonical
    from packages.object_store import checksum

    raw = Path(__file__).with_name("fixtures").joinpath("legacy-inline-v1.json").read_bytes()
    old = ParsedDocument.model_validate_json(raw)
    assert checksum(canonical(old)) == checksum(raw)
    assert canonical(old) == raw
    assert old.parser_version == "1.0.0" and old.fidelity_policy is None
    assert old.blocks[0].text == "Threshold 0. 5 units; non negative ."
    new = document("<p>Threshold 0.<italic>5</italic> units; non<bold>negative</bold>.</p>")
    assert new.parser_version == "2.0.0"
    assert canonical(new) != raw


def test_legacy_chunk_strategy_cannot_label_repaired_output():
    from packages.knowledge_index.chunking import chunk
    from packages.knowledge_index.models import ChunkPolicy

    with pytest.raises(ValueError, match="legacy_chunk_policy_requires_replay"):
        chunk(document("<p>non<b>negative</b></p>"), ChunkPolicy(version="1.0.0"))


@pytest.mark.parametrize("format", ["xml", "html"])
def test_table_layout_text_is_not_silently_lost(format):
    block = document("<table>context<tr><td>0.5</td></tr></table>", format).blocks[0]
    assert "context" in block.text
    assert block.position.table_capability.reason == "table_layout_unsupported"
    assert block.position.table_capability.numeric_evidence is False


def test_html_title_rcdata_is_literal_not_reparsed_markup():
    doc = document(
        "<head><title>non&lt;b&gt;negative&lt;/b&gt;</title></head><p>content</p>", "html"
    )
    assert doc.title == "non<b>negative</b>"


@pytest.mark.parametrize("format", ["xml", "html"])
def test_wrapped_caption_and_long_inline_chunks(format):
    from packages.knowledge_index.chunking import chunk
    from packages.knowledge_index.models import ChunkPolicy

    caption = (
        "<fig><caption>non<b>negative</b> 0.<i>5</i></caption></fig>"
        if format == "xml"
        else "<figure><figcaption>non<b>negative</b> 0.<i>5</i></figcaption></figure>"
    )
    doc = document(
        caption + "<p>" + "Threshold 0.<b>5</b> mg; non<b>negative</b>. " * 8 + "</p>", format
    )
    assert doc.blocks[0].position.kind == "caption"
    blocks = chunk(doc, ChunkPolicy(max_chars=64, overlap_chars=8))
    assert len(blocks) > 3
    canonical_text = "\n".join(b.text for b in doc.blocks)
    assert "0. 5" not in canonical_text and "non negative" not in canonical_text
    for block in blocks:
        assert canonical_text[block.position.span_start : block.position.span_end] == block.text


@pytest.mark.parametrize("format", ["xml", "html"])
def test_embedded_table_is_never_flattened_as_prose(format):
    root = "article" if format == "xml" else "html"
    item = "list-item" if format == "xml" else "li"
    result = parse(
        f"<{root}><{item}>context<table><tr><td>0.5</td></tr></table></{item}></{root}>".encode(),
        "application/xml" if format == "xml" else "text/html",
    )
    assert (result.status, result.reason) == ("unsupported", "embedded_table_unsupported")


@pytest.mark.parametrize("body", [b"<sup>2</sup>", b"<article><sup><p>2</p></sup></article>"])
def test_xml_numeric_root_and_block_wrappers_are_unsupported(body):
    result = parse(body, "application/xml")
    assert (result.status, result.reason) == ("unsupported", "numeric_markup_unsupported")
