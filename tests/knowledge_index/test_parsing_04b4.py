"""Offline format, security and semantic chunking fixtures."""

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import ChunkPolicy, IndexConfig, ParsePolicy
from packages.knowledge_index.parsers import parse
from packages.knowledge_index.ports import FakeEmbedder
from packages.object_store import LocalObjectStore, checksum

XML = (
    b"<article><article-title>Synthetic study</article-title><sec><title>Results</"
    b"title><p>Glaucoma is not demonstrated. Retina negative.</p><list><list-item>"
    b"Do not infer a diagnosis.</list-item></list><table><tr><td>group</td><td>neg"
    b"ative</td></tr></table><fig><caption>Synthetic figure</caption></fig></sec><"
    b"/article>"
)
HTML = (
    b"<html><head><title>Synthetic guideline</title><script>EXFILTRATE()</script><"
    b'link href="https://example.invalid/x"/></head><body><h1>Guidance</h1><h2>Con'
    b"traindications</h2><p>Do not use in children.</p><ul><li>No benefit shown.</"
    b"li></ul><table><tr><td>negative</td></tr></table><figcaption>Figure one</fig"
    b'caption><iframe src="file:///etc/passwd"></iframe></body></html>'
)


def pdf_bytes(text=True, encrypted=False, javascript=False, pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=200, height=200)
        if text:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 10 100 Td (Glaucoma is not demonstrated.) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("synthetic-test")
    if javascript:
        writer.add_js('app.alert("fixture")')
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data,mime", [(XML, "application/xml"), (HTML, "text/html"), (pdf_bytes(), "application/pdf")]
)
def test_formats_have_real_text_positions(data, mime):
    result = parse(data, mime, language="en")
    assert result.status == "parsed", result
    assert all(
        b.position.span_end - b.position.span_start == len(b.text) for b in result.document.blocks
    )
    if mime == "application/pdf":
        assert result.document.blocks[0].position.page == 1
        assert "not demonstrated" in result.document.blocks[0].text
        assert result.document.blocks[0].position.table_status == "unavailable"
    else:
        assert any(
            b.position.kind == "table" and b.position.table_status == "structured"
            for b in result.document.blocks
        )
        assert any(b.position.kind == "caption" for b in result.document.blocks)
        assert any(b.position.kind == "list" for b in result.document.blocks)
    assert "EXFILTRATE" not in result.model_dump_json()


@pytest.mark.parametrize(
    "data,mime,status,reason",
    [
        (b"", "application/xml", "failed", "empty_document"),
        (b"plain text", "application/xml", "failed", "wrong_mime"),
        (XML, "application/pdf", "failed", "wrong_mime"),
        (HTML, "application/xml", "failed", "wrong_mime"),
        (XML, "text/html", "failed", "wrong_mime"),
        (b"PK\x03\x04compressed", "application/xml", "failed", "unsafe_document"),
        (
            b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]><article>&x;</article>',
            "application/xml",
            "failed",
            "unsafe_document",
        ),
        (
            b'<!DOCTYPE article SYSTEM "https://example.invalid/dtd"><article>x</article>',
            "application/xml",
            "failed",
            "unsafe_document",
        ),
        (
            b'<article xmlns:x="http://www.w3.org/2001/XInclude"><x:include href="file:///etc/passwd"/></article>',
            "application/xml",
            "failed",
            "unsafe_document",
        ),
        (b"<article/>", "application/xml", "no_text", "no_text_ocr_unsupported"),
        (pdf_bytes(False), "application/pdf", "no_text", "no_text_ocr_unsupported"),
        (pdf_bytes(encrypted=True), "application/pdf", "failed", "encrypted"),
        (pdf_bytes(javascript=True), "application/pdf", "failed", "unsafe_document"),
        (XML, "application/zip", "unsupported", "unsupported_mime"),
    ],
)
def test_rejected_and_empty_formats(data, mime, status, reason):
    result = parse(data, mime)
    assert (result.status, result.reason) == (status, reason)
    assert result.document is None


@pytest.mark.parametrize(
    "data,mime,policy",
    [
        (XML, "application/xml", ParsePolicy(max_bytes=10)),
        (XML, "application/xml", ParsePolicy(max_chars=20)),
        (XML, "application/xml", ParsePolicy(max_nodes=2)),
        (b"<a>" * 60 + b"x" + b"</a>" * 60, "application/xml", ParsePolicy()),
        (pdf_bytes(pages=2), "application/pdf", ParsePolicy(max_pages=1)),
    ],
)
def test_resource_limits(data, mime, policy):
    assert parse(data, mime, policy=policy).reason == "resource_limit"


def test_stable_semantic_chunk_boundaries_and_policy_versions():
    document = parse(XML, "application/xml").document
    policy = ChunkPolicy(max_chars=64, overlap_chars=8)
    first = chunk(document, policy)
    assert first == chunk(document, policy)
    assert any("not demonstrated" in b.text for b in first)
    assert any(b.position.section_path == ("Results",) for b in first)
    assert all(len(b.text) <= 64 for b in first)
    long = parse(
        b"<article><p>" + b"No benefit shown. " * 100 + b"</p></article>", "application/xml"
    ).document
    assert len(chunk(long, policy)) > 1
    assert all(
        b.position.span_end - b.position.span_start == len(b.text) for b in chunk(long, policy)
    )
    with pytest.raises(ValueError, match="chunk_count_limit"):
        chunk(long, ChunkPolicy(max_chars=64, overlap_chars=8, max_chunks=1))


def test_explicit_fts_only_and_provider_config():
    with pytest.raises(ValueError):
        IndexConfig(mode="fts_only")
    with pytest.raises(ValueError):
        IndexConfig(mode="hybrid")
    assert IndexConfig(mode="hybrid", embedder=FakeEmbedder().spec).reranker_mode == "disabled"
    assert (
        IndexConfig(
            mode="fts_only", vector_weight=0.0, vector_unavailable_reason="provider_unavailable"
        ).mode
        == "fts_only"
    )


def test_bounded_store_rejects_oversize_traversal_and_symlink(tmp_path):
    store = LocalObjectStore(tmp_path / "objects")
    uri = store.promote(store.stage(b"1234"), checksum(b"1234"))
    assert store.read_bounded(uri, 4) == b"1234"
    with pytest.raises(ValueError):
        store.read_bounded(uri, 3)
    with pytest.raises(ValueError):
        store.read_bounded("object://knowledge/objects/../../etc/passwd", 100)
    p = store._path(uri)
    p.unlink()
    p.symlink_to(tmp_path / "secret")
    with pytest.raises(OSError):
        store.read_bounded(uri, 4)


def test_utf16_entity_bomb_is_rejected():
    data = (
        '<!DOCTYPE article [<!ENTITY a "123456789"><!ENTITY b "&a;&a;&a;">]><article>&b;</article>'
    ).encode("utf-16")
    result = parse(data, "application/xml")
    assert result.status == "failed"


def test_pdf_expansion_limit():
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    stream = DecodedStreamObject()
    stream.set_data(b" " * 9_000_000)
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    buffer = io.BytesIO()
    writer.write(buffer)
    assert parse(buffer.getvalue(), "application/pdf").reason == "resource_limit"


def test_parser_timeout_is_a_resource_failure(monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert parse(XML, "application/xml").reason == "resource_limit"


def test_cli_help_runs_offline():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "apps.knowledge_document_index", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "published-query" in result.stdout


def test_expanded_position_metadata_cannot_overflow_parent_output():
    section = b"<sec><title>" + b"x" * 250 + b"</title>"
    data = b"<article>" + section * 30 + b"<p>x</p>" * 800 + b"</sec>" * 30 + b"</article>"
    assert parse(data, "application/xml").reason == "resource_limit"
