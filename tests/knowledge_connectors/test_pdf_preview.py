"""Synthetic PDF bytes; no external network or real medical material."""

import subprocess

import pytest

from apps import knowledge_literature_tools as app
from packages.knowledge_connectors.http import WireResponse
from packages.knowledge_index import pdf_preview
from tests.knowledge_connectors.test_literature_tools import MemoryRepository, recorded_factory
from tests.knowledge_index.test_parsing_04b4 import pdf_bytes


def test_exact_aao_pdf_fetch_has_page_locations_and_preserves_no_archive():
    repo = MemoryRepository()
    app.register(repo)
    data = pdf_bytes(pages=3)
    factory, transports = recorded_factory(
        WireResponse(200, (("content-type", "application/pdf"),), data)
    )
    entry = app.provider_config("aao").catalog[0]
    result = app.fetch(
        repo, "aao", entry.record_id, client_factory=factory, page_start=2, page_count=1
    )
    assert result["status"] == "succeeded" and result["text_extracted"]
    assert not result["archived"]
    assert result["page_count"] == 3 and result["pages_read"] == [2]
    assert result["more_pages_available"] and not result["excerpt_truncated"]
    assert "[PDF page 2]" in result["excerpt"]
    assert "Glaucoma is not demonstrated." in result["excerpt"]
    assert result["checksum"] and transports[0].requests[0][0].url == entry.url


def test_pdf_char_bound_and_last_page():
    result = pdf_preview.preview(pdf_bytes(pages=2), page_start=2, max_chars=20)
    assert result["status"] == "succeeded" and len(result["excerpt"]) == 20
    assert result["excerpt_truncated"] and not result["more_pages_available"]


@pytest.mark.parametrize(
    "data,options,reason",
    [
        (b"<html>Just a moment...</html>", {}, "wrong_mime"),
        (b"%PDF-broken", {}, "parse_failed"),
        (b"%PDF-" + b"x" * pdf_preview.MAX_BYTES, {}, "resource_limit"),
        (pdf_bytes(encrypted=True), {}, "encrypted"),
        (pdf_bytes(text=False), {}, "no_text_ocr_unsupported"),
        (pdf_bytes(), {"page_start": 2}, "page_out_of_range"),
        (pdf_bytes(pages=201), {}, "resource_limit"),
    ],
)
def test_pdf_failure_not_empty_success(data, options, reason):
    result = pdf_preview.preview(data, **options)
    assert result["status"] == "failed" and result["parse_reason"] == reason
    assert not result["text_extracted"] and "excerpt" not in result


@pytest.mark.parametrize(
    "options",
    [
        {"page_start": 0},
        {"page_start": True},
        {"page_count": 11},
        {"max_chars": 16001},
    ],
)
def test_limits_rejected_before_parser(options):
    with pytest.raises(ValueError, match="invalid_pdf_preview_limits"):
        pdf_preview.preview(pdf_bytes(), **options)


def test_worker_timeout_is_bounded_failure(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 20
        raise subprocess.TimeoutExpired("synthetic", 20)

    monkeypatch.setattr(pdf_preview.subprocess, "run", timeout)
    assert pdf_preview.preview(pdf_bytes())["parse_reason"] == "resource_limit"


def test_aao_html_challenge_cannot_be_successful_pdf():
    repo = MemoryRepository()
    app.register(repo)
    factory, _ = recorded_factory(
        WireResponse(200, (("content-type", "text/html"),), b"Enable JavaScript and cookies")
    )
    result = app.fetch(repo, "aao", "AAO_POAG_PPP_2026", client_factory=factory)
    assert result["status"] == "failed" and "excerpt" not in result
